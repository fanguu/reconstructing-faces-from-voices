import os
import torch
import shutil
import numpy as np
import torch.nn.functional as F
import logging

from PIL import Image
from scipy.io import wavfile
from torch.utils.data.dataloader import default_collate
from vad import read_wave, write_wave, frame_generator, vad_collector


class Logger(object):
    def __init__(self,save_path,date):

        self.logger = logging.getLogger('lossesLogger')
        self.logFile = os.path.join('logs', save_path)
        if not os.path.exists(self.logFile):
            os.makedirs(self.logFile)
            # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        handler = logging.FileHandler(self.logFile + '/logFile_{0}.log'.format(date))
        handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(hdlr=handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)
        self.logger.info("starting logger model...")

    def info(self, out):
        self.logger.info(out)


class Meter(object):
    # Computes and stores the average and current value
    def __init__(self, name, display, fmt=':f'):
        self.name = name
        self.display = display
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    # 返回对象的描述信息
    def __str__(self):
        fmtstr = '{name}:{' + self.display  + self.fmt + '},'
        return fmtstr.format(**self.__dict__)

def get_collate_fn(nframe_range):
    # collate_fn这个函数的输入就是一个batch size的数据, 音频数据为变长batch
    def collate_fn(batch):
        min_nframe, max_nframe = nframe_range
        assert min_nframe <= max_nframe, 'the wrong value in nframe'
        num_frame = np.random.randint(min_nframe, max_nframe+1)
        pt = np.random.randint(0, max_nframe-num_frame+1)
        batch = [(item[0][..., pt:pt+num_frame], item[1]) for item in batch]
        b1 = np.array(batch[0])
        # print(b1.shape)
        # print(b1.size)
        return default_collate(batch)
    return collate_fn

def cycle(dataloader):
    while True:
        for data, label in dataloader:
            yield data, label

def save_model(net, model_path):
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
       os.makedirs(model_dir)
    torch.save(net.state_dict(), model_path)

def rm_sil(voice_file, vad_obj):
    """
       This code snippet is basically taken from the repository
           'https://github.com/wiseman/py-webrtcvad'

       It removes the silence clips in a speech recording
    """
    audio, sample_rate = read_wave(voice_file)
    frames = frame_generator(20, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 20, 50, vad_obj, frames)

    if os.path.exists('tmp/'):
       shutil.rmtree('tmp/')
    os.makedirs('tmp/')

    wave_data = []
    for i, segment in enumerate(segments):
        segment_file = 'tmp/' + str(i) + '.wav'
        write_wave(segment_file, segment, sample_rate)
        wave_data.append(wavfile.read(segment_file)[1])
    shutil.rmtree('tmp/')

    if wave_data:
       vad_voice = np.concatenate(wave_data).astype('int16')
    return vad_voice

def get_fbank(voice, mfc_obj):
    # Extract log mel-spectrogra
    fbank = mfc_obj.sig2logspec(voice).astype('float32')

    # Mean and variance normalization of each mel-frequency 
    fbank = fbank - fbank.mean(axis=0)
    fbank = fbank / (fbank.std(axis=0)+np.finfo(np.float32).eps)

    # If the duration of a voice recording is less than 10 seconds (1000 frames),
    # repeat the recording until it is longer than 10 seconds and crop.
    full_frame_number = 1000
    init_frame_number = fbank.shape[0]
    while fbank.shape[0] < full_frame_number:
          fbank = np.append(fbank, fbank[0:init_frame_number], axis=0)
          fbank = fbank[0:full_frame_number,:]
    return fbank


def voice2face(e_net, g_net, voice_file, vad_obj, mfc_obj, GPU=True):
    vad_voice = rm_sil(voice_file, vad_obj)
    fbank = get_fbank(vad_voice, mfc_obj)
    fbank = fbank.T[np.newaxis, ...]
    fbank = torch.from_numpy(fbank.astype('float32'))
    
    if GPU:
        fbank = fbank.cuda()
    embedding = e_net(fbank)
    embedding = F.normalize(embedding)
    face = g_net(embedding)
    return face
