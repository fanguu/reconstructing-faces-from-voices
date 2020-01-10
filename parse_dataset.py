import os
import csv
from config import DATASET_PARAMETERS

def get_RAVEDSS_csv(data_dir, data_ext):
    data_list = []
    list_name ={"wav":"wave","png":"image"}
    headers = ['actor_ID','gender','vocal_channel','emotion','emotion_intensity','{}_path'.format(list_name[data_ext])]
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    # read data directory
    for root, dirs, filenames in os.walk(data_dir):      # 根目录, 子目录, 文件名
        for filename in filenames:
            if filename.endswith("wav"):              # 校验文件后缀名
                wave_path = os.path.join(root, filename)
                flag = filename[:-4].split('-')
                if flag[0] == '01':
                    gend = "female" if int(flag[6])%2 else "male"
                    data_list.append({'actor_ID':flag[6], 'gender':gend,'vocal_channel':flag[1],'emotion':flag[2],
                                      'emotion_intensity':flag[3],'wave_path':wave_path})
                    print("path:{0},actor:{1}".format(wave_path,flag[6]))
            if filename.endswith("png"):
                image_path = os.path.join(root, filename)
                flag = root.split('/')[-1].split('-')
                if flag[0] == '01':
                    gend = "female" if int(flag[6])%2 else "male"
                    data_list.append({'actor_ID':flag[6], 'gender':gend,'vocal_channel':flag[1],'emotion':flag[2],
                                    'emotion_intensity':flag[3],'image_path':image_path})
                    print("path:{0},actor:{1}".format(image_path,flag[6]))

    print("{} number:{}".format(list_name[data_ext], len(data_list)))

    with open('{}_list.csv'.format(list_name[data_ext]),'w') as f:
        f_scv = csv.DictWriter(f,headers)
        f_scv.writeheader()
        f_scv.writerows(data_list)


def parse_metafile(meta_file):
    with open(meta_file, 'r') as f:
        lines = f.readlines()[1:]
    celeb_ids = {}
    for line in lines:
        ID, name, _, _, _ = line.rstrip().split('\t')    # 去除行末字符，按照‘\t’进行划分行数据
        celeb_ids[ID] = name
    return celeb_ids


def get_labels(voice_list, face_list):
    voice_names = {item['name'] for item in voice_list}
    face_names = {item['name'] for item in face_list}
    names = voice_names & face_names

    # temp = []
    # for item in voice_list:
    #     if item['name'] in names:     # 查询list names中是否包含item name
    #         temp.append(item)
    # voice_list = temp

    #  通过列表推导式 保留同类项
    voice_list = [item for item in voice_list if item['name'] in names]
    face_list = [item for item in face_list if item['name'] in names]

    names = list(names)
    label_dict = dict(zip(names, range(len(names))))
    for item in voice_list+face_list:                # 增加序号label_id
        item['label_id'] = label_dict[item['name']]
    return voice_list, face_list, len(names)
    

def get_files(data_dir, data_ext, celeb_ids, split):
    data_list = []
    # read data directory
    for root, dirs, filenames in os.walk(data_dir):      # 根目录, 子目录, 文件名
        for filename in filenames:
            if filename.endswith(data_ext):              # 校验文件后缀名, npy
                filepath = os.path.join(root, filename)
                # so hacky, be careful! 
                folder = filepath[len(data_dir):].split('/')[1]
                celeb_name = celeb_ids.get(folder, folder)
                # if celeb_name.startswith(tuple(split)):
                data_list.append({'filepath': filepath, 'name': celeb_name})
    return data_list


def get_dataset(data_params):
    celeb_ids = parse_metafile(data_params['meta_file'])    # VoxCeleb1 ID    VGGFace1 ID
    
    voice_list = get_files(data_params['voice_dir'],
                                   data_params['voice_ext'],
                                   celeb_ids,
                                   data_params['split'])
    face_list = get_files(data_params['face_dir'],
                                  data_params['face_ext'],
                                  celeb_ids,
                                  data_params['split'])
    return get_labels(voice_list, face_list)


def get_RAVDESS_dataset(data_params):
    voice_list = []
    actor_num = []
    with open(data_params['wave_file']) as f:
        lines = f.readlines()[1:]
        for line in lines:
            # print(line)
            actor_ID,gender,vocal_channel,emotion,emotion_intensity,wave_path = line.split(',')
            actor_num.append(int(actor_ID))
            voice_list.append({'filepath': wave_path, 'name': actor_ID, 'emotion': emotion})

    face_list = []
    with open(data_params['image_file']) as f:
        lines = csv.reader(f)
        head = next(lines)
        for line in lines:
            actor_ID, gender, vocal_channel, emotion, emotion_intensity, image_path = line.split(',')
            face_list.append({'filepath': image_path, 'name': actor_ID, 'emotion': emotion})

    return voice_list, face_list, max(actor_num)

if __name__ == '__main__':
    get_RAVDESS_dataset(DATASET_PARAMETERS)
    # data_dir = 'data/RAVDESS/wave-24'
    # get_RAVEDSS_csv(data_dir,'wav')
    # data_dir = 'data/RAVDESS/image-24'
    # get_RAVEDSS_csv(data_dir, 'png')