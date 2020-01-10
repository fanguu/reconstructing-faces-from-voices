#!/bin/bash 
conda create --name voice2face --file requirements.txt
source activate voice2face
pip3 install -q webrtcvad
