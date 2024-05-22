
import pandas as pd
import os.path
import os
import codecs
from subprocess import PIPE, run
from tqdm import tqdm
import json
import sys

print(sys.argv)

check_json_filePath = "/home/" + sys.argv[1]

print(check_json_filePath)

# loading datasets
target_dataset = pd.read_json(check_json_filePath, lines=True, orient="records")

# total number of records
print(f"There are {target_dataset.shape[0]} records.")

for fn in tqdm(target_dataset['audio_filepath']):

    if "common_voice" in fn:
        base_filename = fn.split("/")[-1]
        newfn = "/home/wavs/" + base_filename

    elif "nst_danish" in fn:
        base_filename = fn.split("/")[-1]
        newfn = "/home/all_wavs/" + base_filename

    else:
        newfn = fn
    
    resulBaseFileName = newfn.split("/")[-1].split(".wav")[0]

    print(newfn)
    print(resulBaseFileName)
    
    cmdGenerateWav2Gsm = "sox " + newfn + " -r 8000 -c 1 /home/gsm_files/" + resulBaseFileName + ".gsm"
    print(cmdGenerateWav2Gsm)
    os.system(cmdGenerateWav2Gsm)
    
    createdgsmfilename= "/home/gsm_files/" + resulBaseFileName + ".gsm"
    
    cmdGenerateGsm2Wav = "ffmpeg -i " + createdgsmfilename + " -acodec pcm_s16le -ac 1 -ar 16000 /home/gsm2wavs/" + resulBaseFileName + ".wav"
    print(cmdGenerateGsm2Wav)
    os.system(cmdGenerateGsm2Wav)
    
    #break










#