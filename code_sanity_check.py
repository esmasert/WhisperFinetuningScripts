import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import glob
from tqdm import tqdm
import pandas as pd
import os.path
import os
from tqdm import tqdm
import csv
import json
import codecs
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import jiwer
import re
from jiwer import wer

from faster_whisper import WhisperModel

# loading datasets
def load_lang_df(lang):
    metadata_file = f"sub1.json"
    df = pd.read_json(metadata_file, lines=True, orient="records")
    df.name=f"{lang} dataset"
    return df

target_dataset = load_lang_df("Danish Data")

# total number of records
print(f"There are {target_dataset.shape[0]} records in the {target_dataset.name}")

normalizer = BasicTextNormalizer()
model = WhisperModel("medium")



sanity_check_results = codecs.open("sanity_check_results.txt", "a")
under_05_file = codecs.open("under_05_files.txt", "a")

for audioFileName, textt in tqdm(zip(target_dataset['audio_filepath'], target_dataset['text']), total = len(target_dataset['audio_filepath'])):

    allPred=""
    
    segments, info = model.transcribe(audioFileName)
    for segment in segments:
        #print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        allPred += segment.text

    reference = normalizer(textt.lower()).strip()
    hypothesis = normalizer(allPred.lower()).strip()

    #print("reference:",reference)
    #print("hypothesis:",hypothesis)

    if len(reference)>1:
        
        wer_rate = wer(reference, hypothesis)
        
        x=jiwer.compute_measures(reference,hypothesis)
        
        #print("PRED:")
        #print(hypothesis)
        #print("GT:")
        #print(reference)
        
        gtlen=len(reference.split())
        wer_r=x["wer"]
        cor=x["hits"]/gtlen * 100.0
        sub=x["substitutions"]/gtlen * 100.0
        ins=x["insertions"]/gtlen * 100.0
        dele=x["deletions"]/gtlen * 100.0
        #print(audioFileName)
        #print({"wer":wer_r,"correct %":cor,"substitutions %":sub,"insertions %":ins,"deletions %":dele})
        
        #print("wer_rate:",wer_rate)
    
        sanity_check_results.write("AUDIO FILENAME: ")
        sanity_check_results.write(audioFileName)
        sanity_check_results.write(" |  WER RATE: ")
        sanity_check_results.write(str(round(wer_rate,2)))
        sanity_check_results.write(" |  DETAILED WER : ")
        sanity_check_results.write("wer:" + str(round(wer_r,2)) + " | correct %:" + str(round(cor,2)) + " | substitutions %: " + str(round(sub,2)) + " | insertions %: "  + str(round(ins,2)) + " | deletions %: " + str(round(dele,2)))      
        sanity_check_results.write(" |  gt text: ")
        sanity_check_results.write(reference)
        sanity_check_results.write(" |  pred text: ")
        sanity_check_results.write(hypothesis)
        sanity_check_results.write("\n")
    
        if wer_r < 0.5:
            under_05_file.write("AUDIO FILENAME: ")
            under_05_file.write(audioFileName)
            under_05_file.write(" |  WER RATE: ")
            under_05_file.write(str(round(wer_rate,2)))
            under_05_file.write(" |  gt text: ")
            under_05_file.write(reference)
            under_05_file.write(" |  pred text: ")
            under_05_file.write(hypothesis)
            under_05_file.write("\n")
        
    else:
        print("EMPTY GT")
        print(audioFileName)
        print("\n")

    #break        
        
sanity_check_results.close()
under_05_file.close()

    
        
