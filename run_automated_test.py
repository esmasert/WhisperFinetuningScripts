import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import glob
from tqdm import tqdm
import whisper
import transformers
from multiple_datasets.hub_default_utils import convert_hf_whisper
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import pandas as pd
import csv
from termcolor import colored as cl
import IPython

import jiwer
import re
from jiwer import cer

import unicodedata as ud
import jiwer
import re
import string
from termcolor import colored as cl
import torch
torch.cuda.is_available()


import markdown

md = markdown.Markdown(extensions=["markdown_cjk_spacing.cjk_spacing"],
    extension_configs={'markdown_cjk_spacing.cjk_spacing':
            {'segment_break': True}})


def wer_C(ref, hyp ,debug=False):
    r = ref.split()
    h = hyp.split()
    #costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    DEL_PENALTY=1 # Tact
    INS_PENALTY=1 # Tact
    SUB_PENALTY=1 # Tact
    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r)+1):
        costs[i][0] = DEL_PENALTY*i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    compares = []
    if debug:
        lines = []
        compares = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i-=1
            j-=1
            if debug:
                lines.append("OK\t" + r[i]+"\t"+h[j])
                compares.append(colored(0, 0, 0, h[j]))
        elif backtrace[i][j] == OP_SUB:
            numSub +=1
            i-=1
            j-=1
            if debug:
                lines.append("SUB\t" + r[i]+"\t"+h[j])
                compares.append(colored(0, 255, 0, h[j]) +  colored(0, 0, 0, f'({r[i]})'))
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j-=1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
                compares.append(colored(0, 0, 255, h[j]))
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i-=1
            if debug:
                lines.append("DEL\t" + r[i]+"\t"+"****")
                compares.append(colored(255, 0, 0, r[i]))
    if debug:
        # print("OP\tREF\tHYP")
        # lines = reversed(lines)
        # for line in lines:
        #     print(line)

        compares = reversed(compares)
        for line in compares:
            print(line, end=" ")
        # print("Ncor " + str(numCor))
        # print("Nsub " + str(numSub))
        # print("Ndel " + str(numDel))
        # print("Nins " + str(numIns))
    wer_result = round( (numSub + numDel + numIns) / (float) (len(r)), 3)
    return {'WER':wer_result, 'Cor':numCor, 'Sub':numSub, 'Ins':numIns, 'Del':numDel}, compares


def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def strike(text, color=None):
    if color:
        return colored(0, 255, 0, ''.join([u'\u0336{}'.format(c) for c in text]))

    else:
        return  colored(0, 0, 0, ''.join([u'\u0332{}'.format(c) for c in text]))

def load_file(text_path):
    with open(text_path) as f:
        lines = (line.rstrip() for line in f)
        lines = list(line for line in lines if line)

    # remove time stamps and other unwanted lines
    text_input = []

    remove_words = ['Subject:','From:','Date:','To:']
    for line in lines:
        if not any(remove_word in line for remove_word in remove_words):
            text_input.append(line)

    # remove blank lines and lines with one word on them
    text_input_clean = []
    for i in range(len(text_input)):
        if len(text_input[i].split(' ')) > 1:
            text_input_clean.append(text_input[i])
    return text_input_clean



# loading datasets
def load_lang_df(lang):
    metadata_file = f"validation.json"
    df = pd.read_json(metadata_file, lines=True, orient="records")
    df.name=f"{lang} dataset"
    return df

target_dataset = load_lang_df("mandarin Data")

audio_files=[]
gt=[]

for audioFileName, textt in tqdm(zip(target_dataset['audio_filepath'], target_dataset['text']), total = len(target_dataset['audio_filepath'])):

    audio_files.append(audioFileName)
    gt.append(textt)


newgt=[]
for agt in gt:
    ntext=agt.replace("。","")
    ntext=ntext.replace("。","")
    ntext=ntext.replace("：","")
    ntext=ntext.replace(",","")
    ntext=ntext.replace("，","")
    ntext=ntext.replace("，","")
    ntext=ntext.replace("！","")
    ntext=ntext.replace("é","e")
    ntext=ntext.replace("-","")
    ntext=ntext.replace("\n","")
    ntext = md.convert(ntext)
    ntext = ntext.replace('<p>', '').replace('</p>', '')
    newgt.append(ntext)

#print(newgt)
#print(len(newgt))

nngt =[]
letters = "abcdefghijklmnopqrstuwvxyz"

normalizer = BasicTextNormalizer()
basic_norm_str = normalizer("a.b")
basic_norm_str

for row in newgt:
    nngttemp=""
    for charact in row:
        if normalizer(charact)  not in letters:
            nngttemp+=normalizer(charact)
            nngttemp+=" "
        else:
            nngttemp+=normalizer(charact)
    nngt.append(nngttemp)

nngt

import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


checkpointFolders = glob.glob("/checkpoint_folder/check*")

sort_nicely(checkpointFolders)

print(len(checkpointFolders))

print(checkpointFolders)

letters = "abcdefghijklmnopqrstuwvxyz"
loanGtWords = []
loanPredWords = []

normalizer = BasicTextNormalizer()
basic_norm_str = normalizer("a.b")
basic_norm_str

idx=0
for onegt in nngt:
    #print(onegt)
    temp=[]
    words = onegt.split()

    for aword in words:
        for i in letters:
            if i in aword:
                temp.append(normalizer(aword))
                break

    loanGtWords+=temp

    idx+=1


for model_path in checkpointFolders:
        
    print(model_path)
    
    basemn = model_path.split("/")[-1]
    newmodelName = "/" + basemn + "_mandarin.pt"
    
    if os.path.isfile(model_path + newmodelName):
        print("MODEL EXIST")
        print(newmodelName)
    else:
        print("MODEL NOT EXIST")
        print(newmodelName)
        # Convert Huggingface model to OpenAi whisper model
        convert_hf_whisper(model_path, model_path + newmodelName)
            
    model = whisper.load_model(model_path + newmodelName)
    
    pred=[]
    files_segments=[]

    print(model_path + newmodelName)

    for x in tqdm(audio_files):
        audio = whisper.load_audio(x)
        result = model.transcribe(audio, language="zh", word_timestamps=True, beam_size=5, best_of=5)
        pred.append(result['text'])
        files_segments.append(result['segments'])


    print(model_path)
    for apred in pred:
        print(apred)
        print("\n")
        
    newpred=[]
    for apred in pred:
        ntext=apred.replace("。","")
        ntext=ntext.replace("。","")
        ntext=ntext.replace("：","")
        ntext=ntext.replace(",","")
        ntext=ntext.replace("，","")
        ntext=ntext.replace("，","")
        ntext=ntext.replace("！","")
        ntext=ntext.replace("é","e")
        ntext=ntext.replace("-","")
        ntext=ntext.replace("\n","")
        ntext = md.convert(ntext)
        ntext = ntext.replace('<p>', '').replace('</p>', '') 
        newpred.append(ntext)

    
    nnpred = []
    letters = "abcdefghijklmnopqrstuwvxyz"

    for row in newpred:
        nnpredtemp=""
        for charact in row:
            if normalizer(charact) not in letters:
                nnpredtemp+=normalizer(charact)
                nnpredtemp+=" "
            else:
                nnpredtemp+=normalizer(charact)
                
        nnpred.append(nnpredtemp)

    print(nnpred)
    
    jiwerResultsFilePath = "results_ckpts.txt"
    jiwerResultsFile = open(jiwerResultsFilePath,"a")
    jiwerResultsFile.write("\n")
    jiwerResultsFile.write("Results of: \n")
    jiwerResultsFile.write(newmodelName)
    jiwerResultsFile.write("\n")

    # Test model
    cntt=0
    allwer=0
    allcor=0
    allins=0
    allsub=0
    alldel=0
    
    for x,y in zip(nngt,nnpred):
        
        jiwerResultsFile.write(audio_files[cntt])
        jiwerResultsFile.write("\n")

        ref=x
        hyp=y

        output, compares = wer_C(ref, hyp ,debug=False)
        
        werr=output['WER']
        gtlen=len(ref.split())
        cor=output['Cor']/gtlen * 100.0
        dell=output['Del']/gtlen * 100.0
        subb=output['Sub']/gtlen * 100.0
        inss=output['Ins']/gtlen * 100.0

        print("COR:",round(cor,2))
        print("DEL:",round(dell,2))
        print("SUB:",round(subb,2))
        print("INS:",round(inss,2))
        print("\n")
        
        jiwerResultsFile.write("WER: "+ str(round(werr*100,2)))
        jiwerResultsFile.write("  |  COR: "+ str(round(cor,2)))
        jiwerResultsFile.write("  |  DEL: "+ str(round(dell,2)))
        jiwerResultsFile.write("  |  SUB: "+ str(round(subb,2)))
        jiwerResultsFile.write("  |  INS: "+ str(round(inss,2)))
        
        jiwerResultsFile.write("\n")
        
        
        allwer+=werr*100
        allcor+=cor
        allins+=inss
        allsub+=subb
        alldel+=dell
        
        cntt+=1
        
    avgcor = allcor/len(audio_files)
    avgwer = allwer/len(audio_files)
    avgins = allins/len(audio_files)
    avgsub = allsub/len(audio_files)
    avgdel = alldel/len(audio_files)

    jiwerResultsFile.write("\n")
    jiwerResultsFile.write("AVERAGE RESULTS:")
    jiwerResultsFile.write("\n")
    jiwerResultsFile.write("WER: "+ str(round(avgwer,2)))
    jiwerResultsFile.write("  |  COR: "+ str(round(avgcor,2)))
    jiwerResultsFile.write("  |  DEL: "+ str(round(avgdel,2)))
    jiwerResultsFile.write("  |  SUB: "+ str(round(avgsub,2)))
    jiwerResultsFile.write("  |  INS: "+ str(round(avgins,2)))

    jiwerResultsFile.write("\n")
    
    print("AVG-COR:",round(avgcor,2))
    print("AVG-DEL:",round(avgdel,2))
    print("AVG-SUB:",round(avgsub,2))
    print("AVG-INS:",round(avgins,2))
    print("AVG-WER:",round(avgwer,2))
    print("\n\n")

    loanPredWords=[]
    idx=0
    for onepred in nnpred:
        temp=[]
        words = onepred.split()

        for aword in words:
            for i in letters:
                if i in aword:
                    temp.append(normalizer(aword))
                    #print(aword)
                    break

        loanPredWords+=temp

        idx+=1

    print("total GT loan_words:",len(loanGtWords))
    print("total PRED loan_words:",len(loanPredWords))
    
    jiwerResultsFile.write("GT lw: " + str(len(loanGtWords)))
    jiwerResultsFile.write("\nPRED lw: " + str(len(loanPredWords)))
   

    tp=0
    fp =0

    for awrd in loanPredWords:
        if awrd in loanGtWords:
            tp+=1
            #loanPredWords.pop(loanPredWords.index(awrd))
        else:
            print(awrd)
            fp+=1
            #loanPredWords.pop(loanPredWords.index(awrd))

    fn=0

    for awrd in loanGtWords:
        if awrd in loanPredWords:
            pass
            loanPredWords.pop(loanPredWords.index(awrd))
        else:
            print(awrd)
            fn+=1


    print("tp:",tp)
    print("fp:",fp)
    print("fn:",fn)

    jiwerResultsFile.write("\ntp: " + str(tp))
    jiwerResultsFile.write("  |  fp: " + str(fp))
    jiwerResultsFile.write("  |  fn: " + str(fn))
    jiwerResultsFile.write("\n")
    
    jiwerResultsFile.close()


#
