import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"

transformers_cache = './cache/transformers_cache/'
dataset_cache = './cache/datasets_cache/'
wandb_cache = './cache/wandb_cache/'

import datasets
from pathlib import Path
datasets.config.DOWNLOADED_DATASETS_PATH = Path(dataset_cache)

from datasets import Dataset, IterableDatasetDict, load_dataset, interleave_datasets, Audio, concatenate_datasets
import evaluate
import torch
import string 
from dataclasses import dataclass
from typing import *

import glob
import os
from tqdm import tqdm
import torch
import whisper

from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import WhisperTokenizer
from transformers import WhisperFeatureExtractor
import wandb
from IPython.display import clear_output

import numpy as np
from huggingface_hub import notebook_login

from transformers import TrainerCallback
from transformers.integrations import WandbCallback
from transformers.trainer_pt_utils import IterableDatasetShard
from torch.utils.data import IterableDataset
from datasets import load_dataset, Audio
from pathlib import Path
import numpy as np
import holoviews as hv
import panel as pn
import tempfile
import librosa
from bokeh.resources import INLINE
hv.extension("bokeh", logo=False)

from io import StringIO
import pandas as pd
import warnings
import jiwer
import jiwer
from jiwer import cer
from datasets import load_dataset, load_metric
warnings.filterwarnings('ignore')

torch.cuda.is_available()
from audiomentations import AddBackgroundNoise, PolarityInversion, Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, ApplyImpulseResponse,TimeMask, Mp3Compression

clear_output()

getFiles = glob.glob("/new_training/*")
if len(getFiles)<2:
    print("Training files not found!")
    exit()
else:
    print("Training scripts exist!")
    # loading datasets
    df = pd.read_json(getFiles[0], lines=True, orient="records")

    check_file = os.path.isfile(df['audio_filepath'][0])
    if check_file==True:
        print("Training files exist!")
    else:
        exit()
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE IS:")
print(device)

os.environ['TRANSFORMERS_CACHE'] = transformers_cache
os.environ['HF_HOME'] = dataset_cache
os.environ['WANDB_CACHE_DIR'] = wandb_cache

model_name_or_path = "openai/whisper-medium"
language = "Chinese"
language_abbr = "zh"
task = "transcribe"

import multiprocessing

preprocessing_num_workers=multiprocessing.cpu_count()
print(preprocessing_num_workers)

run = wandb.init(dir=wandb_cache,project="whisper_medium_zh")

def load_data_splits(is_streaming=False, stopping_strategy="all_exhausted"):
    raw_datasets = {}
    raw_datasets["train"] = load_dataset("/new_training/", split="train", streaming=is_streaming, cache_dir=dataset_cache)
    raw_datasets["test"] = load_dataset("/new_training/", split="test", streaming=is_streaming, cache_dir=dataset_cache)
    raw_datasets["validation"] = load_dataset("/new_training/", split="validation", streaming=is_streaming, cache_dir=dataset_cache)
    return raw_datasets

print('Loading Dataset.....')
vectorized_datasets = load_data_splits()

for x in vectorized_datasets['train']:
    print(x)
    break

from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

#https://iver56.github.io/audiomentations/waveform_transforms/add_background_noise/

from audiomentations import (
    AddBackgroundNoise,
    AddGaussianNoise,
    Compose,
    Gain,
    OneOf,
    PitchShift,
    PolarityInversion,
    TimeStretch,
)

musan_dir = "/musan"

# define augmentation
augment_waveform = Compose(
    [
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2, leave_length_unchanged=False),
        Gain(min_gain_in_db=-2, max_gain_in_db=2, p=0.1),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
        OneOf(
            [
                AddBackgroundNoise(sounds_path=musan_dir, min_snr_in_db=1.0, max_snr_in_db=2.0, noise_transform=PolarityInversion(), p=0.9),
                AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=0.4),
            ],
            p=1.0,
        ),
    ]
)

def augment_dataset(batch):

    audio = batch["audio"]['array']
    # apply augmentation
    augmented_audio = augment_waveform(samples=np.array(audio), sample_rate=16000)

    batch["audio"]['array'] = augmented_audio

    return batch


from transformers.models.whisper.english_normalizer import BasicTextNormalizer

do_augmentation=True

def prepare_audio(batch):
    audio, sr = librosa.load(batch["audio_filepath"],sr=16000)#, offset=start, duration=dur)
    batch["audio"]['array'] = audio
    batch["audio"]['sampling_rate']=16000
    return batch

vectorized_datasets['train'] = vectorized_datasets['train'].map(prepare_audio,num_proc=preprocessing_num_workers, desc="add audio array train dataset")#.with_format("torch")
vectorized_datasets['test'] = vectorized_datasets['test'].map(prepare_audio,num_proc=preprocessing_num_workers, desc="add audio array test dataset")#.with_format("torch")
vectorized_datasets['validation'] = vectorized_datasets['validation'].map(prepare_audio,num_proc=preprocessing_num_workers, desc="add audio array validation dataset")#.with_format("torch")

normalizer = BasicTextNormalizer()

import re
def prepare_dataset(batch):
    # load and (possibly) resample audio data to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    
    # optional pre-processing steps
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    transcription = batch["text"]#.replace('<si>','<|nospeech|>')    
    batch["labels_length"] = len(processor.tokenizer(transcription, add_special_tokens=False).input_ids)
    
    return batch

print("Preparing Data......")
vectorized_datasets['train'] = vectorized_datasets['train'].map(prepare_dataset,num_proc=None, desc="preparing train dataset")
vectorized_datasets['test'] = vectorized_datasets['test'].map(prepare_dataset,num_proc=None, desc="preparing test dataset")
vectorized_datasets['validation'] = vectorized_datasets['validation'].map(prepare_dataset,num_proc=None, desc="preparing validation dataset")
if do_augmentation:
    print("Augmenting Data......")
    # augment training data
    augmented_raw_training_dataset = vectorized_datasets["train"].map(
        augment_dataset, num_proc=preprocessing_num_workers, desc="augment train dataset"
    )
    # combine
    vectorized_datasets["train"] = concatenate_datasets([vectorized_datasets["train"], augmented_raw_training_dataset])

#vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(seed=0)
#vectorized_datasets["test"] = vectorized_datasets["test"]#.shuffle(seed=0)

max_input_length = 30.0
max_lbl_length = 400

def is_audio_in_length_range(length):
    """Filter inputs with zero input length or longer than 30s"""
    return 0 < length < max_input_length
    
def filter_labels(labels_length):
    """Filter empty label sequences"""
    return labels_length < max_lbl_length

print('==========================')
print(len(vectorized_datasets["train"]))

vectorized_datasets["train"] = vectorized_datasets["train"].filter(
    is_audio_in_length_range,
    input_columns=["input_length"],
)

vectorized_datasets["train"] = vectorized_datasets["train"].filter(
    filter_labels,
    input_columns=["labels_length"],
)

vectorized_datasets["test"] = vectorized_datasets["test"].filter(
    filter_labels,
    input_columns=["labels_length"],
)

print(len(vectorized_datasets["train"]))
print('==========================')
    
    
# removing empty strings

min_input_length = 1
def remove_empty_strings(text):
    text_length = len(text)
    """Filter inputs with zero input length or longer than 30s"""
    return min_input_length < text_length

print('==========================Checking Empty Strings==========================')
print(len(vectorized_datasets["train"]))
print(len(vectorized_datasets["test"]))
print(len(vectorized_datasets["validation"]))

vectorized_datasets["train"] = vectorized_datasets["train"].filter(
    remove_empty_strings,
    input_columns=["text"],
)

vectorized_datasets["test"] = vectorized_datasets["test"].filter(
    remove_empty_strings,
    input_columns=["text"],
)

vectorized_datasets["validation"] = vectorized_datasets["validation"].filter(
    remove_empty_strings,
    input_columns=["text"],
)

print("After Filter:")
print(len(vectorized_datasets["train"]))
print(len(vectorized_datasets["test"]))
print(len(vectorized_datasets["validation"]))
print('==========================Checking Empty Strings==========================')

    
model_name = "openai/whisper-medium"
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="zh", task="transcribe")

from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(model_name, language="zh", task="transcribe")
    
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = load_metric("cer")

# evaluate with the 'normalised' CER
do_normalize_eval = False

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if do_normalize_eval:
        pred_str = [normalizer(pred) for pred in pred_str]
        label_str = [normalizer(label) for label in label_str]

    errorCER = cer(label_str, pred_str)
    
    return {"cer": errorCER}

from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(model_name).to("cuda")
model.generation_config.language = "zh"

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
#############################################################################################
# trainer callback to reinitialise and reshuffle the streamable datasets at the beginning of each epoch
class ShuffleCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        if isinstance(train_dataloader.dataset, IterableDatasetShard):
            pass  # set_epoch() is handled by the Trainer
        elif isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)
            

def load_samples_dataset(dataset, num_samples=20):
    samples = []
    for i, item in enumerate(dataset):
        samples.append(item)
        if i == (num_samples-1):
            break
    sample_dataset = Dataset.from_list(samples)
    return sample_dataset

def compute_spectrograms(example):
    #waveform_= example["audio"]
    specs = processor.feature_extractor(example["audio"]["array"], sampling_rate=example["audio"]["sampling_rate"], padding="do_not_pad").input_features[0]
    return {"spectrogram": specs}


def record_to_html(sample_record):
    # print(sample_record)
    audio_array = np.array(sample_record["audio"]["array"])
    audio_sr = sample_record["audio"]["sampling_rate"]
    #audio_array,audio_sr= librosa.load(sample_record["path"],sr=16000)
    length = sample_record["input_length"]
    if length:
        audio_duration = length
    else:
        sample_record["input_length"] = audio_duration = len(audio_array)/audio_sr
    audio_duration = sample_record["input_length"]
    audio_spectrogram = np.array(sample_record["spectrogram"])#[0:int(audio_duration*16000)]
    audio_spectrogram = np.flip(audio_spectrogram, axis=0)
    bounds = (0,0, audio_duration, audio_spectrogram.max())

    waveform_int = np.int16(audio_array * 32767)
    
    
    hv_audio = pn.pane.Audio(waveform_int, sample_rate=audio_sr, name='Audio', throttle=500)
    
    slider = pn.widgets.FloatSlider(end=audio_duration, visible=False, step=0.001)
    line_audio = hv.VLine(0).opts(color='black')
    line_spec = hv.VLine(0).opts(color='red')
    
    
    slider.jslink(hv_audio, value='time', bidirectional=True)
    slider.jslink(line_audio, value='glyph.location')
    slider.jslink(line_spec, value='glyph.location')
    
    time = np.linspace(0, audio_duration, num=len(audio_array))
    line_plot_hv = hv.Curve(
        (time, audio_array), ["Time (s)", "amplitude"]).opts(
        width=500, height=150, axiswise=True) * line_audio
    
    hv_spec_gram = hv.Image(
        audio_spectrogram,bounds=(bounds), kdims=["Time (s)", "Frequency (hz)"]).opts(
        width=500, height=150, labelled=[], axiswise=True, color_levels=512)* line_spec
    
    
    combined = pn.Row(hv_audio, hv_spec_gram, line_plot_hv, slider)
    audio_html = StringIO()
    combined.save(audio_html)
    sample_record["audio_with_spec"] = audio_html
    return sample_record


def dataset_to_records(dataset, is_test=False):
    records = []
    if is_test:
        for item in dataset:
            record = {}
            # sample_record = record_to_html(item)
            #record["audio_with_spec"] = item["audio_with_spec"]
            record["text"] = item["text"]
            record["normalised_text"] = item["text"] #normalizer(item["text"].replace('<|nospeech|>','').lower().strip())
            record["input_length"] = item["input_length"]
            records.append(record)
    #records = pd.DataFrame(records)
    else:
        for item in dataset:
            record = {}
            sample_record = record_to_html(item)
            record["audio_with_spec"] = wandb.Html(sample_record["audio_with_spec"])
            record["text"] = sample_record["text"]
            record["normalised_text"] = sample_record["text"] #normalizer(sample_record["text"].replace('<|nospeech|>','').lower().strip())
            
            record["input_length"] =  sample_record["input_length"]
            records.append(record)
    records = pd.DataFrame(records)
    return records
    
def decode_predictions(trainer, predictions):
    pred_ids = predictions
    #print(np.shape(pred_ids))
    #pred_ids =  np.where(pred_ids != -100, labels, processor.tokenizer.pad_token_id)
    #pred_str = trainer.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, )
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    return pred_str



def compute_measures(predictions, labels):
    res=[]
    for ps,ls in zip(predictions, labels):
        gtlen=len(ls)
        x = jiwer.compute_measures(ls, ps)
        wer=x["wer"]
        cor=x["hits"]/gtlen * 100.0
        sub=x["substitutions"]/gtlen * 100.0
        ins=x["insertions"]/gtlen * 100.0
        dele=x["deletions"]/gtlen * 100.0
        errorCER = cer(ls, ps)
        res.append({"normalised_text":str(ls),"cer":errorCER,"wer":wer,"correct %":cor,"substitutions %":sub,"insertions %":ins,"deletions %":dele})
    # measures =[jiwer.compute_measures(ls, ps) for ps, ls in zip(predictions, labels)]
    # measures = [x['normalised_sentence']=y for x,y in zip(measures, labels)]
    measures_df = pd.DataFrame(res)[["normalised_text", "cer", "wer", "correct %", "substitutions %", "insertions %", "deletions %"]]
    return measures_df

class WandbProgressResultsCallback(WandbCallback):
    def __init__(self, trainer, sample_dataset, test_dataset): 
        super().__init__()
        self.trainer = trainer
        self.sample_dataset = sample_dataset
        self.test_dataset = test_dataset
        self.records_df = dataset_to_records(sample_dataset)
        self.test_df = dataset_to_records(test_dataset, True)
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        super().on_log(args, state, control, model, logs)
        
        sample_dataloader = DataLoader(self.sample_dataset, batch_size=32, collate_fn=data_collator)
        test_dataloader = DataLoader(self.test_dataset, batch_size=32, collate_fn=data_collator)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
        predictions=[]
        test_pred=[]
        self.trainer.model.eval()
        for step, batch in enumerate(tqdm(sample_dataloader,desc="running test set")):
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    generated_tokens = (
                        self.trainer.model.generate(
                            input_features=batch["input_features"].to("cuda"),
                            forced_decoder_ids=forced_decoder_ids,
                            max_new_tokens=255,
                        )
                        .cpu()
                        .numpy()
                    )
                    decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    #predictions.append(decoded_preds[0])
                    #predictions.append(normalizer(decoded_preds[0].lower()).strip())
                    predictions.append(decoded_preds)
                    # print(" ".join(decoded_preds))
                    # print(" ".join(decoded_labels)) 
            del generated_tokens, batch
            gc.collect()
            
        self.trainer.model.train()
        # predictions = trainer.predict(self.sample_dataset)
        #predictions = decode_predictions(self.trainer, predictions)
        predictions = [item for sublist in predictions for item in sublist]
        predictions = [normalizer(x.lower()).strip() for x in predictions]
        
        #test_pred = [item for sublist in test_pred for item in sublist]
        #test_pred = [normalizer(x.lower()).strip() for x in test_pred]
        
        measures_df = compute_measures(predictions,self.records_df["normalised_text"].tolist() )
        #measure_test_df = compute_measures(test_pred,self.test_df["normalised_text"].tolist() )
        
        eval_correct = np.average( measures_df['correct %'].tolist())
        eval_cer = np.average( measures_df['cer'].tolist())
        
        #test_correct = np.average( measure_test_df['correct %'].tolist())
        #test_cer = np.average( measure_test_df['cer'].tolist())
        
        self._wandb.log({"validation/correct %": eval_correct})
        self._wandb.log({"validation/cer": eval_cer})
        
        #self._wandb.log({"eval/correct %": test_correct})
        #self._wandb.log({"eval/cer": test_cer})
        
        records_df = pd.concat([self.records_df, measures_df], axis=1)
        #records_df["normalised_sentence"] = self.records_df["normalised_sentence"].tolist()
        records_df["prediction"] = predictions
        records_df["step"] = state.global_step
        records_table = self._wandb.Table(dataframe=records_df)
        self._wandb.log({"validation/sample_predictions": records_table})
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        peft_model_path = os.path.join(checkpoint_folder)
        #kwargs["model"].save_pretrained(peft_model_path)
        self.trainer.save_model(peft_model_path)

        # pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        # if os.path.exists(pytorch_model_path):
        #     os.remove(pytorch_model_path)
        print("model saved to: ", checkpoint_folder)
       
    
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./medium-finetuned-mandarin_new/",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=6.25e-6,
    warmup_steps=100,
    max_steps=71765,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1250,
    eval_steps=1250,
    logging_steps=500,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    #push_to_hub=True,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=vectorized_datasets["train"],
    eval_dataset=vectorized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    #callbacks=[ShuffleCallback(),] ################################
)

processor.save_pretrained(training_args.output_dir)

print("preparing Sample dataset")
samples_dataset = load_samples_dataset(vectorized_datasets["validation"]).map(compute_spectrograms)
print(f"{len(samples_dataset)} loaded in samples dataset")

progress_callback = WandbProgressResultsCallback(trainer, samples_dataset, vectorized_datasets["test"])
clear_output()

trainer.add_callback(progress_callback)

print("Training Start")
trainer.train(resume_from_checkpoint=False)

"""
try:
    print("Training Start")
    trainer.train(resume_from_checkpoint=False)
except:
    print("ERROR happened!")
    print('The CPU usage is: ', psutil.cpu_percent(4))
    # Getting % usage of virtual_memory ( 3rd field)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)    
    os.system("free")

"""



