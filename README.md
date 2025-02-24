# WhisperFinetuningScripts
Scripts used from collecting data to the fine-tuning process.

Whisper fine-tuning is the process of adapting the pre-trained Whisper ASR model to improve its performance on specific datasets or tasks. By training it further with specialized data, it enhances accuracy for particular use cases, such as different accents, specific topics, or unique environments.

# Collecting Data

The order of the data types based on quality from good to bad:
 
-	Contact Centre
-	Multi speaker - Conversational
-	YouTube Data
-	Single Speaker
-	Clean Data
-	Read Speech

It's better if the dataset includes a variety of accents, demographics, ages, and genders.


## Downloading YouTube Audio Files and Subtitles

A text file should be created containing all the desired YouTube IDs. Then with, yt_dlp library (https://github.com/yt-dlp/yt-dlp) YouTube audios can be downloaded. 

Example Script:
https://github.com/IntelligentVoice/WhisperFinetuningScripts/blob/main/download_youtubefiles.py

Downloaded audio files often have excessively long names. File names should be trimmed to only include IDs.

Example Script:
https://github.com/IntelligentVoice/WhisperFinetuningScripts/blob/main/FixDownloadedYoutubeFileNames.ipynb

To download corresponding subtitles, youtube_transcript_api library (https://pypi.org/project/youtube-transcript-api/) can be used. When dealing with subtitles, it's important to understand which ID is used for the subtitle language. For example, although Cantonese is a Chinese language, subtitles for it may have various IDs such as ```zh-CN```, ```zh```, ```zh-Hant```, ```zh-Hans```, or ```zh-HK```. These cases should be included in the script. 

This script uses the same text file containing a list of intended IDs and downloads their subtitles if available.

Example Script: 
https://github.com/IntelligentVoice/WhisperFinetuningScripts/blob/main/downloadScript.py

After downloading all possible subtitles, a cleaning script, which is explained below, should be run for the downloaded subtitles.


## Generating Synthetic Data

If there is not enough data for a language, or terms or if we want to boost centatin keywords we can generate Syntheitc data.

As the steps, first we need to generate syntehic sentences having particulater topics or keywords with using LLM (CoPilot), or alternatively we can do web scraping and collect all the sentence that have the keywords.

There is a script to collect the books in txt formats from www.gutenberg.org website. Project Gutenberg is an online library of free eBooks. Then according to keywords the script extract the sentences that have the keywords.

WebScrapingBooks.ipynb

Also there is another script to do webscraping from Wikipedia.

WebScrapingWikipedia.ipynb

After having the texts we can generate audios by using ToucanTTS (https://github.com/DigitalPhonetics/IMS-Toucan). IMS Toucan is a toolkit for training, using, and teaching state-of-the-art Text-to-Speech Synthesis, developed at the Institute for Natural Language Processing (IMS), University of Stuttgart, Germany, official home of the massively multilingual (more than 7000 languages) ToucanTTS system. 
However before running the Toucan TTS script, we also need to create a folder that have refernece audios. These audio files will have the voices of synthetic audio files to be genrated. I usually use CommonVoice files as reference audios to generate synthetic audio files because they are clean, understandable, and free from background noise. 

If the reference audio is noisy, the generated synthetic audio will also be noisy, but not in a natural way. 

Therefore, I believe it is better to first generate clean synthetic audios and then apply augmentations, such as adding background noise, overtalk, and backchannel effects. 

The script to generate synthetic audio files with Toucan TTS : 

GenerateSyntheticAudioToucanTTS.py

Remember! ToucanTTS is not a perfect tool. Everyday there are a new publisted TTS tools.

## Cleaning The Text Data

Data should have accurate letters/characters. For example, a Mandarin dataset should contain only simplified Chinese characters, whereas a Cantonese dataset should contain only traditional characters.

Example Script: 
https://github.com/IntelligentVoice/WhisperFinetuningScripts/blob/main/CheckSimplifiedCharacters.ipynb

## Normalizing The Data

All downloaded audio files should be converted into WAV format with a sample rate of 16kHz and a single channel. To expedite processing, the data can be divided into groups, and commands can be executed in parallel.

These are some of the useful Linux commands:

Converts mp3 to wav:
```for i in *.mp3; do ffmpeg -i "$i" -acodec pcm_s16le -ac 1 -ar 16000 "$(basename -s .mp3 "$i").wav"; done  ```

Converts flac to wav:
```for i in *.flac; do ffmpeg -i "$i" "$(basename -s .flac "$i").wav"; done```

Converts m4a to wav:
```for i in *.m4a; do ffmpeg -i "$i" -ar 16000 "${i%.m4a}.wav"; done```

Converts webm to wav:
```for i in *.webm; do ffmpeg -i "$i" -c:a pcm_f32le "${i%.webm}.wav"; done```

Converts mkv to wav:
```for i in *.mkv; do ffmpeg -i "$i" -acodec pcm_s16le -ac 1 "${i%.mkv}.wav"; done```

Converts any wav files to 1 channel, 16kHz:
```for file in *.wav; do sox $file -r 16000 -c 1 -b 16 16kHz_$file; done```


After downloading YouTube files, we need to split the audio files based on their subtitle timing information. Additionally, subtitles and their corresponding audio files should be written into a .json file. Furthermore, we may want to extract subtitles that contain only desired keywords.

Example script to create a .json file from downloaded subtitles with having only wanted keywords:
https://github.com/IntelligentVoice/WhisperFinetuningScripts/blob/main/JsonFromDownloadedYoutubeSubtitlesWithKeywords.ipynb

We will take the timing information from the json script that is created above.

Example script to split YouTube files based on timing information:
https://github.com/IntelligentVoice/WhisperFinetuningScripts/blob/main/SplitAudiosFromJson.ipynb
Also, we can create json files from many type of datasets.

To create .json script from TextGrid files:
https://github.com/IntelligentVoice/WhisperFinetuningScripts/blob/main/JSONfromTextGrid.ipynb

To create .json script from CSV files:
https://github.com/IntelligentVoice/WhisperFinetuningScripts/blob/main/JSONfromCSV.ipynb

To create .json script from CommonVoice txt files:
https://github.com/IntelligentVoice/WhisperFinetuningScripts/blob/main/JSONfromCommonVoice.ipynb

After these steps, .json file should be checked to see if all audio files have the correct paths. 

Example Script:
https://github.com/IntelligentVoice/WhisperFinetuningScripts/blob/main/CheckFileExists.ipynb

Also, all special characters and blank ground truth texts (GT) should be eliminated from training datasets.

Example script:
https://github.com/IntelligentVoice/WhisperFinetuningScripts/blob/main/FindWeirdText.ipynb

If wanted, duration of the whole dataset can be calculated. There are two ways for this; Linux command or a python script if there are massive number of audio files.

Linux command to get total durations in a folder in seconds:
```soxi -DT *.wav```

Example python script:
https://github.com/IntelligentVoice/WhisperFinetuningScripts/blob/main/CheckDurations.ipynb

For APAC languages, proper spacing is crucial, especially when they include loanwords. Typically, there are no spaces between Chinese characters, but spaces should be inserted between loanwords.

Example script:
https://github.com/IntelligentVoice/WhisperFinetuningScripts/blob/main/FixSpaces.ipynb

Additionally, for Cantonese, we need to extract only files that have 'verbal form.' We don't want written forms because, in most cases, what is written and what is spoken are not the same. This discrepancy is not ideal for training an ASR model.

Example python script:
https://github.com/IntelligentVoice/WhisperFinetuningScripts/blob/main/ExtractVerbalFormCantonese.py


# Validating The Collected Data

## Sanity Check

All necessary datasets should undergo a sanity check to eliminate false ground truths. For this purpose, Faster Whisper Medium model can be run with each file and saving the results to a .txt file. If the character error rate (CER) is below 0.5, the file can be included in the training scripts.

Example script to run sanity check:
https://github.com/IntelligentVoice/WhisperFinetuningScripts/blob/main/code_sanity_check.py

For a faster process, audio files can be split into several folders, and the above script can be run separately in screen sessions.

After the sanity check is completed, a new .json file can be created from the result .txt file, containing only the files with a CER of less than 0.5.

Example script:
https://github.com/IntelligentVoice/WhisperFinetuningScripts/blob/main/EliminateSanityCheckResults.ipynb


## Review of Collected Datasets

If possible, collected and processed files can be listened to ensure that everything is correct with the audio. Furthermore, having a native speaker to review a few samples of the datasets will provide valuable insights into the data quality.


# Augmenting Training Files

## GSM Encoding

Before training has started, ideally, we want to augment the audio quality. Therefore, we are converting half of the dataset to GSM format and then back to WAV format.

Example Script:
https://github.com/IntelligentVoice/WhisperFinetuningScripts/blob/main/run_create_gsm.py

## Increasing The Number of Data

Moreover, we can increase the amount of data by augmenting the existing dataset through trimming and combining different audio files. Here is the script for YouTube files:

GenerateMoreYoutubeFilesAugmentation.ipynb

## Augmentation in Training

Furthermore, in training script, there is an option for augmentation. It uses Musan dataset (https://arxiv.org/pdf/1510.08484) and combines music, background noise etc. audios together. 

Here is the example code:

```
# define augmentation
augment_waveform = Compose([TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2, leave_length_unchanged=False),
        Gain(min_gain_in_db=-2, max_gain_in_db=2, p=0.1),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
        OneOf([AddBackgroundNoise(sounds_path=musan_dir, min_snr_in_db=1.0, max_snr_in_db=2.0, noise_transform=PolarityInversion(), p=0.9),
  AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=0.4),],p=1.0,),])

def augment_dataset(batch):
   audio = batch["audio"]['array']
    # apply augmentation
    augmented_audio = augment_waveform(samples=np.array(audio), sample_rate=16000)
batch["audio"]['array'] = augmented_audio
return batch
```

# Creating Training Json Files

For training we should have 3 or at least 2 different .json files. These are; train.json, test.json and validation.json.

Typically, the proposed ratio is 0.8 for training, 0.1 for testing, and 0.1 for validation. However, if there are many files, such as over a million, the ratio for the test and validation datasets can be less to not slow the process.

## Adding Silence to Audio Files, Background noises and “…” Markers

To decrease the hallucinations, we can add audios that have silence or background noises with the corresponding “…” GT text. We are using Musan dataset for this for now. (https://www.openslr.org/17/)

We can concatenate the Musan data to train.json as shown below:

```cat train.json >> noise_and_silence.json```


# Training

Depending on the number of training files, training may take up to a week. Ensure all necessary libraries are downloaded. Here is the list of all necessary libraries:

To create files with a Conda environment on Jupyter: 
```
pip install notebook
pip install nbconvert
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=esma_env
```
Required libraries for finetuning:
```
pip install ffmpeg
pip install pandas
pip install matplotlib
pip install seaborn
pip install tqdm
pip install datasets==2.17.1
pip install evaluate
pip install torchvision 
pip install git+https://github.com/openai/whisper.git 
pip install transformers==4.36.2
pip install wandb
pip install holoviews
pip install librosa
pip install jiwer
pip install audiomentations
pip install transformers[torch]
pip install accelerate -U
pip install wordsegment
pip install yt-dlp[default]
```
Required libraries for testing the generated checkpoints:
```
pip install faster-whisper==1.0.1
pip install git+https://github.com/bayartsogt-ya/whisper-multiple-hf-datasets.git
pip install termcolor
pip install markdown_cjk_spacing
pip install jiwer
```

This is an example of Whisper Medium Model Finetuning script for Mandarin:
https://github.com/IntelligentVoice/WhisperFinetuningScripts/blob/main/whMedium_zh_Mandarin_Finetuning_Augment.py

This is another script for Finnish language:

WhisperFinetuning_fi_Finnish.py

Training script can be run within a screen session.

Important Note:
In Convproc2, multiple trainings cannot be started at the same time if data cache folder is not created yet. Because, if any other process is running, during the data cache process, it gives this error:
RuntimeError: One of the subprocesses has abruptly died during map operation.To debug the error, disable multiprocessing.


# Testing The Generated Checkpoints

## Find The Best Generated Checkpoint

To find the best checkpoint among all the checkpoints, a short test dataset can be defined. Subsequently, an automated test can be conducted.

Example Script:
https://github.com/IntelligentVoice/WhisperFinetuningScripts/blob/main/run_automated_test.py

Depending on the number of checkpoints, this script can take a few hours to complete. The script mentioned above creates a .txt file and records the results of COR, WER, INS, DEL, and SUB for each file with respect to each checkpoint.

Two checkpoints can be selected: one with the highest COR and one with the lowest WER.

After this, these two checkpoints can be converted into Faster Whisper models and tested with different test datasets within the IV-ASR environment. Before converting to Faster Whisper, the corresponding tokenizer.json should be downloaded from here: https://huggingface.co/openai/whisper-medium/tree/main (if the Medium model is fine-tuned). Otherwise, download tokenizer.json from the corresponding model folder.

Linux command to convert Whisper Models to Faster Whisper:
```ct2-transformers-converter --model <model_folder> --output_dir <target_folder> --copy_files tokenizer.json --quantization float16```


## Conducting Tests Within IV-ASR Environment

### Running Models to Test Files

For testing, first we need to convert Whisper model to Faster Whisper model. Here is the command to convert models with faster-whisper package. 

```ct2-transformers-converter --model /path-to/checkpoint-folder--output_dir checkpoint-mandarin-ct2 --copy_files tokenizer.json --quantization float16```

After converting the model, it should be put in here ```/opt/models/ASR/``` in Empathic machine, with a similar folder name like: ```IntelligentVoice_zh-001_16kHz_24_general_V2_NASRv5.1```

As different models added, we can change the “zh” which is language id and “V3” which is a version number. Also “24” is the number of layers that model has. 

•	A folder should be created in ```/home/chase/esma/iv_asr/``` folder. 

•	Then, put audio files (8kHz or 16kHz) in a folder ```<test>```

•	To create res.batch, run VAD as: ```/opt/vad/prep.sh <test>```

•	Create ```nemo_job_models``` file with a model in each line:

(```<model_id> <model_path> <model_wight> [beamSearchBeamSize]```)
  	Example:
```1 /opt/models/ASR/IntelligentVoice_en-001_16kHz_24_general_V1_NASRv5.1 1```

•	Run ASR script in ```<res.batch>``` folder:

```/home/esma/iv_asr/run_iv.sh```

•	Parameters can be changed in file (gpu_processes: be able to run # number of processes in a GPU):

```/opt/IV/kaldi/groupProcess/ASR.conf```
```/opt/IV/lock/gpu_lock```

•	Before running test scripts again, run cleanup script which would delete results in ```res.batch``` folder:

```/home/esma/iv_asr/cleanup.sh```

•	Before running test scripts again, run cleanup script which would delete results in ```res.batch``` folder:


### Running Sclite Scoring Script to Get the Scores

For running the Sclite scoring script, these are the steps should be processed in the ```/home/chase/esma/sclite/``` folder:
•	Put GT txt files in ```<test>``` folder.

•	Put prediction files (psrt or txt) in a subfolder ```<test>/<pred>```

•	Run  ```python putspace_loanwords.py <test>```

•	Run ```python putspace_loanwords.py <test>/<pred>```

•	Run score_list.sh (or ```score_list_ch.sh``` for char-based scoring) inside ```<test>``` folder as:

```/home/chase/esma/sclite/score_list.sh <test>/<pred>```

•	If you want to omit some files in scoring, delete them in ```<test>/test.files``` file created after the first scoring run. 

•	For punctuation cleanup of txt files run below command in the location of txt files:

```/home/chase/esma/sclite/dataPrep/pre_replace.sh```

•	For number to text conversion (This may not be necessary if both GT and prediction have numbers in the same format) run below command in the location of txt files:

```/home/esma/sclite/dataPrep/cleanup.sh <lang_code>```

•	  Make sure that ```replace_<lang_code>_file.sh``` exists in ```/home/esma/sclite/dataPrep``` folder. Otherwise, you can't do numbers to text conversion for that language.

•	  Additionally, you can create a colored analysis using:

ColouredAnalysis.ipynb

## Checking TP Rate:
To analyze keywords or loan words, we can check and count the number of True Positives.

Script: scoreCSV_FP.py

Important Note:
Don’t forget to check all test results to see whether punctuation has been generated correctly!



