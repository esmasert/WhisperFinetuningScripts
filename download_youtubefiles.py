import os
import glob
from tqdm import tqdm
URLS = []


theIDList = []


for aline in theIDList:
    #print(aline)
    theID = aline

    #print(theID)
    URLS.append("https://www.youtube.com/watch?v="+theID)

print(len(URLS))

import yt_dlp

# if there is error saying name too long use this:
# yt-dlp -o "%(id)s.%(ext)s" "https://www.youtube.com/watch?v=7JexmGkneqg"


ydl_opts = {
    'format': 'm4a/bestaudio/best',
    # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
    'postprocessors': [{  # Extract audio using ffmpeg
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }]
}



for anURL in URLS:

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download(anURL)
    except:
        print("didn't work!")
        print("trying another method")

        cmd = 'yt-dlp -o "%(id)s.%(ext)s" "' + anURL + '"'

        print("************")
        print(cmd)
        print("************")

        os.system(cmd)
    













#