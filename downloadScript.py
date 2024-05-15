
import os
import glob
from tqdm import tqdm
plainURLs=[]



theIDList = []



for aline in theIDList:
    #print(aline)
    theID = aline[1]

    #print(theID)
    plainURLs.append(theID)

print(len(plainURLs))



#Find Language Codes:

from youtube_transcript_api import YouTubeTranscriptApi 

cntworked=0


for an_id in plainURLs:

    flag=0
    
    try:
        srt = YouTubeTranscriptApi.get_transcript(an_id, languages=['zh-CN'])
    except:
        try:
            srt = YouTubeTranscriptApi.get_transcript(an_id, languages=['zh'])
        except:
            try:
                srt = YouTubeTranscriptApi.get_transcript(an_id, languages=['zh-Hant'])
            except:
                try:
                    srt = YouTubeTranscriptApi.get_transcript(an_id, languages=['zh-TW'])
                except:
                    try:
                        srt = YouTubeTranscriptApi.get_transcript(an_id, languages=['zh-Hans'])
                    except:
                        try:
                            srt = YouTubeTranscriptApi.get_transcript(an_id, languages=['zh-HK'])
                        except:
                            print("didn't work")
                            flag=1

    if flag != 1 and len(srt)>1:
        
        newtxtName = "/downloaded_subtitles/"+an_id+".txt"
        thefile = open(newtxtName, "a")
    
        for anitem in srt:
            thefile.write(str(anitem) + ",\n")
        
        cntworked +=1
    
        print(cntworked)
        
    
print(cntworked)




















#