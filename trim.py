from os import listdir
from os.path import isfile, join
from pydub import AudioSegment
from mutagen.mp3 import MP3
import random

path = '/Users/armaanlalani/Documents/Engineering Science Year 3/ECE324 - Intro to Machine Intelligence/Project/Extra Music/Rock'
files = [f for f in listdir(path) if isfile(join(path, f))]
print(len(files))

for i in files:
    if i != '.DS_Store':
        print(i)
        song = AudioSegment.from_mp3(path+'/'+i)
        audio = MP3(path+'/'+i)
        start = random.uniform(0,audio.info.length-60)
        end = start + 30
        start_min = start // 60
        start_sec = start - start_min*60
        end_min = end // 60
        end_sec = end - end_min*60
        start_time = start_min*60*1000+start_sec*1000
        end_time = end_min*60*1000+end_sec*1000
        extract = song[start_time:end_time]
        extract.export('/Users/armaanlalani/Documents/Engineering Science Year 3/ECE324 - Intro to Machine Intelligence/Project/Extra Music/Rock/Cut/'+i, format='mp3')
        print("---SUCCESS: " + i + ' has been saved in the folder Cut')

