import librosa
import librosa.display
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np 
import time
import os 

def mp3ToMelSpectrogram(fileName):
    """
    This function converts a power spectrogram (amplitude squared) to decibel (dB) units
    Inputs:
        fileName: relative/absolute path of file to convert
        
    Return:
        S_DB: array representation of mel spectrogram with dB units
        sr: sampling rate
    """
    # load song <y> as numpy array and sampling rate <s>
    y, sr = librosa.load(fileName)

    # trim leading and trainling silence from audio signal
    song, _ = librosa.effects.trim(y)
    S = librosa.feature.melspectrogram(song)
    S_DB = librosa.power_to_db(S, ref=np.max)

    return S_DB, sr



def main(dirName):

    for dirpath, dirnames, files in os.walk('./'+dirName):
        if dirpath == './' + dirName:
            continue
        print(f"Currently in directory: {dirpath}")

        newFolder = dirpath[10:]
        
        print("\tConverting files...")

        num = 0 
        tic = time.time()
        for file_name in files:
            if file_name.endswith('.mp3'):
                print(f"Converting {newFolder} file {num+1}")
                fmp3 = dirpath + '/' + file_name 

                S_DB, sr = mp3ToMelSpectrogram(fmp3)
                librosa.display.specshow(S_DB,sr=sr)
                plt.axis('off')

                newPath = './dataMP3/'+newFolder+'/'+newFolder+str(num)+'.jpg'
                plt.savefig(newPath,bbox_inches='tight',pad_inches = 0)
                
                num += 1

        toc = time.time()
        print(f"Spent {toc-tic} seconds converting {newFolder} songs")
        print(f"\tThere are {num} in {dirpath}")
        print()

    return True


if __name__ == '__main__':
    main('Dataset')
