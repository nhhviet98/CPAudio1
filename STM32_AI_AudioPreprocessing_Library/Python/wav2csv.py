import librosa
import pandas as pd
import numpy as np

ys, sr = librosa.load("1-02fb6c5b.wav", sr=16000, mono=True)
ys = ys[:1024]
ys = pd.DataFrame(ys)
#ys = round(ys, 5)
ys = ys.transpose()
ys.to_csv("audio.csv", header=False, index=False)

print("xong!")