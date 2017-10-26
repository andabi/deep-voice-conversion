# -*- coding: utf-8 -*-
#!/usr/bin/env python

import glob
from audio_utils import read
import librosa
import numpy as np

src_path = '/Users/avin/git/vc/datasets/kate/sense_and_sensibility_split'
# src_path = '/Users/avin/git/vc/datasets/arctic/slt'
# src_path = '/Users/avin/git/vc/datasets/kate/therese_raquin_split'
sr = 16000
n_fft = 512
win_length = 400
hop_length = 80
n_sample = 200

values = []
for filepath in glob.glob('{}/*.wav'.format(src_path))[:n_sample]:
    wav = read(filepath, sr, mono=True)
    spec = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # (n_fft/2+1, t)
    amp = np.abs(spec)
    # db = librosa.amplitude_to_db(amp)
    log_amp = np.log(amp)
    values.extend(log_amp.flatten())
values = np.array(values)

mean = np.mean(values)
std = np.std(values)

max = np.max(values)
min = np.min(values)

# normalized = (values - mean) / std
normalized = (values - min) / (max - min)

print(max, min)
print(mean, std)
print(values)
print(normalized)