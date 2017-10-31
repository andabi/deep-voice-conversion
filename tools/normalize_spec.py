# -*- coding: utf-8 -*-
#!/usr/bin/env python

import glob
from audio_utils import read
import librosa
import numpy as np

src_path = '/Users/avin/git/vc/datasets/timit/TIMIT/TRAIN/*/*'
# src_path = '/Users/avin/git/vc/datasets/kate/sense_and_sensibility_split'
# src_path = '/Users/avin/git/vc/datasets/arctic/bdl'
# src_path = '/Users/avin/git/vc/datasets/kate/therese_raquin_split'
sr = 16000
n_fft = 512
win_length = 400
hop_length = 80
n_sample = 200

amps = []
log_amps = []
dbs = []
for filepath in glob.glob('{}/*.wav'.format(src_path))[:n_sample]:
    wav = read(filepath, sr, mono=True)
    spec = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # (n_fft/2+1, t)
    amp = np.abs(spec)
    amps.extend(amp.flatten())

    log_amp = np.log(amp)
    log_amps.extend(log_amp.flatten())

    db = librosa.amplitude_to_db(amp)
    dbs.extend(db.flatten())

amps = np.array(amps)
log_amps = np.array(log_amps)
dbs = np.array(dbs)


mean = np.mean(amps)
std = np.std(amps)

max = np.max(amps)
min = np.min(amps)

# mean = np.mean(dbs)
# std = np.std(dbs)
#
# max = np.max(dbs)
# min = np.min(dbs)

# mean = np.mean(log_amps)
# std = np.std(log_amps)
#
# max = np.max(log_amps)
# min = np.min(log_amps)

# normalized = (values - mean) / std
# normalized = (log_amps - min) / (max - min)

print("max: {}, min: {}, mean: {}, std: {}".format(max, min, mean, std))
# print(normalized)