# -*- coding: utf-8 -*-
#!/usr/bin/env python

import glob
from audio_utils import read
import librosa
import numpy as np

src_path = '/Users/avin/git/vc/datasets/IU/v_app_split'
# src_path = '/Users/avin/git/vc/datasets/IU/melon_radio_season1_split'
# src_path = '/Users/avin/git/vc/datasets/IU/melon_radio_season2_split'
# src_path = '/Users/avin/git/vc/datasets/timit/TIMIT/TRAIN/*/*'
# src_path = '/Users/avin/git/vc/datasets/kate/sense_and_sensibility_split'
# src_path = '/Users/avin/git/vc/datasets/arctic/bdl'
# src_path = '/Users/avin/git/vc/datasets/arctic/slt'
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


mean_amps = np.mean(amps)
std_amps = np.std(amps)
max_amps = np.max(amps)
min_amps = np.min(amps)

mean_dbs = np.mean(dbs)
std_dbs = np.std(dbs)
max_dbs = np.max(dbs)
min_dbs = np.min(dbs)

mean_log_amps = np.mean(log_amps)
std_log_amps = np.std(log_amps)
max_log_amps = np.max(log_amps)
min_log_amps = np.min(log_amps)

# normalized = (values - mean) / std
# normalized = (log_amps - min) / (max - min)

print("[amps]     max: {}, min: {}, mean: {}, std: {}".format(max_amps, min_amps, mean_amps, std_amps))
print("[log_amps] max: {}, min: {}, mean: {}, std: {}".format(max_log_amps, min_log_amps, mean_log_amps, std_log_amps))
print("[decibels] max: {}, min: {}, mean: {}, std: {}".format(max_dbs, min_dbs, mean_dbs, std_dbs))
# print(normalized)