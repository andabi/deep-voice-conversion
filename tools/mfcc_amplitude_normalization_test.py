# -*- coding: utf-8 -*-
#!/usr/bin/env python

import librosa
import numpy as np

filename = '/Users/avin/git/vc/datasets/timit/TIMIT/TRAIN/DR1/FCJF0/SA1.wav'  # sample rate = 16,000 Hz

sr = 16000
n_fft = 512
win_length = 400
hop_length = 80
n_mels = 128
n_mfcc = 40

# Load waveforms
y, _ = librosa.load(filename, mono=True, sr=sr)

# Get spectrogram
D = librosa.stft(y=y,
                 n_fft=n_fft,
                 hop_length=hop_length,
                 win_length=win_length)
mag = np.abs(D)
scaled_mag = mag * 2

# Get mel-spectrogram
mel_basis = librosa.filters.mel(sr, n_fft, n_mels)  # (n_mels, 1+n_fft//2)
mel = np.dot(mel_basis, mag ** 1)  # (n_mels, t) # mel spectrogram
scaled_mel = np.dot(mel_basis, scaled_mag ** 1)

# Get mfccs
db = librosa.power_to_db(mel)
scaled_db = librosa.power_to_db(scaled_mel)

mfccs = np.dot(librosa.filters.dct(n_mfcc, db.shape[0]), mel)
scaled_mfccs = np.dot(librosa.filters.dct(n_mfcc, db.shape[0]), scaled_mel)

mfccs = mfccs.T  # (t, n_mfccs)
scaled_mfccs = scaled_mfccs.T

assert(np.all(mfccs * 2 == scaled_mfccs))

print(mfccs)
print(scaled_mfccs)