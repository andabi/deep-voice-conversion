# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import librosa
import numpy as np
from audio_utils import spectrogram2wav as griffin_lim

file_path = '/Users/avin/git/vc/datasets/kate/sense_and_sensibility/1-1-001.wav'
sr = 22050
n_fft = 512
win_length = 400
hop_length = win_length / 4
num_iters = 30

wav, sr = librosa.load(file_path, sr=sr, mono=True)
spec = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
mag, phase = np.abs(spec), np.angle(spec)

# Griffin Lim reconstruction
wav_recon = griffin_lim(mag, n_fft=n_fft, win_length=win_length, hop_length=hop_length, num_iters=num_iters, length=wav.shape[0])
# sum_wav = np.sum(np.abs(wav))
# sum_wav_recon = np.sum(np.abs(wav_recon))
# print(sum_wav, sum_wav_recon, (sum_wav - sum_wav_recon) / sum_wav * 100)

# Write
librosa.output.write_wav('wav_orig.wav', wav, sr)
librosa.output.write_wav('wav_recon_{}.wav'.format(num_iters), wav_recon, sr)