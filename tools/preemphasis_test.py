# -*- coding: utf-8 -*-
#!/usr/bin/env python

import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from audio_utils import read, write, preemphasis, inv_preemphasis
import sys

filename = '/Users/avin/git/vc/datasets/timit/TIMIT/TEST/DR1/FAKS0/SA1.wav'
sr = 22050
n_fft = 4096
len_hop = n_fft / 4
write_wav = True
plot_wav = True
plot_spec = True

# Waveforms
wav = read(filename, sr, mono=True)
wav_empha = preemphasis(wav)
diff = wav - inv_preemphasis(wav_empha)
assert(np.all(np.less(diff, sys.float_info.epsilon)))

# Spectrogram
spec = librosa.stft(wav, n_fft=n_fft, hop_length=len_hop)
spec_empha = librosa.stft(wav_empha, n_fft=n_fft, hop_length=len_hop)

# Write wav
if write_wav:
    write(wav, sr, 'original2.wav')
    write(wav_empha, sr, 'preemphasized2.wav')

# Plot waveforms
if plot_wav:
    plt.figure(1)

    plt.subplot(2, 1, 1)
    librosa.display.waveplot(wav, sr=sr, color='b')
    plt.title('before preemphasis')

    plt.subplot(2, 1, 2)
    librosa.display.waveplot(wav_empha, sr=sr, color='r')
    plt.title('after preemphasis')

    plt.tight_layout()
    plt.show()

# Plot spectrogram
if plot_spec:
    plt.figure(2)

    plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(spec, ref=np.max), sr=sr, hop_length=len_hop, y_axis='log', x_axis='time')
    plt.title('before preemphasis')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(spec_empha, ref=np.max), sr=sr, hop_length=len_hop, y_axis='log', x_axis='time')
    plt.title('after preemphasis')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.show()