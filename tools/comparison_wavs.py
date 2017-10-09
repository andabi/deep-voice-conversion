# -*- coding: utf-8 -*-
#!/usr/bin/env python

import librosa.display
import utils
import numpy as np
import matplotlib.pyplot as plt

filename_a = '/Users/avin/git/vc/a.wav'
filename_b = '/Users/avin/git/vc/b.wav'

n_fft = 4096
len_hop = n_fft / 4
plot_wav = True
plot_spec = True

# Waveforms
wav_a, sr = librosa.load(filename_a, mono=True)
wav_b, _ = librosa.load(filename_b, mono=True)

# Spectrogram
spec_a = librosa.stft(wav_a, n_fft=n_fft, hop_length=len_hop)
spec_b = librosa.stft(wav_b, n_fft=n_fft, hop_length=len_hop)

# Plot waveforms
if plot_wav:
    plt.figure(1)

    plt.subplot(2, 1, 1)
    librosa.display.waveplot(wav_a, sr=sr, color='b')
    plt.title('A')

    plt.subplot(2, 1, 2)
    librosa.display.waveplot(wav_b, sr=sr, color='r')
    plt.title('B')

    plt.tight_layout()
    plt.show()

# Plot spectrogram
if plot_spec:
    plt.figure(2)

    plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(spec_a, ref=np.max), sr=sr, hop_length=len_hop, y_axis='log', x_axis='time')
    plt.title('A')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(spec_b, ref=np.max), sr=sr, hop_length=len_hop, y_axis='log', x_axis='time')
    plt.title('B')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.show()