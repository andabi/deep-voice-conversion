# -*- coding: utf-8 -*-
#!/usr/bin/env python

import librosa.display
import utils
import numpy as np
import matplotlib.pyplot as plt
from audio_utils import read, write

filename = '/Users/avin/git/vc/datasets/timit/TIMIT/TEST/DR1/FAKS0/SA1.wav'
sr = 22050
n_fft = 4096
len_hop = n_fft / 4
plot_wav = True
plot_spec = True

# Waveforms
wav = read(filename, sr, mono=True)
# wav = np.where(wav == 0, 1000, wav)
# wav = np.zeros_like(wav)
# wav[0] = np.ones_like(wav[0])

# Spectrogram
spec = librosa.stft(wav, n_fft=n_fft, hop_length=len_hop)

# Plot waveforms
if plot_wav:
    plt.figure(1)

    librosa.display.waveplot(wav, sr=sr, color='b')
    plt.title('waveform')

    plt.tight_layout()
    plt.show()

# Plot spectrogram
if plot_spec:
    plt.figure(2)

    librosa.display.specshow(librosa.amplitude_to_db(spec, ref=np.max), sr=sr, hop_length=len_hop, y_axis='log', x_axis='time')
    plt.title('spectrogram')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.show()