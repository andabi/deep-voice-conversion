# -*- coding: utf-8 -*-
#!/usr/bin/env python

import librosa
import datetime

filename = '/Users/avin/git/vc/datasets/timit/TIMIT/TRAIN/DR1/FCJF0/SA1.wav'  # sample rate = 16,000 Hz

# same sample rate
s = datetime.datetime.now()
wav, _ = librosa.load(filename, mono=True, sr=16000)
e = datetime.datetime.now()
diff = e - s
print(diff.microseconds)

# different sample rate (22,050 Hz)
s = datetime.datetime.now()
wav, _ = librosa.load(filename, mono=True)
e = datetime.datetime.now()
diff = e - s
print(diff.microseconds)
