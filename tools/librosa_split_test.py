# -*- coding: utf-8 -*-
#!/usr/bin/env python

import librosa
import datetime

filename = '/Users/avin/git/vc/datasets/kate/sense_and_sensibility/1-1-003.wav'

wav, sr = librosa.load(filename, mono=True, sr=22050)
intervals = librosa.effects.split(wav, top_db=30)
for i, interval in enumerate(intervals):
    s, e = interval[0], interval[1]
    w = wav[s:e]
    librosa.output.write_wav('/Users/avin/git/vc/{}.wav'.format(i), w, sr)
