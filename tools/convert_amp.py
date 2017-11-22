# -*- coding: utf-8 -*-
#!/usr/bin/env python

from pydub import AudioSegment
import glob
from audio_utils import split_path
import os

src_path = '/Users/avin/git/vc/datasets/IU/v_app_split'
target_path = '{}_amp'.format(src_path)
target_amp_in_db = -20


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

if not os.path.exists(target_path):
    os.mkdir(target_path)

for filepath in glob.glob('{}/*.wav'.format(src_path)):
    basepath, filename, _ = split_path(filepath)
    sound = AudioSegment.from_wav(filepath)
    normalized_sound = match_target_amplitude(sound, target_amp_in_db)
    normalized_sound.export('{}/{}.wav'.format(target_path, filename), 'wav')