# -*- coding: utf-8 -*-
#!/usr/bin/env python

import glob
import audio_utils

SOURCE_PATH = 'datasets/test/godfather2'
TARGET_PATH = 'datasets/test/godfather2'

# Write mp3 to wav
search_path = '{}/*.mp3'.format(SOURCE_PATH)
for source_path in glob.glob(search_path):
    target_path = '{}/{}.wav'.format(SOURCE_PATH, 'a')
    audio_utils.rewrite_mp3_to_wav(TARGET_PATH, target_path)
