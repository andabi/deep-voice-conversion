# -*- coding: utf-8 -*-
#/usr/bin/python2


# path
## remote
# data_path_base = '/data/private/vc/datasets'
# logdir_path = '/data/private/vc/logdir'

## local
data_path_base = './datasets'
logdir_path = './logdir'


class Hyperparams:
    # signal processing
    sr = 16000 # Sampling rate.
    frame_shift = 0.005  # seconds
    frame_length = 0.025  # seconds
    n_fft = 512

    # sr = 20000 # Sampling rate.
    # frame_shift = 0.0125  # seconds
    # frame_length = 0.05  # seconds
    # n_fft = 1024

    hop_length = int(sr*frame_shift) # samples.  This is dependent on the frame_shift.
    win_length = int(sr*frame_length) # samples. This is dependent on the frame_length.

    preemphasis = 0.97
    n_mfcc = 40
    n_iter = 60 # Number of inversion iterations
    n_mels = 128
    duration = 1

    ###########
    # default #
    ###########
    # model
    hidden_units = 256 # alias = E
    num_banks = 16
    num_highwaynet_blocks = 4
    norm_type = 'ins'  # a normalizer function. value: bn, ln, ins, or None
    t = 1.0  # temperature
    dropout_rate = 0.2

    batch_size = 32

    class train1:
        data_path = '{}/timit/TIMIT/TRAIN/*/*/*.wav'.format(data_path_base)

        # model
        # hidden_units = 256  # alias = E
        # num_banks = 16
        # num_highwaynet_blocks = 4
        # norm_type = 'ins'  # a normalizer function. value: bn, ln, ins, or None
        # t = 1.0  # temperature
        # dropout_rate = 0.2

        batch_size = 32
        lr = 0.0003
        num_epochs = 1000
        save_per_epoch = 2

    class test1:
        data_path = '{}/timit/TIMIT/TEST/*/*/*.wav'.format(data_path_base)
        batch_size = 32

    class train2:
        data_path = '{}/arctic/slt/*.wav'.format(data_path_base)
        # data_path = '{}/kate/sense_and_sensibility_split/*.wav'.format(data_path_base)
        # data_path = '{}/kate/therese_raquin_split/*.wav'.format(data_path_base)
        # data_path = '{}/kate/*_split/*.wav'.format(data_path_base)

        # model
        # hidden_units = 512  # alias = E
        # num_banks = 16
        # num_highwaynet_blocks = 8
        # norm_type = 'ins'  # a normalizer function. value: bn, ln, ins, or None
        # t = 1.0  # temperature
        # dropout_rate = 0.2

        batch_size = 32
        lr = 0.0005
        num_epochs = 10000
        save_per_epoch = 50

    class test2:
        batch_size = 32

    class convert:
        # data_path = '{}/test/godfather2/*.wav'.format(data_path_base)
        data_path = '{}/arctic/bdl/*.wav'.format(data_path_base)
        # data_path = '{}/kate/sense_and_sensibility_split/*.wav'.format(data_path_base)
        # data_path = '{}/kate/therese_raquin_split/*.wav'.format(data_path_base)
        # wav_files = glob.glob('{}/iKala/Wavfile/*.wav'.format(hp.data_path))
        batch_size = 4
        emphasis_magnitude = 1.3