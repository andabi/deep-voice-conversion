# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function

import argparse
import multiprocessing
import os

from tensorpack.callbacks.saver import ModelSaver
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.train.interface import TrainConfig
from tensorpack.train.interface import launch_train_with_config
from tensorpack.train.trainers import SyncMultiGPUTrainerReplicated
from tensorpack.utils import logger
from tensorpack.input_source.input_source import QueueInput
from data_load import Net1DataFlow
from hparam import hparam as hp
from models import Net1
import tensorflow as tf


def train(args, logdir):

    # model
    model = Net1()

    # dataflow
    df = Net1DataFlow(hp.train1.data_path, hp.train1.batch_size)

    # set logger for event and model saver
    logger.set_logger_dir(logdir)

    session_conf = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
        ),)

    train_conf = TrainConfig(
        model=model,
        data=QueueInput(df(n_prefetch=1000, n_thread=4)),
        callbacks=[
            ModelSaver(checkpoint_dir=logdir),
            # TODO EvalCallback()
        ],
        max_epoch=hp.train1.num_epochs,
        steps_per_epoch=hp.train1.steps_per_epoch,
        # session_config=session_conf
    )
    ckpt = '{}/{}'.format(logdir, args.ckpt) if args.ckpt else tf.train.latest_checkpoint(logdir)
    if ckpt:
        train_conf.session_init = SaverRestore(ckpt)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        train_conf.nr_tower = len(args.gpu.split(','))

    trainer = SyncMultiGPUTrainerReplicated(hp.train1.num_gpu)

    launch_train_with_config(train_conf, trainer=trainer)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name')
    parser.add_argument('-ckpt', help='checkpoint to load model.')
    parser.add_argument('-gpu', help='comma separated list of GPU(s) to use.')
    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':
    args = get_arguments()
    hp.set_hparam_yaml(args.case)
    logdir_train1 = '{}/train1'.format(hp.logdir)

    print('case: {}, logdir: {}'.format(args.case1, args.case, logdir_train1))

    train(args, logdir=logdir_train1)

    print("Done")