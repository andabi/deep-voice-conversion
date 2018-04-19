# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function

import tensorflow as tf
from models import Net2
import argparse
from hparam import hparam as hp
from tensorpack.predict.base import OfflinePredictor
from tensorpack.predict.config import PredictConfig
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.tfutils.sessinit import ChainInit
from data_load import Net2DataFlow


def get_eval_input_names():
    return ['x_mfccs', 'y_spec']


def get_eval_output_names():
    return ['net2/eval/summ_loss']


def eval(logdir1, logdir2):
    # Load graph
    model = Net2()

    # dataflow
    df = Net2DataFlow(hp.test2.data_path, hp.test2.batch_size)

    ckpt1 = tf.train.latest_checkpoint(logdir1)
    ckpt2 = tf.train.latest_checkpoint(logdir2)
    session_inits = []
    if ckpt2:
        session_inits.append(SaverRestore(ckpt2))
    if ckpt1:
        session_inits.append(SaverRestore(ckpt1, ignore=['global_step']))
    pred_conf = PredictConfig(
        model=model,
        input_names=get_eval_input_names(),
        output_names=get_eval_output_names(),
        session_init=ChainInit(session_inits))
    predictor = OfflinePredictor(pred_conf)

    x_mfccs, y_spec, _ = next(df().get_data())
    summ_loss, = predictor(x_mfccs, y_spec)

    writer = tf.summary.FileWriter(logdir2)
    writer.add_summary(summ_loss)
    writer.close()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case1', type=str, help='experiment case name of train1')
    parser.add_argument('case2', type=str, help='experiment case name of train2')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    hp.set_hparam_yaml(args.case2)
    logdir_train1 = '{}/{}/train1'.format(hp.logdir_path, args.case1)
    logdir_train2 = '{}/{}/train2'.format(hp.logdir_path, args.case2)

    eval(logdir1=logdir_train1, logdir2=logdir_train2)

    print("Done")