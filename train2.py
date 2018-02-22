# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function

from tqdm import tqdm

import tensorflow as tf
from models import Model
import convert, eval2
from data_load import get_batch
import argparse
from hparam import hparam as hp
import math
import os
from utils import remove_all_files


def train(logdir1, logdir2, queue=True):

    model = Model(mode="train2", batch_size=hp.train2.batch_size, hp=hp, queue=queue)

    # Loss
    loss_op = model.loss_net2()

    # Training Scheme
    epoch, gs = Model.get_epoch_and_global_step(logdir2)
    global_step = tf.Variable(gs, name='global_step', trainable=False)

    lr = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net2')

        # Gradient clipping to prevent loss explosion
        gvs = optimizer.compute_gradients(loss_op, var_list=var_list)
        gvs = [(tf.clip_by_value(grad, hp.train2.clip_value_min, hp.train2.clip_value_max), var) for grad, var in gvs]
        gvs = [(tf.clip_by_norm(grad, hp.train2.clip_norm), var) for grad, var in gvs]

        train_op = optimizer.apply_gradients(gvs, global_step=global_step)

    # Summary
    # for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net2'):
    #     tf.summary.histogram(v.name, v)
    tf.summary.scalar('net2/train/loss', loss_op)
    tf.summary.scalar('net2/train/lr', lr)
    summ_op = tf.summary.merge_all()

    session_conf = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=0.6,
        ),
    )
    # Training
    with tf.Session(config=session_conf) as sess:
        # Load trained model
        sess.run(tf.global_variables_initializer())
        model.load(sess, mode='train2', logdir=logdir1, logdir2=logdir2)

        writer = tf.summary.FileWriter(logdir2)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(epoch + 1, hp.train2.num_epochs + 1):
            for _ in tqdm(range(model.num_batch), total=model.num_batch, ncols=70, leave=False, unit='b'):
                gs = sess.run(global_step)

                # Cyclic learning rate
                feed_dict = {lr: get_cyclic_lr(gs)}
                if queue:
                    sess.run(train_op, feed_dict=feed_dict)
                else:
                    mfcc, spec, mel = get_batch(model.mode, model.batch_size)
                    feed_dict.update({model.x_mfcc: mfcc, model.y_spec: spec, model.y_mel: mel})
                    sess.run(train_op, feed_dict=feed_dict)

            # Write checkpoint files at every epoch
            gs = sess.run(global_step)
            feed_dict = {lr: get_cyclic_lr(gs)}
            summ = sess.run(summ_op, feed_dict=feed_dict)

            if epoch % hp.train2.save_per_epoch == 0:
                tf.train.Saver().save(sess, '{}/epoch_{}_step_{}'.format(logdir2, epoch, gs))

                # Eval at every n epochs
                with tf.Graph().as_default():
                    eval2.eval(logdir2, queue=False, writer=writer)

                # Convert at every n epochs
                with tf.Graph().as_default():
                    convert.convert(logdir2, queue=False, writer=writer)

            writer.add_summary(summ, global_step=gs)

        writer.close()
        coord.request_stop()
        coord.join(threads)


def get_cyclic_lr(step):
    lr_margin = hp.train2.lr_cyclic_margin * math.sin(2. * math.pi / hp.train2.lr_cyclic_steps * step)
    lr = hp.train2.lr + lr_margin
    return lr


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case1', type=str, help='experiment case name of train1')
    parser.add_argument('case2', type=str, help='experiment case name of train2')
    parser.add_argument('-r', action='store_true', help='start training from the beginning.')
    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':
    args = get_arguments()
    hp.set_hparam_yaml(args.case2)
    logdir1 = '{}/{}/train1'.format(hp.logdir_path, args.case1)
    logdir2 = '{}/train2'.format(hp.logdir)

    if args.r:
        ckpt = '{}/checkpoint'.format(os.path.join(logdir2))
        if os.path.exists(ckpt):
            os.remove(ckpt)
            remove_all_files(os.path.join(hp.logdir, 'events.out'))
            remove_all_files(os.path.join(hp.logdir, 'epoch_'))

    print('case1: {}, case2: {}, logdir1: {}, logdir2: {}'.format(args.case1, args.case2, logdir1, logdir2))

    train(logdir1=logdir1, logdir2=logdir2)

    print("Done")