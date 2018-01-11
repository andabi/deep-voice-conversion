# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function
import hparams as hp
from hparams import logdir_path
from tqdm import tqdm

from modules import *
from models import Model
import eval1
from data_load import get_batch
import argparse


def train(logdir='logdir/default/train1', queue=True):
    model = Model(mode="train1", batch_size=hp.Train1.batch_size, queue=queue)

    # Loss
    loss_op = model.loss_net1()

    # Accuracy
    acc_op = model.acc_net1()

    # Training Scheme
    global_step = tf.Variable(0, name='global_step', trainable=False)

    optimizer = tf.train.AdamOptimizer(learning_rate=hp.Train1.lr)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net1')
        train_op = optimizer.minimize(loss_op, global_step=global_step, var_list=var_list)

    # Summary
    # for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net1'):
    #     tf.summary.histogram(v.name, v)
    tf.summary.scalar('net1/train/loss', loss_op)
    tf.summary.scalar('net1/train/acc', acc_op)
    summ_op = tf.summary.merge_all()

    session_conf = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
        ),
    )
    # Training
    with tf.Session(config=session_conf) as sess:
        # Load trained model
        sess.run(tf.global_variables_initializer())
        model.load(sess, 'train1', logdir=logdir)

        writer = tf.summary.FileWriter(logdir, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(1, hp.Train1.num_epochs + 1):
            for step in tqdm(range(model.num_batch), total=model.num_batch, ncols=70, leave=False, unit='b'):
                if queue:
                    sess.run(train_op)
                else:
                    mfcc, ppg = get_batch(model.mode, model.batch_size)
                    sess.run(train_op, feed_dict={model.x_mfcc: mfcc, model.y_ppgs: ppg})

            # Write checkpoint files at every epoch
            if queue:
                summ, gs = sess.run([summ_op, global_step])
            else:
                summ, gs = sess.run([summ_op, global_step], feed_dict={model.x_mfcc: mfcc, model.y_ppgs: ppg})
            
            if epoch % hp.Train1.save_per_epoch == 0:
                tf.train.Saver().save(sess, '{}/epoch_{}_step_{}'.format(logdir, epoch, gs))

            # Write eval accuracy at every epoch
            with tf.Graph().as_default():
                eval1.eval(logdir=logdir, queue=False)

            writer.add_summary(summ, global_step=gs)

        writer.close()
        coord.request_stop()
        coord.join(threads)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name')
    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':
    args = get_arguments()
    case = args.case
    logdir = '{}/{}/train1'.format(logdir_path, case)
    train(logdir=logdir)
    print("Done")
