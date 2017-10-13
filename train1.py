# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function
from hyperparams import Hyperparams as hp
from tqdm import tqdm

from modules import *
from models import Model
import eval1
from data_load import get_batch
import argparse


def train(logdir='logdir/train1', queue=True):
    model = Model(mode="train1", batch_size=hp.train1.batch_size, queue=queue)

    # Loss
    loss_op = model.loss_net1()

    # Accuracy
    acc_op = model.acc_net1()

    # Training Scheme
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=hp.train1.lr)
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net1')
    train_op = optimizer.minimize(loss_op, global_step=global_step, var_list=var_list)

    # Summary
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net1'):
        tf.summary.histogram(v.name, v)
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

        for epoch in range(1, hp.train1.num_epochs + 1):
            for step in tqdm(range(model.num_batch), total=model.num_batch, ncols=70, leave=False, unit='b'):
                if queue:
                    sess.run(train_op)
                else:
                    x, y = get_batch(model.mode, model.batch_size)
                    sess.run(train_op, feed_dict={model.x_mfcc: x, model.y_ppgs: y})

            # Write checkpoint files at every epoch
            summ, gs = sess.run([summ_op, global_step])
            writer.add_summary(summ, global_step=gs)

            if epoch % hp.train1.save_per_epoch == 0:
                tf.train.Saver().save(sess, '{}/epoch_{}_step_{}'.format(logdir, epoch, gs))

            # Write eval accuracy at every epoch
            with tf.Graph().as_default():
                eval1.eval(logdir=logdir, queue=False)

        writer.close()
        coord.request_stop()
        coord.join(threads)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', type=str, help='logdir path', default='{}/logdir/train1'.format(hp.logdir_path))
    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':
    args = get_arguments()
    train(logdir=args.logdir)
    print("Done")