# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function

from hyperparams import Hyperparams as hp
from tqdm import tqdm

import tensorflow as tf
from models import Model
import convert, eval2
from data_load import get_batch


def main(logdir1='logdir/train1', logdir2='logdir/train2', queue=True):
    model = Model(mode="train2", batch_size=hp.train2.batch_size, queue=queue)

    # Loss
    loss_op = model.loss_net2()

    # Training Scheme
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=hp.train2.lr)
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net2')
    train_op = optimizer.minimize(loss_op, global_step=global_step, var_list=var_list)

    # Summary
    summ_op = summaries(loss_op)

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

        writer = tf.summary.FileWriter(logdir2, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(1, hp.train2.num_epochs + 1):
            for step in tqdm(range(model.num_batch), total=model.num_batch, ncols=70, leave=False, unit='b'):
                if queue:
                    sess.run(train_op)
                else:
                    x, y = get_batch(model.mode, model.batch_size)
                    sess.run(train_op, feed_dict={model.x_mfcc: x, model.y_spec: y})

            # Write checkpoint files at every epoch
            summ, gs = sess.run([summ_op, global_step])
            writer.add_summary(summ, global_step=gs)

            if epoch % hp.train2.save_per_epoch == 0:
                tf.train.Saver().save(sess, '{}/epoch_{}_step_{}'.format(logdir2, epoch, gs))

                # Eval at every n epochs
                with tf.Graph().as_default():
                    eval2.eval(logdir2, queue=False)

                # Convert at every n epochs
                with tf.Graph().as_default():
                    convert.convert(logdir2, queue=False)

        writer.close()
        coord.request_stop()
        coord.join(threads)


def summaries(loss):
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net2'):
        tf.summary.histogram(v.name, v)
    tf.summary.scalar('net2/train/loss', loss)
    return tf.summary.merge_all()


if __name__ == '__main__':
    main(logdir1='logdir/train1', logdir2='logdir/train2', queue=True)
    print("Done")