# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function

from hyperparams import Hyperparams as hp
from tqdm import tqdm

import tensorflow as tf
from models import Model


def main(logdir='logdir/train2'):
    model = Model(mode="train2")

    # Loss
    loss_op = model.loss_net2()

    # Training Scheme
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=hp.train.lr)
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net2')
    train_op = optimizer.minimize(loss_op, global_step=global_step, var_list=var_list)

    # Summary
    summ_op = summaries(loss_op)

    session_conf = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
        ),
    )
    # Training
    with tf.Session(config=session_conf) as sess:
        # Load trained model
        sess.run(tf.global_variables_initializer())
        model.load_variables(sess, 'train2', logdir=logdir)

        writer = tf.summary.FileWriter(logdir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(1, hp.train.num_epochs + 1):
            for step in tqdm(range(model.num_batch), total=model.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(train_op)

            # Write checkpoint files at every epoch
            summ, gs = sess.run([summ_op, global_step])
            writer.add_summary(summ, global_step=gs)

            if epoch % hp.train.save_per_epoch == 0:
                tf.train.Saver().save(sess, '{}/epoch_{}_step_{}'.format(logdir, epoch, gs))

        coord.request_stop()
        coord.join(threads)


def summaries(loss):
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net2'):
        tf.summary.histogram(v.name, v)
    tf.summary.scalar('net2/train/loss', loss)
    return tf.summary.merge_all()


if __name__ == '__main__':
    main()
    print("Done")

# var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net')
# sv = tf.train.Supervisor(logdir='logdir/train2', save_model_secs=0, init_op=tf.variables_initializer(var_list))
# with sv.managed_session(config=session_conf) as sess:
#     for epoch in range(1, hp.num_epochs + 1):
#         for step in tqdm(range(model.num_batch), total=model.num_batch, ncols=70, leave=False, unit='b'):
#             sess.run(train_op)
#
#         # Write checkpoint files at every epoch
#         summ, gs = sess.run([summ_op, global_step])
#         sv.summary_computed(sess, summ, global_step=gs)
#
#         if epoch % 5 == 0:
#             tf.train.Saver().save(sess, 'logdir/train2/step_%d' % gs)
