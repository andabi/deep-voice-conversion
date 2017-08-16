# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/vc
'''

from __future__ import print_function
from hyperparams import Hyperparams as hp
from tqdm import tqdm

from modules import *
from models import Model


def summaries(loss):
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net1'):
        tf.summary.histogram(v.name, v)
    tf.summary.scalar('loss_net1', loss)
    return tf.summary.merge_all()


# TODO refactoring
def main():
    model = Model(mode="train1")
    print("Training Graph loaded")

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net1')

    # Training Scheme
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
    train_op = optimizer.minimize(model.loss_net1(), global_step=global_step, var_list=var_list)

    # Summary
    summ_op = summaries(model.loss_net1())

    # Training
    sv = tf.train.Supervisor(logdir="logdir/train1",
                             save_model_secs=0)
    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs + 1):
            if sv.should_stop(): break
            for step in tqdm(range(model.num_batch), total=model.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(train_op)

            summ, gs = sess.run([summ_op, global_step])
            sv.summary_computed(sess, summ)

            sv.saver.save(sess, 'logdir/train1/epoch_%02d_gs_%d' % (epoch, gs))

if __name__ == '__main__':
    main()
    print("Done")
