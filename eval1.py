import argparse

from tensorpack.compat import is_tfv2, tfv1 as tf
from tensorpack.predict.base import OfflinePredictor
from tensorpack.predict.config import PredictConfig
from tensorpack.tfutils.sessinit import SaverRestore

from data_load import Net1DataFlow, phns, load_vocab
from hparam import hparam as hp
from models import Net1
from utils import plot_confusion_matrix


def get_eval_input_names():
    return ['x_mfccs', 'y_ppgs']


def get_eval_output_names():
    return ['net1/eval/y_ppg_1d', 'net1/eval/pred_ppg_1d',  'net1/eval/summ_loss', 'net1/eval/summ_acc']


def eval(logdir):
    if is_tfv2(): 
        tf.disable_v2_behavior()

    # Load graph
    model = Net1()

    # dataflow
    df = Net1DataFlow(hp.test1.data_path, hp.test1.batch_size)

    ckpt = tf.train.latest_checkpoint(logdir)

    pred_conf = PredictConfig(
        model=model,
        input_names=get_eval_input_names(),
        output_names=get_eval_output_names())
    if ckpt:
        pred_conf.session_init = SaverRestore(ckpt)
    predictor = OfflinePredictor(pred_conf)

    y_ppg_1d, pred_ppg_1d, summ_loss, summ_acc = predictor(*next(df().ds.get_data()))

    # plot confusion matrix
    _, idx2phn = load_vocab()
    y_ppg_1d = [idx2phn[i] for i in y_ppg_1d]
    pred_ppg_1d = [idx2phn[i] for i in pred_ppg_1d]
    summ_cm = plot_confusion_matrix(y_ppg_1d, pred_ppg_1d, phns)

    writer = tf.summary.FileWriter(logdir)
    writer.add_summary(summ_loss)
    writer.add_summary(summ_acc)
    writer.add_summary(summ_cm)
    writer.close()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    hp.set_hparam_yaml(args.case)
    logdir = '{}/train1'.format(hp.logdir)
    eval(logdir=logdir)

    print("Done")