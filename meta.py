from data import Dataset
import batch_evals2 as evals2
from model import Model
import teach

import numpy as np
from torch import optim
from torch.optim import lr_scheduler as opt_sched
from matplotlib import pyplot as plt
from pandas import DataFrame
import seaborn as sns
import scipy
from tensorboardX import SummaryWriter
from absl import logging, flags
import os
from tb_utils import parse_tb_event_files
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', default=128, help='batch size')
flags.DEFINE_integer('epochs', default=20, help='epoch train')
flags.DEFINE_integer('batchs_per_epoch', default=70, help='batchs in each epoch')
flags.DEFINE_integer('teach_epochs', default=5, help='teaching epoch')


def unwrap(var):
    return var.data.cpu().numpy()


def info(reps):
    buckets = np.zeros((64, 30))
    for rep in reps:
        for i in range(len(rep)):
            bucket = 15 + int(rep[i] * 30)
            bucket = max(bucket, 0)
            bucket = min(bucket, 29)
            buckets[i, bucket] += 1
    buckets += 1e-7
    probs = buckets / buckets.sum(axis=1, keepdims=True)
    logprobs = np.log(probs)
    entropies = -(probs * logprobs).sum(axis=1)
    return entropies.mean()


TRN_LOSS = 'trn_loss'
TRN_ACC = 'trn_acc'
VAL_LOSS = 'val_loss'
VAL_ACC = 'val_acc'
CVAL_ACC = 'cval_acc'
INFO_TX = 'MI'
ISOM = 'isom'
HOM = 'hom'
HOM_WO_MEAN = 'hom_wo_mean'
CHOM = 'c_hom'
LB = 'learnability'


def validate(dataset, model):
    model.eval()
    metrics = {}

    # Learnability
    lb = teach.get_learnability(dataset, teacher=model)
    metrics[LB] = lb

    val_batch = dataset.get_val_batch()
    val_loss, val_acc, _, val_reps = model(val_batch)
    val_acc = val_acc.item()
    val_loss = val_loss.item()
    metrics[VAL_ACC] = val_acc
    metrics[VAL_LOSS] = val_loss

    cval_batch = dataset.get_cval_batch()
    _, cval_acc, _, cval_reps = model(cval_batch)
    cval_acc = cval_acc.item()
    metrics[CVAL_ACC] = cval_acc

    prim_batch = dataset.get_prim_batch()
    _, _, _, prim_reps = model(prim_batch)

    prim_rseq = [unwrap(prim_reps[i, ...]) for i in range(prim_reps.shape[0])]
    val_rseq = [unwrap(val_reps[i, ...]) for i in range(val_reps.shape[0])]

    comp = evals2.evaluate(
        reps=prim_rseq + val_rseq,
        exprs=prim_batch.lf + val_batch.lf, quiet=True,
        subtract_mean=False
    )
    metrics[HOM] = np.mean(comp)

    comp = evals2.evaluate(
        reps=prim_rseq + val_rseq,
        exprs=prim_batch.lf + val_batch.lf, quiet=True,
        subtract_mean=True,
    )
    metrics[HOM_WO_MEAN] = np.mean(comp)

    info_tx = info(unwrap(val_reps))
    metrics[INFO_TX] = info_tx
    return metrics


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train(dataset, model, tb_writer):
    opt = optim.Adam(model.parameters(), lr=5e-4)
    sched = opt_sched.ReduceLROnPlateau(opt, factor=0.5, verbose=True, mode='max')

    epoch = 0
    while True:
        model.eval()
        val_metric = validate(dataset, model)
        logging_str = ['VAL[epoch={}]'.format(epoch)]
        for key, val in val_metric.items():
            logging_str.append("{}:{:.4f}".format(key, val))
            tb_writer.add_scalar('{}'.format(key), val, epoch)
        logging.info(' '.join(logging_str))

        # Train loop
        if epoch == FLAGS.epochs:
            logging.info('Reach maximum epoch')
            break

        trn_loss = 0
        trn_acc = 0
        model.train()
        for j in range(FLAGS.batchs_per_epoch):
            batch = dataset.get_train_batch(FLAGS.batch_size)
            loss, acc, _, _ = model(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            trn_loss += loss.item()
            trn_acc += acc.item()
        epoch += 1
        trn_loss /= FLAGS.batchs_per_epoch
        trn_acc /= FLAGS.batchs_per_epoch
        
        sched.step(val_metric[VAL_ACC])

        train_metric = {TRN_LOSS: trn_loss,
                        TRN_ACC: trn_acc,
                        'lr': get_lr(opt)}
        logging_str = ['TRAIN[epoch={}]'.format(epoch)]
        for key, val in train_metric.items():
            logging_str.append("{}:{:.4f}".format(key, val))
            tb_writer.add_scalar('{}'.format(key), val, epoch)
        logging.info(' '.join(logging_str))


def run(training_folder):
    logs = []
    dataset = Dataset()
    writer = SummaryWriter(os.path.join(training_folder, 'logs'))
    model = Model()
    if FLAGS.cuda:
        model = model.cuda()
    train(dataset, model, writer)
    writer.close()

    ## Create
    #sns.set(font_scale=1.5)
    #sns.set_style("ticks", {'font.family': 'serif'})
    #plt.tight_layout()

    #new_logs = [[(epoch, info_x, tre, val_acc, lb, val_loss, tre_no_mean)
    #             for epoch, info_x, tre, val_acc, lb, val_loss, tre_no_mean in zip(log[LB]['steps'], log[INFO_TX]['values'],
    #                                                                               log[HOM]['values'], log[VAL_ACC]['values'],
    #                                                                               log[LB]['values'],
    #                                                                               log[VAL_LOSS]['values'],
    #                                                                               log[HOM_WO_MEAN]['values'])]
    #             for log in logs]
    #log = sum(new_logs, [])
    #data = DataFrame(np.asarray(log), columns=['epoch', 'MI', 'TRE', 'val_acc',
    #                                           'LB', 'val_loss', 'TRE_no_mean'])
    #tobeplot = data.columns[1:]
    #for x_id in range(len(tobeplot)):
    #    for y_id in range(x_id + 1, len(tobeplot)):
    #        names = [tobeplot[x_id], tobeplot[y_id]]
    #        names = sorted(names)
    #        x_name, y_name = names
    #        sns.lmplot(x=x_name, y=y_name, data=data)
    #        logging.info("pearson coeffcient {} {}: {}".format(
    #            x_name, y_name,
    #            scipy.stats.pearsonr(data[x_name], data[y_name])))
    #        plt.savefig(os.path.join(training_folder, '{}_{}.pdf'.format(x_name, y_name)),
    #                    format='pdf')
