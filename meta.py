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

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', default=128, help='batch size')
flags.DEFINE_integer('epochs', default=20, help='epoch train')
flags.DEFINE_integer('runs', default=10, help='nb runs')
flags.DEFINE_integer('batchs_per_epoch', default=100, help='batchs in each epoch')
flags.DEFINE_integer('teach_epochs', default=10, help='teaching epoch')


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


EPOCH = 'epoch'
TRN_LOSS = 'trn loss'
TRN_ACC = 'trn acc'
VAL_ACC = 'val acc'
CVAL_ACC = 'cval acc'
INFO_TX = 'I(T;X)'
ISOM = 'isom'
HOM = 'hom'
CHOM = 'c_hom'
LB = 'learnabiliy'


def validate(dataset, model):
    model.eval()
    metrics = {}

    # Learnability
    lb = teach.get_learnability(dataset, teacher=model)
    metrics[LB] = lb

    val_batch = dataset.get_val_batch()
    _, val_acc, _, val_reps = model(val_batch)
    val_acc = val_acc.item()
    metrics[VAL_ACC] = val_acc

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
        exprs=prim_batch.lf + val_batch.lf, quiet=True)
    metrics[HOM] = np.mean(comp)

    info_tx = info(unwrap(val_reps))
    metrics[INFO_TX] = info_tx
    return metrics


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train(dataset, model, tb_writer):
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sched = opt_sched.ReduceLROnPlateau(opt, factor=0.5, verbose=True, mode='max')
    logs = {}
    for i in range(FLAGS.epochs):
        model.eval()
        val_metric = validate(dataset, model)
        logging_str = ['VAL[epoch={}]'.format(i)]
        for key, val in val_metric.items():
            logging_str.append("{}:{:.4f}".format(key, val))
            tb_writer.add_scalar('val/{}'.format(key), val, i)
        logging.info(' '.join(logging_str))

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
        
        trn_loss /= FLAGS.batchs_per_epoch
        trn_acc /= FLAGS.batchs_per_epoch
        
        sched.step(val_metric[VAL_ACC])

        train_metric = {TRN_LOSS: trn_loss,
                        TRN_ACC: trn_acc,
                        'lr': get_lr(opt)}
        logging_str = ['TRAIN[epoch={}]'.format(i)]
        for key, val in train_metric.items():
            logging_str.append("{}:{:.4f}".format(key, val))
            tb_writer.add_scalar('train/{}'.format(key), val, i)
        logging.info(' '.join(logging_str))

        import ipdb; ipdb.set_trace()
        metric = {**val_metric, **train_metric}
        logs.append(metric)
    return logs


def run(training_folder):
    logs = []
    dataset = Dataset()
    for i in range(FLAGS.runs):
        writer = SummaryWriter(os.path.join(training_folder, 'train/run{}'.format(i)))
        model = Model()
        if FLAGS.cuda:
            model = model.cuda()
        log = train(dataset, model, writer)
        logs.append(log)
    sns.set(font_scale=1.5)
    sns.set_style("ticks", {'font.family': 'serif'})
    plt.tight_layout()

    my_logs = logs
    log = sum(my_logs, [])
    data = DataFrame(np.asarray(log), columns=['epoch', 'I(θ;X)', 'TRE', 'val', 'learnability'])
    sns.lmplot(x='I(θ;X)', y='TRE', data=data)
    print(scipy.stats.pearsonr(data['I(θ;X)'], data['TRE']))
    plt.savefig('info_tre.pdf', format='pdf')

    sns.lmplot(x='I(θ;X)', y='learnability', data=data)
    print(scipy.stats.pearsonr(data['I(θ;X)'], data['learnability']))
    plt.savefig('info_lb.pdf', format='pdf')

    sns.lmplot(x='TRE', y='learnability', data=data)
    print(scipy.stats.pearsonr(data['TRE'], data['learnability']))
    plt.savefig('tre_lb.pdf', format='pdf')



