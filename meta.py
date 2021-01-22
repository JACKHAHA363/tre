from cls2_data.data import Dataset
import batch_evals2 as evals2
from model import Model
from util import Logger
import teach

import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler as opt_sched
from matplotlib import pyplot as plt
from pandas import DataFrame
import seaborn as sns
import scipy


SEED = 0
HAS_CUDA = torch.cuda.is_available()
np_random = np.random.seed(SEED)
N_BATCH = 128
N_EPOCH = 20


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
LOG_KEYS = [EPOCH, TRN_LOSS, TRN_ACC, VAL_ACC, HOM,   ISOM,  INFO_TX, LB]
LOG_FMTS = ['d',   '.3f',    '.3f',   '.3f',   '.3f', '.3f', '.3f', '.3f']


class Composition(nn.Module):
    def forward(self, x, y):
        return x + y


comp_fn = Composition()
err_fn = evals2.CosDist()


def validate(dataset, model, logger, plot_log, epoch):
    # Learnability
    lb = teach.get_learnability(dataset, teacher=model)
    logger.update(LB, lb)

    val_batch = dataset.get_val_batch()
    _, val_acc, _, val_reps = model(val_batch)
    val_acc = val_acc.item()
    logger.update(VAL_ACC, val_acc)
    
    cval_batch = dataset.get_cval_batch()
    _, cval_acc, _, cval_reps = model(cval_batch)
    cval_acc = cval_acc.item()
    logger.update(CVAL_ACC, cval_acc)
    
    prim_batch = dataset.get_prim_batch()
    _, _, _, prim_reps = model(prim_batch)
    
    prim_rseq = [unwrap(prim_reps[i, ...]) for i in range(prim_reps.shape[0])]
    val_rseq = [unwrap(val_reps[i, ...]) for i in range(val_reps.shape[0])]
    cval_rseq = [unwrap(cval_reps[i, ...]) for i in range(cval_reps.shape[0])]
    
    comp = evals2.evaluate(
        prim_rseq + val_rseq, prim_batch.lf + val_batch.lf,
        comp_fn, err_fn, quiet=True)
    logger.update(HOM, np.mean(comp))
    
    #ccomp = evals2.evaluate(
    #    prim_rseq + cval_rseq, prim_batch.lf + cval_batch.lf,
    #    comp_fn, err_fn)[-len(cval_rseq):]
    #logger.update(CHOM, np.mean(ccomp))
    
    #logger.update(ISOM, eval_isom_tree(unwrap(val_reps), val_batch.lf))
    #info_tx = info(unwrap(nn.functional.tanh(val_reps)))
    info_tx = info(unwrap(val_reps))
    logger.update(INFO_TX, info_tx)
    
    plot_log.append((epoch, info_tx, np.mean(comp), val_acc, lb))
    return val_acc


def train(dataset, model):
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sched = opt_sched.ReduceLROnPlateau(opt, factor=0.5, verbose=True, mode='max')
    logger = Logger(LOG_KEYS, LOG_FMTS, width=10)
    logger.begin()
    plot_log = []
    validate(dataset, model, logger, plot_log, -1)
    logger.print()
    import ipdb; ipdb.set_trace()
    for i in range(N_EPOCH):
        trn_loss = 0
        trn_acc = 0
        for j in range(100):
            batch = dataset.get_train_batch(N_BATCH)
            loss, acc, _, _ = model(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            trn_loss += loss.item()
            trn_acc += acc.item()
        
        trn_loss /= 100
        trn_acc /= 100
        
        logger.update(EPOCH, i)
        logger.update(TRN_LOSS, trn_loss)
        logger.update(TRN_ACC, trn_acc)
        val_acc = validate(dataset, model, logger, plot_log, i)
        sched.step(val_acc)
        logger.print()
    return plot_log


logs = []
dataset = Dataset()
for i in range(1):
    model = Model()
    if HAS_CUDA:
        model = model.cuda()
    log = train(dataset, model)
    logs.append(log)
import ipdb; ipdb.set_trace()
sns.set(font_scale=1.5)
sns.set_style("ticks", {'font.family': 'serif'})
plt.tight_layout()
#cmap = sns.color_palette("coolwarm", 10)

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



