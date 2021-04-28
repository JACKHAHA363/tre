from model import Model
import torch.optim as optim
import torch.optim.lr_scheduler as opt_sched
import torch
import numpy as np
from absl import flags, logging
import torch.nn.functional as F

FLAGS = flags.FLAGS
ZERO = 1e-32


def normalize(vec):
    return vec / (vec.norm(p=2, dim=1, keepdim=True) + ZERO)


def get_sim(repr1, repr2):
    """ Unreduced loss """
    return F.cosine_similarity(repr1, repr2)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_learnability(dataset, teacher, zero_mean=True):
    teacher.eval()
    mean_repr = get_mean_repr(dataset, teacher) if zero_mean else None
    best_lb = -10
    student = Model()
    if FLAGS.cuda:
        student.cuda()
    opt = optim.Adam(student.parameters(), lr=5e-4)
    sched = opt_sched.ReduceLROnPlateau(opt, factor=0.5, verbose=True, mode='max')
    for i in range(FLAGS.teach_epochs):
        student.eval()
        val_lb = val_loop(dataset, teacher, student, mean_repr)
        if val_lb > best_lb:
            best_lb = val_lb
        student.train()
        _ = train_loop(dataset, teacher, student, opt, mean_repr)
        sched.step(val_lb)

    student.eval()
    val_lb = val_loop(dataset, teacher, student, mean_repr)
    if val_lb > best_lb:
        best_lb = val_lb
    logging.info('val_lb: {:.4f} best_lb: {:.4f} lr: {:.4f}'.format(val_lb,
                                                                    best_lb,
                                                                    get_lr(opt)))
    return best_lb


def get_mean_repr(dataset, teacher):
    # Get mean train vec
    mean_repr = 0
    count = 0
    with torch.no_grad():
        for _ in range(FLAGS.batchs_per_epoch):
            batch = dataset.get_train_batch(FLAGS.batch_size)
            res = teacher.get_repr(batch)
            mean_repr += res.sum(0)
            count += res.shape[0]
    mean_repr /= count
    return mean_repr


def train_loop(dataset, teacher, student, opt, mean_repr=None):
    trn_lb = 0
    count = 0

    for _ in range(FLAGS.batchs_per_epoch):
        batch = dataset.get_train_batch(FLAGS.batch_size)
        with torch.no_grad():
            teacher_repr = teacher.get_repr(batch)
            if mean_repr is not None:
                teacher_repr = teacher_repr - mean_repr
        student_repr = student.get_repr(batch)
        sims = get_sim(teacher_repr, student_repr)
        opt.zero_grad()
        (-sims).mean().backward()
        opt.step()

        trn_lb += sims.sum().item()
        count += sims.shape[0]
        if FLAGS.debug:
            break

    trn_lb /= count
    return trn_lb


def val_loop(dataset, teacher, student, mean_repr=None):
    val_lb = 0
    count = 0
    with torch.no_grad():
        batch = dataset.get_val_batch()
        with torch.no_grad():
            teacher_repr = teacher.get_repr(batch)
            if mean_repr is not None:
                teacher_repr = teacher_repr - mean_repr
        student_repr = student.get_repr(batch)
        sims = get_sim(teacher_repr, student_repr)
        val_lb += sims.sum().item()
        count += teacher_repr.shape[0]
        val_lb /= count
    return val_lb
