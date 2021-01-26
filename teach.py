from model import Model
import torch.optim as optim
import torch.optim.lr_scheduler as opt_sched
import torch
from absl import flags, logging

FLAGS = flags.FLAGS
ZERO = 1e-32


def normalize(vec):
    return vec / (vec.norm(p=2, dim=1, keepdim=True) + ZERO)


def get_loss(repr1, repr2):
    """ Unreduced loss """
    teach_repr = normalize(repr1)
    student_repr = normalize(repr2)
    return -(student_repr * teach_repr).sum(dim=1)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_learnability(dataset, teacher):
    teacher.eval()
    student = Model()
    if FLAGS.cuda:
        student.cuda()
    opt = optim.Adam(student.parameters(), lr=5e-4)
    sched = opt_sched.ReduceLROnPlateau(opt, factor=0.5, verbose=True, mode='max')

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

    best_lb = -10
    for i in range(FLAGS.teach_epochs):
        student.eval()
        val_lb = val_loop(dataset, teacher, student, mean_repr)
        if val_lb > best_lb:
            best_lb = val_lb
        logging.info('val_lb: {:.4f} best_lb: {:.4f} lr: {:.4f}'.format(val_lb,
                                                                        best_lb, 
                                                                        get_lr(opt)))

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


def train_loop(dataset, teacher, student, opt, mean_repr):
    trn_lb = 0
    count = 0

    for _ in range(FLAGS.batchs_per_epoch):
        batch = dataset.get_train_batch(FLAGS.batch_size)
        with torch.no_grad():
            teacher_repr = teacher.get_repr(batch) - mean_repr
        student_repr = student.get_repr(batch)
        loss = get_loss(teacher_repr, student_repr)
        opt.zero_grad()
        loss.mean().backward()
        opt.step()

        trn_lb += -loss.sum().item()
        count += loss.shape[0]

    trn_lb /= count
    return trn_lb


def val_loop(dataset, teacher, student, mean_repr):
    val_lb = 0
    count = 0
    with torch.no_grad():
        batch = dataset.get_val_batch()
        with torch.no_grad():
            teacher_repr = teacher.get_repr(batch) - mean_repr
        student_repr = student.get_repr(batch)
        loss = get_loss(teacher_repr, student_repr)
        val_lb += -loss.sum().item()
        count += teacher_repr.shape[0]
        val_lb /= count
    return val_lb
