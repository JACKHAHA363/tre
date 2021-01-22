from model import Model
import torch.optim as optim
import torch.optim.lr_scheduler as opt_sched
import torch

N_BATCH = 128
ZERO = 1e-32
TEACH_EPOCH = 5
HAS_CUDA = torch.cuda.is_available()


def normalize(vec):
    return vec / (vec.norm(p=2, dim=1, keepdim=True) + ZERO)

def get_loss(repr1, repr2):
    """ Unreduced loss """
    teach_repr = normalize(repr1)
    student_repr = normalize(repr2)
    return -(student_repr * teach_repr).sum(dim=1)


def get_learnability(dataset, teacher):
    teacher.eval()
    student = Model()
    if HAS_CUDA:
        student.cuda()
    opt = optim.Adam(student.parameters(), lr=1e-3)
    sched = opt_sched.ReduceLROnPlateau(opt, factor=0.5, verbose=True, mode='max')

    # Get mean train vec
    mean_repr = 0
    count = 0
    with torch.no_grad():
        for _ in range(100):
            batch = dataset.get_train_batch(N_BATCH)
            res = teacher.get_repr(batch)
            mean_repr += res.sum(0)
            count += res.shape[0]

    best_lb = -10
    for i in range(TEACH_EPOCH):
        student.eval()
        val_lb = val_loop(dataset, teacher, student, mean_repr)
        if val_lb > best_lb:
            best_lb = val_lb

        student.train()
        _ = train_loop(dataset, teacher, student, opt, mean_repr)
        sched.step(val_lb)
    return best_lb


def train_loop(dataset, teacher, student, opt, mean_repr):
    trn_lb = 0

    for _ in range(100):
        batch = dataset.get_train_batch(N_BATCH)
        with torch.no_grad():
            teacher_repr = teacher.get_repr(batch) - mean_repr
        student_repr = student.get_repr(batch)
        loss = get_loss(teacher_repr, student_repr)
        opt.zero_grad()
        loss.mean().backward()
        opt.step()

        trn_lb += -loss.mean().item()

    trn_lb /= 100
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
