from util import flatten

import torch
from torch import nn
from torch import optim

class L1Dist(nn.Module):
    def forward(self, pred, target):
        return torch.abs(pred - target).sum(-1)

class CosDist(nn.Module):
    def forward(self, x, y):
        nx, ny = nn.functional.normalize(x), nn.functional.normalize(y)
        return 1 - (nx * ny).sum(-1)

class Objective(nn.Module):
    def __init__(self, vocab, repr_size, comp_fn, err_fn, zero_init):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), repr_size)
        if zero_init:
            self.emb.weight.data.zero_()
        self.comp = comp_fn
        self.err = err_fn

    def compose(self, e):
        if isinstance(e, tuple) and len(e) > 1:
            args = (self.compose(ee) for ee in e)
            return self.comp(*args)
        if isinstance(e, tuple) and len(e) == 1:
            e = e[0]
        return self.emb(e)

    def forward(self, rep, expr):
        return self.err(self.compose(expr), rep)

def evaluate(reps, exprs, comp_fn, err_fn, quiet=False, steps=400, include_pred=False, zero_init=True):
    vocab = {}
    for expr in exprs:
        toks = flatten(expr)
        for tok in toks:
            if tok not in vocab:
                vocab[tok] = len(vocab)

    def index(e):
        if isinstance(e, tuple):
            return tuple(index(ee) for ee in e)
        return torch.LongTensor([vocab[e]])

    treps = [torch.FloatTensor([r]) for r in reps]
    texprs = [index(e) for e in exprs]
    final_errs = torch.zeros(len(treps)).to(device=treps[0].device)

    nb_tuples = set([len(expr) for expr in texprs])
    for nb_tuple in nb_tuples:
        selected_id = [idx for idx, expr in enumerate(texprs)
                       if len(expr) == nb_tuple]

        batch_reps = torch.cat([treps[i] for i in selected_id])
        batch_exprs = [torch.LongTensor([texprs[i][tuple_id].item()
                                     for i in selected_id])
                       for tuple_id in range(nb_tuple)]
        batch_exprs = tuple(batch_exprs)

        obj = Objective(vocab, batch_reps[0].shape[0], comp_fn, err_fn, zero_init)
        opt = optim.RMSprop(obj.parameters(), lr=0.01)
        for t in range(steps):
            opt.zero_grad()
            errs = obj(batch_reps, batch_exprs)
            loss = sum(errs)
            loss.backward()
            if not quiet and t % 100 == 0:
                print(loss.item())
            opt.step()
        final_errs[selected_id] = errs


    #for r, e in zip(treps, texprs):
    #    print(r, obj.compose(e))
    #assert False
    final_errs = [err.item() for err in final_errs]
    if include_pred:
        lexicon = {
            k: obj.emb(torch.LongTensor([v])).data.cpu().numpy()
            for k, v in vocab.items()
        }
        composed = [obj.compose(t) for t in texprs]
        return final_errs, lexicon, composed
    else:
        return final_errs
