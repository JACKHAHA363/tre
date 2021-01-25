import tensorflow as tf
import os


def get_empty_series(steps):
    result = Series()
    for step in steps:
        result.add(step, 0)
    return result


class Series:
    def __init__(self):
        self.values = []
        self.steps = []

    def add(self, step, val):
        """ Insert step and value. Maintain sorted w.r.t. steps """
        if len(self.steps) == 0:
            self.steps.append(step)
            self.values.append(val)
        else:
            for idx in reversed(range(len(self.steps))):
                if step > self.steps[idx]:
                    break
            else:
                idx = -1
            self.steps.insert(idx + 1, step)
            self.values.insert(idx + 1, val)

    def verify(self):
        for i in range(len(self.steps) - 1):
            assert self.steps[i] <= self.steps[i + 1]

    def to_dict(self):
        self.verify()
        return {'steps': self.steps,
                'values': self.values}

def get_empty_series(steps):
    result = Series()
    for step in steps:
        result.add(step, 0)
    return result


def parse_tb_event_files(event_dir, tags=None):
    data = {}
    event_files = [os.path.join(event_dir, fname) for fname in os.listdir(event_dir)
                   if 'events.' in fname and not os.path.isdir(fname)]
    print('Found {} event file'.format(len(event_files)))
    for event_file in event_files:
        for e in tf.compat.v1.train.summary_iterator(event_file):
            for v in e.summary.value:
                tag = v.tag.replace('/', '_')
                if tags is None or tag in tags:
                    if data.get(tag) is None:
                        data[tag] = Series()
                    data[tag].add(step=e.step, val=v.simple_value)
    for tag in data:
        data[tag].verify()
        steps = data[tag].steps

    if tags is not None:
        for tag in tags:
            if tag not in data:
                data[tag] = get_empty_series(steps)
    return data