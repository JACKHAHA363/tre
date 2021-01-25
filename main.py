"""
Main entry & setup logging
"""
import sys
from absl import flags, logging
from absl import app
import time
import datetime
import traceback
import string
import torch
import random
import os
import git
import meta

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
EXP_FOLDER = os.path.join(ROOT_DIR, 'experiments')
repo = git.Repo(path=ROOT_DIR)
sha = repo.head.object.hexsha
CURR_VERSION = sha[:10]


FLAGS = flags.FLAGS

flags.DEFINE_bool('debug', default=False, help='Flag for debug mode')
flags.DEFINE_enum('mode', default='meta', enum_values=['meta'],
                  help='choosing modes')
flags.DEFINE_string('experiment', default=None, help='Name of experiment')
flags.DEFINE_bool('cuda', default=False, help='Use cuda')


def handler(type, value, tb):
    logging.exception("Uncaught exception: %s", str(value))
    logging.exception("\n".join(traceback.format_exception(type, value, tb)))


def random_string():
    return ''.join(random.sample(string.ascii_lowercase + string.ascii_uppercase, k=5))


def setup_logging_and_exp_folder():
    # Use time stamp or user specified if not debug
    ts = time.time()
    FLAGS.experiment = FLAGS.experiment if FLAGS.experiment is not None else \
        "{}_{}".format(FLAGS.mode,
                       datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M'))

    # Random string if debug
    if FLAGS.debug:
        FLAGS.experiment = "{}_debug_{}".format(FLAGS.experiment, random_string())

    training_folder = os.path.join(EXP_FOLDER, FLAGS.experiment)

    # Create train folder
    if os.path.exists(training_folder):
        print('{} exists!'.format(training_folder))
        exit(-1)
    else:
        os.makedirs(training_folder, exist_ok=False)

    # set up logging
    if FLAGS.debug:
        logging.get_absl_handler().python_handler.stream = sys.stdout
    else:
        logging.get_absl_handler().use_absl_log_file('absl_logging', training_folder)
    return training_folder


def main(_):
    training_folder = setup_logging_and_exp_folder()
    FLAGS.cuda = FLAGS.cuda and torch.cuda.is_available()
    logging.info('Use Cuda: {}'.format(FLAGS.cuda))
    logging.info('Current git SHA: ' + CURR_VERSION)

    # save options
    fpath = os.path.join(training_folder, 'flagfile')
    with open(fpath, 'w') as f:
        f.write(FLAGS.flags_into_string())

    meta.run(training_folder)
    logging.info('Done')


if __name__ == '__main__':
    FLAGS(sys.argv)
    app.run(main)
