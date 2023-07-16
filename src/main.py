import argparse

import numpy as np
import tensorflow as tf

from default import get_default_config
from models import Baseline
from preprocessing import gen_training_samples
from utils import random_sampling, rescale, uniform_sampling

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))


def parser():
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="src/config.yaml")
    return args.parse_args()


def main(cfg):
    np.random.seed(cfg.TRAIN.seed)
    tf.random.set_seed(cfg.TRAIN.seed)

    y_test = np.load(cfg.DATASET.data_path)
    y_test = rescale(y_test)

    if cfg.DATASET.phi is not None:
        x_test = y_test.copy()
        phi = cfg.DATASET.phi
        x_test[:, phi] = 0
    else:
        subsampling = (
            random_sampling if cfg.TRAIN.subtype == "random" else uniform_sampling
        )
        x_test, phi = subsampling(y_test, cfg.TRAIN.subrate, testing=True)

    x_train, y_train = gen_training_samples(
        x_test, phi, cfg.TRAIN.subrate, cfg.TRAIN.subtype
    )

    model = Baseline(cfg.MODEL.nfilters, cfg.MODEL.ksize, cfg.MODEL.depth)


if __name__ == "__main__":
    args = parser()
    cfg = get_default_config()
    cfg.merge_from_file(args.config)
    cfg.freeze()
    main(cfg)
