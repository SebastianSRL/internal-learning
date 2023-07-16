from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()


_C.TRAIN = CN()
_C.TRAIN.batch_size = 1
_C.TRAIN.epochs = 3000
_C.TRAIN.lr = 0.0001
_C.TRAIN.subrate = 0.5
_C.TRAIN.subtype = "random"
_C.TRAIN.seed = 42

_C.DATASET = CN()
_C.DATASET.data_path = "data/syn3D_cross-spread.npy"
_C.DATASET.phi = [2, 4, 6, 8, 10, 12]


def get_default_config():
    return _C.clone()
