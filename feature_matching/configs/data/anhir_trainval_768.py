from configs.data.base import cfg


TRAIN_BASE_PATH = "path/to/anhir_dataset"
cfg.DATASET.TRAIN_DATA_ROOT = f"{TRAIN_BASE_PATH}/evaluation"
cfg.DATASET.VAL_DATA_ROOT = f"{TRAIN_BASE_PATH}/training"
cfg.DATASET.IMG_PAD_SIZE = 768
