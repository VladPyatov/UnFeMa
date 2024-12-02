from src.config.default import _CN as cfg

cfg.LOFTR.MATCH_COARSE.MATCH_TYPE = "dual_softmax"

cfg.LOFTR.MATCH_COARSE.SPARSE_SPVS = False # default True
cfg.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.5 # default 0.2, quadtree 0.3

cfg.LOFTR.RESNETFPN.INITIAL_DIM = 128
cfg.LOFTR.RESNETFPN.BLOCK_DIMS = [128, 196, 256]
cfg.LOFTR.COARSE.D_MODEL = 256
cfg.LOFTR.COARSE.BLOCK_TYPE = "quadtree"
cfg.LOFTR.COARSE.ATTN_TYPE = "B"
cfg.LOFTR.COARSE.TOPKS=[16, 8, 8]

cfg.LOFTR.FINE.D_MODEL = 128

cfg.LOFTR.LOSS.COARSE_TYPE = 'ncc_local_multi'
cfg.LOFTR.LOSS.FINE_TYPE = None

cfg.TRAINER.CANONICAL_LR = 1e-4
cfg.TRAINER.CANONICAL_BS = 64
cfg.TRAINER.WARMUP_STEP = 35  # 3 epochs if bs=2+2=4, then 481/4=120 steps per epoch, then 3 epoches equals to 360
cfg.TRAINER.WARMUP_RATIO = 0.1#0.1
cfg.TRAINER.WARMUP_TYPE = 'constant'
cfg.TRAINER.SCHEDULER = 'MultiStepLR'
cfg.TRAINER.ELR_GAMMA = 0.995#0.999992
cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
cfg.TRAINER.SEED = 42
cfg.TRAINER.N_VAL_PAIRS_TO_PLOT = 23     # number of val/test paris for plotting
cfg.TRAINER.PLOT_MODE = 'anhir'

cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1