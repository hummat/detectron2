import skopt
from PIL import Image
import numpy as np


def get_space():
    space = [skopt.space.Real(1e-6, 1e-3, name="learning_rate", prior='log-uniform'),
             skopt.space.Categorical([2, 4, 8], name="batch_size"),
             # skopt.space.Categorical([.1, .2, .5, 1.], name="epochs"),
             # skopt.space.Categorical([True, False], name="rotate"),
             # skopt.space.Categorical([0., .05, .1, .2, .5, .7, 1.], name="photometric"),
             # skopt.space.Categorical([0., .0001, .0005, .001, .005, .01, .05, .1], name="noise"),
             # skopt.space.Categorical([0., .1, .2, .3, .4, .5], name="cam_noise"),
             # skopt.space.Categorical([0., .1, .2, .3, .4, .5], name="motion_blur"),
             # skopt.space.Categorical([0., .1, .3, .5, 1.], name="cutout"),
             # skopt.space.Categorical([10, 50, 100, 200], name="cutout_sizes"),
             # skopt.space.Categorical([0., .1, .5, 1.], name="sharpen"),
             # skopt.space.Categorical([0., .1, .5, 1.], name="clahe"),
             # skopt.space.Categorical([0., .1, .5, 1.], name="channel_dropout"),
             # skopt.space.Categorical([0., .1, .5, 1.], name="grayscale"),
             # skopt.space.Categorical([0., .1, .5, 1.], name="invert"),
             # skopt.space.Categorical([0., .01, .1, .3, .5, 1.], name="hist"),
             # skopt.space.Real(0., 1., name="photometric"),
             # skopt.space.Real(0., .1, name="noise"),
             # skopt.space.Real(0., .5, name="cam_noise"),
             # skopt.space.Real(0., .5, name="motion_blur"),
             # skopt.space.Real(0., 1., name="gaussian_blur"),
             # skopt.space.Real(0., 1., name="cutout"),
             # skopt.space.Categorical([10, 50, 100, 200], name="cutout_sizes"),
             # skopt.space.Real(0., 1., name="sharpen")]
             # skopt.space.Real(0., 1., name="clahe"),
             # skopt.space.Real(0., 1., name="channel_dropout"),
             # skopt.space.Real(0., 1., name="grayscale"),
             # skopt.space.Real(0., 1., name="invert"),
             # skopt.space.Real(0., 3., name="hist_fda"),
             # skopt.space.Categorical([True, False], name="vignette"),
             # skopt.space.Categorical([True, False], name="chromatic")]
             # skopt.space.Categorical([True, False], name="denoise"),
             # skopt.space.Categorical([Image.BILINEAR, Image.NEAREST, Image.LANCZOS, Image.BICUBIC, Image.LINEAR,
             #                          Image.CUBIC], name="interp")]
             # skopt.space.Categorical(["SGD", "ADAM"], name="optimizer"),
             skopt.space.Real(1e-15, 1e-2, name="weight_decay", prior='log-uniform'),
             # skopt.space.Categorical([0.0, 0.9, 0.95, 0.99], name="momentum"),
             # skopt.space.Categorical([True, False], name="nesterov"),
             skopt.space.Categorical(["WarmupCosineLR", "WarmupMultiStepLR"], name="lr_scheduler"),
             skopt.space.Categorical([0., .1, .2, .3], name="warmup_fraction"),
             # skopt.space.Categorical([.5, .6, .7, .8, .9, 1.], name="random_crop"),
             # skopt.space.Categorical([(480,), (640,), (800,), (480, 800), (640, 672, 704, 736, 768, 800)], name="scales"),
             skopt.space.Categorical([True, True], name="clip_gradients"),
             skopt.space.Categorical([.0001, .0005, .001, .005, .01, .1, .5, 1.], name="clip_value"),
             skopt.space.Categorical(["norm", "value"], name="clip_type"),
             skopt.space.Categorical([1., 2., np.inf], name="clip_norm"),
             skopt.space.Categorical([0., .1, .9, .99], name="reduce_lr")]
             # skopt.space.Categorical([True, False], name="change_num_classes"),
             #skopt.space.Categorical([#"COCO-Detection/retinanet_R_50_FPN_1x.yaml",
                                      #"COCO-Detection/retinanet_R_50_FPN_3x.yaml"
                                      #"COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",
                                      #"COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
                                      #], name="model")]
    return space


@skopt.utils.use_named_args(dimensions=get_space())
def get_param_names(**params):
    return list(params.keys())


def set_cfg_values(cfg, values):
    if "optimizer" in values:
        cfg.OPTIMIZER = values["optimizer"]
    if "momentum" in values:
        cfg.SOLVER.MOMENTUM = values["momentum"]
    if "learning_rate" in values:
        cfg.SOLVER.BASE_LR = values["learning_rate"]
    if "lr_scheduler" in values:
        cfg.SOLVER.LR_SCHEDULER_NAME = values["lr_scheduler"]
    if "warmup_fraction" in values:
        if values["warmup_fraction"] != 0.:
            cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.MAX_ITER * values["warmup_fraction"])
            cfg.SOLVER.WARMUP_FACTOR = 1. / float(cfg.SOLVER.WARMUP_ITERS)
        else:
            cfg.SOLVER.WARMUP_ITERS = 0
            cfg.SOLVER.WARMUP_FACTOR = 1.
    if "reduce_lr" in values:
        if values["reduce_lr"] != 0.:
            if values["reduce_lr"] == 0.1:
                cfg.SOLVER.STEPS = [int(fraction * cfg.SOLVER.MAX_ITER) for fraction in [0.75]]
            else:
                start_step = 1 - values["reduce_lr"]
                cfg.SOLVER.STEPS = [int(fraction * cfg.SOLVER.MAX_ITER) for fraction in np.arange(start_step, 1, start_step)]
            cfg.SOLVER.GAMMA = values["reduce_lr"]
        else:
            cfg.SOLVER.STEPS = (1,)
            cfg.SOLVER.GAMMA = 1.

    if "clip_gradients" in values:
        if values["clip_gradients"]:
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
            cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = values["clip_type"]
            cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = values["clip_value"]
            cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = values["norm_type"]
        else:
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED = False

    if "weight_decay" in values:
        cfg.SOLVER.WEIGHT_DECAY = values["weight_decay"]
    if "scales" in values:
        cfg.INPUT.MIN_SIZE_TRAIN = values["scales"]
        cfg.INPUT.MIN_SIZE_TEST = cfg.INPUT.MIN_SIZE_TRAIN[-1]
    if "random_crop" in values:
        cfg.INPUT.CROP.ENABLED = True if values["random_crop"] != 1. else False
        cfg.INPUT.CROP.SIZE = [values["random_crop"], values["random_crop"]]
    if "rotate" in values:
        cfg.ROTATE = values["rotate"] if values["batch_size"] <= 4 else False

    if "random_types" in values:
        cfg.RANDOM_TYPES = values["random_types"]
    if "photometric" in values:
        cfg.PHOTOMETRIC = [0., values["photometric"]] if values["photometric"] else []
        if "photometric_types" in values:
            cfg.PHOTOMETRIC_TYPES = values["photometric_types"]
    if "noise" in values:
        cfg.NOISE = [0., values["noise"]] if values["noise"] else []
    if "cam_noise" in values:
        cfg.CAM_NOISE = [0., values["cam_noise"]] if values["cam_noise"] else []
        cfg.CAM_NOISE_SHIFT = (0.01, 0.05)
    if "motion_blur" in values:
        cfg.MOTION_BLUR = values["motion_blur"]
    if "gaussian_blur" in values:
        cfg.GAUSSIAN_BLUR = values["gaussian_blur"]
    if "cutout" in values:
        cfg.CUTOUT = values["cutout"]
        if "cutout_sizes" in values:
            cfg.CUTOUT_SIZES = [values["cutout_sizes"],
                                values["cutout_sizes"],
                                None,
                                None] if cfg.CUTOUT else []
    if "sharpen" in values:
        cfg.SHARPEN = values["sharpen"]
        cfg.SHARPEN_RANGE = (0., values["sharpen"])
    if "clahe" in values:
        cfg.CLAHE = values["clahe"]
    if "channel_dropout" in values:
        cfg.CHANNEL_DROPOUT = values["channel_dropout"]
    if "grayscale" in values:
        cfg.GRAYSCALE = values["grayscale"]
    if "vignette" in values:
        cfg.VIGNETTE = (0., 0.8) if values["vignette"] else 0.
    if "chromatic" in values:
        cfg.CHROMATIC_ABERRATION = (0., .5) if values["chromatic"] else 0.
    if "denoise" in values:
        cfg.DENOISE = values["denoise"]
    if "interp" in values:
        cfg.INTERP = values["interp"]
    if "hist_fda" in values:
        if 1. < values["hist_fda"] <= 2.:
            cfg.HISTOGRAM = [0., values["hist_fda"] - 1.]
            cfg.FDA = 0.
        elif 2. < values["hist_fda"] <= 3.:
            cfg.FDA = values["hist_fda"] - 2.
            cfg.HISTOGRAM = []
        else:
            cfg.HISTOGRAM = []
            cfg.FDA = []
    if "invert" in values:
        cfg.INVERT = values["invert"] if values["invert"] else 0.
