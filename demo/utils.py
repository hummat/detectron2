import skopt
from PIL import Image
import numpy as np


def get_space():
    space = [# skopt.space.Real(1e-5, 1e-3, name="learning_rate", prior='log-uniform'),
             # skopt.space.Categorical([2, 4, 8], name="batch_size"),
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
             skopt.space.Real(0., 1., name="photometric"),
             # skopt.space.Real(0., .1, name="noise"),
             # skopt.space.Real(0., .5, name="cam_noise"),
             # skopt.space.Real(0., .5, name="motion_blur"),
             skopt.space.Real(0., 1., name="gaussian_blur"),
             skopt.space.Real(0., 1., name="cutout"),
             skopt.space.Categorical([10, 50, 100, 200], name="cutout_sizes"),
             skopt.space.Real(0., 1., name="sharpen")]
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
             # skopt.space.Categorical(["ADAM"], name="optimizer"),
             # skopt.space.Categorical([0., 1e-10, 1e-7, 1e-5, 1e-3], name="weight_decay"),
             # skopt.space.Categorical([0.0, 0.9, 0.95, 0.99], name="momentum"),
             # skopt.space.Categorical([True, False], name="nesterov"),
             # skopt.space.Categorical(["WarmupCosineLR", "WarmupMultiStepLR"], name="lr_scheduler"),
             # skopt.space.Categorical([0., .1, .2, .3, .4, .5], name="warmup_fraction"),
             # skopt.space.Categorical([.5, .6, .7, .8, .9, 1.], name="random_crop"),
             # skopt.space.Categorical([(480,), (640,), (800,), (480, 800), (640, 672, 704, 736, 768, 800)], name="scales"),
             # skopt.space.Categorical([True, False], name="clip_gradients"),
             # skopt.space.Categorical([0., .1, .9, .99], name="reduce_lr"),
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


def set_cfg_values(cfg, training_values=None, augmentation_values=None):
    if training_values is not None:
        if "learning_rate" in training_values:
            cfg.SOLVER.BASE_LR = training_values["learning_rate"]
        if "lr_scheduler" in training_values:
            cfg.SOLVER.LR_SCHEDULER_NAME = training_values["lr_scheduler"]
        if "warmup_fraction" in training_values:
            if training_values["warmup_fraction"] != 0.:
                cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.MAX_ITER * training_values["warmup_fraction"])
                cfg.SOLVER.WARMUP_FACTOR = 1. / float(cfg.SOLVER.WARMUP_ITERS)
            else:
                cfg.SOLVER.WARMUP_ITERS = 0
                cfg.SOLVER.WARMUP_FACTOR = 1.
        if "reduce_lr" in training_values:
            if training_values["reduce_lr"] != 0.:
                if training_values["reduce_lr"] == 0.1:
                    cfg.SOLVER.STEPS = [fraction * cfg.SOLVER.MAX_ITER for fraction in [0.75]]
                else:
                    start_step = 1 - training_values["reduce_lr"]
                    cfg.SOLVER.STEPS = [fraction * cfg.SOLVER.MAX_ITER for fraction in np.arange(start_step, 1, start_step)]
                cfg.SOLVER.GAMMA = training_values["reduce_lr"]
            else:
                cfg.SOLVER.STEPS = (1,)
                cfg.SOLVER.GAMMA = 1.

        if "clip_gradients" in training_values:
            if training_values["clip_gradients"]:
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
                cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
                cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = .001
                cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.
            else:
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED = False

        # cfg.SOLVER.IMS_PER_BATCH = training_values["batch_size"]
        if "weight_decay" in training_values:
            cfg.SOLVER.WEIGHT_DECAY = training_values["weight_decay"]
        if "scales" in training_values:
            cfg.INPUT.MIN_SIZE_TRAIN = training_values["scales"]
            cfg.INPUT.MIN_SIZE_TEST = cfg.INPUT.MIN_SIZE_TRAIN[-1]
        if "random_crop" in training_values:
            cfg.INPUT.CROP.ENABLED = True if training_values["random_crop"] != 1. else False
            cfg.INPUT.CROP.SIZE = [training_values["random_crop"], training_values["random_crop"]]
        if "rotate" in training_values:
            cfg.ROTATE = training_values["rotate"] if training_values["batch_size"] <= 4 else False

    if augmentation_values is not None:
        if "random_types" in augmentation_values:
            cfg.RANDOM_TYPES = augmentation_values["random_types"]
        if "photometric" in augmentation_values:
            cfg.PHOTOMETRIC = [0., augmentation_values["photometric"]] if augmentation_values["photometric"] else []
            if "photometric_types" in augmentation_values:
                cfg.PHOTOMETRIC_TYPES = augmentation_values["photometric_types"]
        if "noise" in augmentation_values:
            cfg.NOISE = [0., augmentation_values["noise"]] if augmentation_values["noise"] else []
        if "cam_noise" in augmentation_values:
            cfg.CAM_NOISE = [0., augmentation_values["cam_noise"]] if augmentation_values["cam_noise"] else []
            cfg.CAM_NOISE_SHIFT = (0.01, 0.05)
        if "motion_blur" in augmentation_values:
            cfg.MOTION_BLUR = augmentation_values["motion_blur"]
        if "gaussian_blur" in augmentation_values:
            cfg.GAUSSIAN_BLUR = augmentation_values["gaussian_blur"]
        if "cutout" in augmentation_values:
            cfg.CUTOUT = augmentation_values["cutout"]
            if "cutout_sizes" in augmentation_values:
                cfg.CUTOUT_SIZES = [augmentation_values["cutout_sizes"],
                                    augmentation_values["cutout_sizes"],
                                    None,
                                    None] if cfg.CUTOUT else []
        if "sharpen" in augmentation_values:
            cfg.SHARPEN = augmentation_values["sharpen"]
            cfg.SHARPEN_RANGE = (0., augmentation_values["sharpen"])
        if "clahe" in augmentation_values:
            cfg.CLAHE = augmentation_values["clahe"]
        if "channel_dropout" in augmentation_values:
            cfg.CHANNEL_DROPOUT = augmentation_values["channel_dropout"]
        if "grayscale" in augmentation_values:
            cfg.GRAYSCALE = augmentation_values["grayscale"]
        if "vignette" in augmentation_values:
            cfg.VIGNETTE = (0., 0.8) if augmentation_values["vignette"] else 0.
        if "chromatic" in augmentation_values:
            cfg.CHROMATIC_ABERRATION = (0., .5) if augmentation_values["chromatic"] else 0.
        if "denoise" in augmentation_values:
            cfg.DENOISE = augmentation_values["denoise"]
        if "interp" in augmentation_values:
            cfg.INTERP = augmentation_values["interp"]
        if "hist_fda" in augmentation_values:
            if 1. < augmentation_values["hist_fda"] <= 2.:
                cfg.HISTOGRAM = [0., augmentation_values["hist_fda"] - 1.]
                cfg.FDA = 0.
            elif 2. < augmentation_values["hist_fda"] <= 3.:
                cfg.FDA = augmentation_values["hist_fda"] - 2.
                cfg.HISTOGRAM = []
            else:
                cfg.HISTOGRAM = []
                cfg.FDA = []
        if "invert" in augmentation_values:
            cfg.INVERT = augmentation_values["invert"] if augmentation_values["invert"] else 0.
