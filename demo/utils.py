import numpy as np

import skopt


def get_results_dict() -> dict:
    res_dict = {
        "case_color": [
            40.20318213, 35.35496166, 31.02521989, 29.54828771, 27.14293436,
            27.09203161, 26.40146892, 23.90050957, 23.23194967, 22.82641386
        ],  # checked
        "case_still_haven_v2": [
            54.18586341, 47.75698969, 43.20926884, 36.57481208, 29.19424202
        ],
        "case_still_haven": [
            50.44628744, 48.09125223, 47.50439963, 42.09104433, 37.28185969
        ],
        "case_still_v2": [
            56.66434772, 51.39635758, 45.88279167, 38.80435224, 33.65569984
        ],
        "case_front3d_v2": [
            59.0415041, 56.01378549, 54.5582662, 51.2175725, 45.57333254
        ],
        "case_front3d": [
            63.6688337, 53.53222694, 48.59676356, 48.50549171, 45.80506092
        ],
        "case_better_color": [
            52.97515493, 48.77915566, 47.3969108, 46.21510695, 37.15266009
        ],
        "case_suncg_still": [
            53.24553563, 49.94363528, 47.32345091, 40.10024404, 34.41072612
        ],
        "case_better_still": [
            51.00154199, 43.314532, 42.72145512, 42.58400301, 35.25105671
        ],
        "case_still": [
            51.14193958, 48.10064759, 44.07146235, 42.59578138, 41.95914484
        ],
        "case_room": [
            49.3709824, 45.66797636, 42.06408144, 38.47201768, 38.318354
        ],
        "case_smile": [
            38.76968107, 35.22062779, 31.3497792, 28.46137785, 22.17331094
        ],
        "case_image": [
            47.08983178, 42.3895778, 39.37276908, 39.00633617, 35.12822138
        ],
        "case_image_a": [
            45.60643148, 31.29878447, 31.01271054, 28.92226343, 14.79099625
        ],
        "case_suncg": [
            46.02135091, 38.33751242, 35.62200875, 33.529601, 33.34456426
        ],
        "case_junk_alt": [
            43.56432474, 41.1069868, 40.1089639, 38.97250156, 36.57701396
        ],
        "case_junk": [
            45.42713099, 43.03683623, 35.58363348, 28.97702246, 18.69816359
        ],
        "case_hdr_only": [
            46.15308288, 44.93498767, 40.18497103, 38.49819022, 29.2418928
        ],
        "case_haven_hdr": [
            41.75135572, 34.1886493, 33.31176793, 29.1229799, 28.53087389
        ],
        "case_no_alpha": [
            48.87262553, 47.59580096, 44.83208555, 40.46059594, 40.40870896
        ],
        "case_new_cam_tless_new": [
            48.16598277, 46.6853439, 45.43259078, 42.09467064, 38.85442359
        ],
        "case_new_cam_tless_a": [
            54.17362365, 48.28175825, 46.88282047, 42.33050659, 41.49927066
        ],
        "case_new_cam_tless": [
            45.47795505, 44.66041727, 40.51798259, 36.96881034, 29.26575374
        ],
        "case_new_cam": [
            44.75878892, 42.88613014, 39.33238446, 32.79025745, 32.22301464
        ],
        "case_new_cam_a": [
            40.63473765, 33.00102775, 31.36526296, 30.9117473, 30.7852782
        ],
        "case_new_cam_b": [
            41.39483998, 39.97691288, 38.96729607, 37.88191947, 34.72803834
        ],
        # checked
        "case_best_off": [
            38.27442956, 36.98222766, 33.18468101, 32.26386027, 30.13600164
        ],
        # checked
        "case_green": [
            42.07711803, 33.84280772, 33.08787521, 31.47131132, 30.18678932
        ],
        "case_white": [21.0838, 17.2644, 12.5009, 11.3386, 6.06284],
        # checked
        "case_flat": [
            37.91878455, 30.64923742, 28.30067853, 27.75824945, 27.34716009
        ],
        # checked
        "case_smooth": [
            30.82600167, 30.42064908, 27.06868854, 25.34457596, 20.87147743
        ],
        "case_refined": [22.3332, 21.3385, 21.0649, 18.4229, 14.5228],
        # checked
        "case_cam_k": [
            29.96887896, 26.99492782, 22.50186606, 16.9611616, 15.7358362
        ],
        "case_all_white": [12.8596, 12.1132, 10.5135, 10.4775, 8.79655],
        "case_50_samples": [29.6049, 23.6589, 23.5817, 21.2605, 20.5096],
        "case_512_samples": [26.1091, 23.8615, 22.8284, 21.5225, 9.46198],
        # checked
        "case_2048_samples": [
            25.51789534, 23.77423735, 22.73706366, 22.58643221, 20.10152691
        ],
        # checked
        "case_color_texture": [
            42.27489632, 33.6542957, 32.44540635, 30.33389404, 20.45308856
        ],
        # checked
        "case_texture": [
            30.93216935, 30.5514081, 28.04813986, 24.42527646, 19.33343886
        ],
        # checked
        "case_more_bounces": [
            33.83921761, 29.46399028, 27.03319279, 26.84496801, 25.03445091
        ],
        # checked
        "case_area_light": [
            28.64929296, 26.14490815, 21.8578116, 19.71479663, 17.28886026
        ],
        # checked
        "case_cam_uniform3d_inplane": [
            31.96673795, 31.26197291, 30.49044743, 24.15643791, 23.57493626
        ],
        # checked
        "case_cam_uniform3d": [
            26.97584852, 26.62969497, 26.14628782, 21.09606741, 20.16900372
        ],
        # checked
        "case_cam_poi": [
            10.86430437, 7.79265517, 7.42004987, 6.64692292, 6.02255621
        ],
        # checked
        "case_white_light": [
            31.87034131, 29.22695543, 26.59384891, 23.18313712, 20.93746476
        ],
        # checked
        "case_no_bounces": [
            32.4185546, 28.64617559, 25.05437995, 23.88907582, 19.67199348
        ],
        # checked
        "case_uniform_clutter": [
            33.40435191, 31.27687028, 26.87701708, 25.97052168, 23.90796012
        ],
        # checked
        "case_clutter": [
            36.49185575, 30.14689522, 25.91438832, 25.32848971, 20.45497139
        ],
        # checked
        "case_high_res": [
            32.10037995, 27.17417729, 17.91426037, 14.40626157, 12.24492306
        ],
        # checked
        "case_point_lights_only": [
            27.49361244, 25.31737831, 21.3253789, 14.1208087, 11.48722047
        ],
        # checked
        "case_less_light": [
            26.20951466, 23.10560782, 21.613864, 20.07971469, 4.9723414
        ],
        # checked
        "case_two_lights": [
            26.07114511, 19.92342411, 11.68455528, 9.60110245, 3.3309312
        ],
        # checked
        "case_even_less_cam_rotation": [
            41.14436033, 37.58032475, 35.9072579, 34.13007927, 30.49871135
        ],
        # checked
        "case_fov": [
            36.76167819, 35.08760321, 31.27037743, 30.37089514, 26.70382605
        ],
        "case_less_cam_rotation": [25.7747, 18.8846, 18.2181, 17.4002,
                                   10.1219],  # checked
        # checked
        "case_train_green": [
            36.01153481, 34.57256129, 31.54693327, 30.22182292, 22.28776532
        ],
        "case_test_green": [27.6192, 21.9087, 21.3022, 20.4311,
                            17.6845],  # checked
        # checked
        "case_no_specular": [
            19.30362239, 18.9428661, 17.37443433, 13.01278383, 11.54669053
        ],
        "case_no_denoiser": [30.0072, 27.3804, 25.6204, 22.7732,
                             19.3679],  # checked
        "case_no_glossy": [35.6989, 29.8466, 29.3413, 25.6552,
                           17.015],  # checked
        # checked
        "case_no_rough": [
            35.69471537, 33.93573952, 30.49634283, 30.43674663, 24.45885994
        ],
        "case_no_point_light": [
            40.83368553, 34.71397511, 32.09706752, 27.477971, 25.45296477
        ]
    }  # checked
    return res_dict


def parse_data(data: list, dataset_names: list) -> list:
    if len(data) == 1:
        data = data[0].split(' ')
    if len(data) == 1:
        data = data[0]
        assert (data in dataset_names or
                data in ['all', 'best']), f"Unknown dataset {data}"
        train_datasets = list()
        if data == 'all':
            train_datasets = dataset_names
        elif data == 'best':
            best_datasets = list()
            for k, v in get_results_dict().items():
                best_datasets.append([np.mean(v), np.max(v)])
            best_mean = np.quantile(np.array(best_datasets)[:, 0], q=.9)
            best_max = np.quantile(np.array(best_datasets)[:, 1], q=.9)
            for k, v in get_results_dict().items():
                if np.mean(v) >= best_mean or np.max(v) >= best_max:
                    train_datasets.append(k)
        else:
            train_datasets.append(data)
    elif isinstance(data, (list, tuple)):
        train_datasets = data
    else:
        raise AttributeError
    return train_datasets


def get_space() -> list:
    space = [
        skopt.space.Real(1e-6, 1e-4, name="learning_rate", prior='log-uniform'),
        skopt.space.Categorical(
            [0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
            name="weight_decay"),
        # skopt.space.Categorical([True, False], name="random_data"),
        skopt.space.Categorical([1, 2, 3, 5], name="epochs"),
        skopt.space.Categorical([2, 4, 8], name="batch_size"),
        skopt.space.Categorical(["coco", "imagenet", "none"], name="weights"),
        # skopt.space.Real(0.0, 2.0, name="photometric"),
        # skopt.space.Categorical([True, False], name="random_types"),
        # skopt.space.Categorical([True, False], name="cam_noise"),
        # skopt.space.Categorical([True, False], name="noise"),
        # skopt.space.Real(0.0, 2.0, name="motion_blur"),
        # skopt.space.Real(0.0, 2.0, name="cutout"),
        # skopt.space.Categorical([10, 50, 100, 200], name="cutout_sizes"),
        # skopt.space.Real(0.0, 2.0, name="sharpen"),
        # skopt.space.Real(0.0, 2.0, name="clahe"),
        # skopt.space.Real(0.0, 2.0, name="channel_dropout"),
        # skopt.space.Real(0.0, 2.0, name="invert"),
        # skopt.space.Real(0.0, 2.0, name="hist"),
        # skopt.space.Categorical([True, False], name="vignette"),
        # skopt.space.Categorical([True, False], name="chromatic"),
        skopt.space.Categorical([0.0, 0.1, 0.2, 0.3], name="warmup_fraction"),
        skopt.space.Categorical([0.0, 0.9, 0.99], name="reduce_lr"),
        skopt.space.Categorical(["retinanet", "mask_rcnn", "faster_rcnn"],
                                name="model")
    ]
    return space


@skopt.utils.use_named_args(dimensions=get_space())
def get_param_names(**params) -> list:
    return list(params.keys())


def set_cfg_values(cfg, values):
    # Typecast to standard Python types if not already
    for k, v in values.items():
        try:
            values[k] = v.item()
        except AttributeError:
            pass

    if "optimizer" in values:
        cfg.OPTIMIZER = values["optimizer"]
    if "momentum" in values:
        cfg.SOLVER.MOMENTUM = values["momentum"]
    if "learning_rate" in values:
        cfg.SOLVER.BASE_LR = values["learning_rate"]
    if "lr_scheduler" in values:
        cfg.SOLVER.LR_SCHEDULER_NAME = values["lr_scheduler"]
    if "warmup_fraction" in values:
        if values["warmup_fraction"] != 0.0:
            cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.MAX_ITER *
                                          values["warmup_fraction"])
            cfg.SOLVER.WARMUP_FACTOR = 1. / float(cfg.SOLVER.WARMUP_ITERS)
        else:
            cfg.SOLVER.WARMUP_ITERS = 0
            cfg.SOLVER.WARMUP_FACTOR = 1.0
    if "reduce_lr" in values:
        if values["reduce_lr"] != 0.0:
            if values["reduce_lr"] == 0.1:
                cfg.SOLVER.STEPS = [
                    int(fraction * cfg.SOLVER.MAX_ITER) for fraction in [0.75]
                ]
            else:
                start_step = 1 - values["reduce_lr"]
                cfg.SOLVER.STEPS = [
                    int(fraction * cfg.SOLVER.MAX_ITER)
                    for fraction in np.arange(start_step, 1, start_step)
                ]
            cfg.SOLVER.GAMMA = values["reduce_lr"]
        else:
            cfg.SOLVER.STEPS = (1,)
            cfg.SOLVER.GAMMA = 1.0

    if "clip_gradients" in values:
        if values["clip_gradients"]:
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
            if "clip_type" in values:
                cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = values["clip_type"]
            if "clip_value" in values:
                cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = values["clip_value"]
            if "norm_type" in values:
                cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = values["norm_type"]
        else:
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED = False

    if "weight_decay" in values:
        cfg.SOLVER.WEIGHT_DECAY = values["weight_decay"]
    if "scales" in values:
        cfg.INPUT.MIN_SIZE_TRAIN = values["scales"]
        cfg.INPUT.MIN_SIZE_TEST = cfg.INPUT.MIN_SIZE_TRAIN[-1]
    if "random_crop" in values:
        cfg.INPUT.CROP.ENABLED = True if values["random_crop"] != 1.0 else False
        cfg.INPUT.CROP.SIZE = [values["random_crop"], values["random_crop"]]
    if "rotate" in values:
        cfg.ROTATE = values["rotate"] if values["batch_size"] <= 4 else False

    if "random_types" in values:
        cfg.RANDOM_TYPES = values["random_types"]
    if "photometric" in values:
        if values["photometric"] <= 1.0:
            max_val = values["photometric"]
            cfg.PHOTOMETRIC = [max_val, max_val, max_val, max_val
                              ] if values["photometric"] else []
            if "photometric_types" in values:
                cfg.PHOTOMETRIC_TYPES = values["photometric_types"]
    if "noise" in values:
        cfg.NOISE = [0.0, 0.1] if values["noise"] else []
    if "cam_noise" in values:
        cfg.CAM_NOISE = [1.0, 1.0] if values["cam_noise"] else []
        cfg.CAM_NOISE_SHIFT = (0.05, 0.1)
    if "motion_blur" in values:
        if values["motion_blur"] <= 1.0:
            cfg.MOTION_BLUR = values["motion_blur"]
    if "gaussian_blur" in values:
        cfg.GAUSSIAN_BLUR = values["gaussian_blur"]
    if "cutout" in values:
        if values["cutout"] <= 1.0:
            cfg.CUTOUT = values["cutout"]
            if "cutout_sizes" in values:
                cfg.CUTOUT_SIZES = [
                    values["cutout_sizes"], values["cutout_sizes"], None, None
                ] if cfg.CUTOUT else []
    if "sharpen" in values:
        if values["sharpen"] <= 1.0:
            cfg.SHARPEN = values["sharpen"]
            cfg.SHARPEN_RANGE = (0., values["sharpen"])
    if "clahe" in values:
        if values["clahe"] <= 1.0:
            cfg.CLAHE = values["clahe"]
    if "channel_dropout" in values:
        if values["channel_dropout"] <= 1.0:
            cfg.CHANNEL_DROPOUT = values["channel_dropout"]
    if "grayscale" in values:
        if values["grayscale"] <= 1.0:
            cfg.GRAYSCALE = values["grayscale"]
    if "vignette" in values:
        cfg.VIGNETTE = (0.0, 0.8) if values["vignette"] else 0.0
    if "chromatic" in values:
        cfg.CHROMATIC_ABERRATION = (0.0, 0.5) if values["chromatic"] else 0.0
    if "denoise" in values:
        cfg.DENOISE = values["denoise"]
    if "interp" in values:
        cfg.INTERP = values["interp"]
    if "hist_fda" in values:
        if 1.0 < values["hist_fda"] <= 2.0:
            cfg.HISTOGRAM = [0.0, values["hist_fda"] - 1.0]
            cfg.FDA = 0.0
        elif 2.0 < values["hist_fda"] <= 3.0:
            cfg.FDA = values["hist_fda"] - 2.0
            cfg.HISTOGRAM = []
        else:
            cfg.HISTOGRAM = []
            cfg.FDA = 0.0
    if "hist" in values:
        if values["hist"] <= 1.0:
            cfg.HISTOGRAM = [0.0, values["hist"]]
    if "invert" in values:
        if values["invert"] <= 1.0:
            cfg.INVERT = values["invert"]
