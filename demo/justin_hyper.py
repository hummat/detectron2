import os
from math import isnan

import skopt
import tabulate
import numpy as np

from justin import build_config, train_eval, load_data


def main():
    output_dir = "/home/matthias/Data/Ubuntu/data/justin_training"
    data_root = "/home/matthias/Data/Ubuntu/data/datasets/justin"
    load_data(data_root)

    space = [skopt.space.Real(1e-7, 1e-2, name="learning_rate", prior='log-uniform'),
             skopt.space.Categorical([True, False], name="rotate"),
             skopt.space.Categorical([0., 0.001, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2], name="photometric"),
             skopt.space.Categorical([0., 1e-8, 0.00001, 0.0001, 0.005, 0.01, 0.025, 0.05, 0.1], name="noise"),
             # skopt.space.Categorical(["ADAM"], name="optimizer"),
             skopt.space.Categorical([0., 1e-10, 1e-7, 1e-5, 1e-3, 1e-1], name="weight_decay", prior='log-uniform'),
             # skopt.space.Categorical([0.0, 0.9, 0.95, 0.99], name="momentum"),
             # skopt.space.Categorical([True, False], name="nesterov"),
             skopt.space.Categorical(["WarmupCosineLR", "WarmupMultiStepLR"], name="lr_scheduler"),
             skopt.space.Categorical([0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.], name="warmup_fraction"),
             skopt.space.Categorical([.5, .6, .7, .8, .9, 1.], name="random_crop"),
             skopt.space.Categorical([(480,), (640,), (800,), (480, 800), (640, 672, 704, 736, 768, 800)], name="scales"),
             skopt.space.Categorical([True, False], name="clip_gradients"),
             skopt.space.Categorical([True, False], name="reduce_lr"),
             skopt.space.Categorical([True, False], name="change_num_classes"),
             skopt.space.Categorical([#"COCO-Detection/retinanet_R_50_FPN_1x.yaml",
                                      "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
                                      #"COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",
                                      #"COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
                                      ], name="model")]

    @skopt.utils.use_named_args(dimensions=space)
    def objective(**params):
        cfg = build_config(params["model"], output_dir)

        cfg.SOLVER.BASE_LR = params["learning_rate"]
        cfg.ROTATE = params["rotate"]
        cfg.PHOTOMETRIC = params["photometric"]
        cfg.NOISE = params["noise"]
        # cfg.OPTIMIZER = params["optimizer"]
        cfg.SOLVER.WEIGHT_DECAY = params["weight_decay"]
        # cfg.SOLVER.NESTEROV = params["nesterov"]
        # if not (params["nesterov"] and params["momentum"] == 0.):
        #     cfg.SOLVER.MOMENTUM = params["momentum"]
        # cfg.SOLVER.LR_SCHEDULER_NAME = params["lr_scheduler"]
        cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
        if params["warmup_fraction"] != 0.:
            cfg.SOLVER.WARMUP_ITERS = int(round(cfg.SOLVER.MAX_ITER * params["warmup_fraction"]))
            cfg.SOLVER.WARMUP_FACTOR = 1. / float(cfg.SOLVER.WARMUP_ITERS)
        else:
            cfg.SOLVER.WARMUP_ITERS = 0
            cfg.SOLVER.WARMUP_FACTOR = 1.
        cfg.INPUT.MIN_SIZE_TRAIN = (800,)
        if params["clip_gradients"]:
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
            cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
            cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = .001
            cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.
        else:
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED = False
        if params["reduce_lr"]:
            cfg.SOLVER.STEPS = list(cfg.NUM_BATCHES * np.arange(2, cfg.EPOCHS, 2))
            cfg.SOLVER.GAMMA = .1
        else:
            cfg.SOLVER.STEPS = (cfg.SOLVER.MAX_ITER,)
            cfg.SOLVER.GAMMA = 1.
        if params["random_crop"] != 1.:
            cfg.INPUT.CROP.ENABLED = True
        else:
            cfg.INPUT.CROP.ENABLED = False
        cfg.INPUT.CROP.SIZE = [params["random_crop"], params["random_crop"]]
        if params["change_num_classes"]:
            cfg.MODEL.RETINANET.NUM_CLASSES = 1
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        print(params)
        try:
            result = train_eval(cfg)
        except:
            return 100.
        mAP = list(result.items())[0][1]["AP"]
        print("mAP:", mAP)
        return 100. if isnan(mAP) else 100. - mAP

    res = skopt.dummy_minimize(func=objective, dimensions=space, n_calls=10, verbose=True)
    res_sorted = np.concatenate([np.expand_dims(100. - res.func_vals, axis=1), res.x_iters], axis=1)
    print()
    print(tabulate.tabulate(res_sorted[res_sorted[:, 0].argsort()[::-1]],
                            headers=["mAP", "Learning rate", "Rotate", "Color change (+/-)", "Noise",
                                     "Weight decay", "LR schedule", "Warmup fraction",
                                     "Crop fraction", "Scales", "Clip gradiengs", "Reduce LR", "Change # classes",
                                     "Model"]))

    try:
        skopt.dump(res, os.path.join(output_dir, "skopt_results.pkl"))
    except:
        print("Trying to store the result without the objective.")
        skopt.dump(res, os.path.join(output_dir, "skopt_results.pkl"), store_objective=False)
    finally:
        print("Deleting the objective.")
        del res.specs['args']['func']
        skopt.dump(res, os.path.join(output_dir, "skopt_results.pkl"))


if __name__ == "__main__":
    main()
