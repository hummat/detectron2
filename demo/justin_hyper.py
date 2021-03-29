import numpy as np
import tabulate

import argparse
import os
import random
import skopt
import time
from justin import build_config, load_datasets, set_all_seeds, train_eval
from typing import Tuple, Union, Any
from utils import get_param_names, get_space, parse_data, set_cfg_values


def main(seed=None):
    if seed is not None:
        set_all_seeds(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--data",
                        type=str,
                        nargs='+',
                        help="List of datasets used for training.")
    parser.add_argument("--path_prefix",
                        default="/home/matthias/Data/Ubuntu/data",
                        type=str)
    parser.add_argument("--train_dir", default="datasets/case", type=str)
    parser.add_argument("--val_dir", default="datasets/justin", type=str)
    parser.add_argument("--out_dir", default="justin_training", type=str)
    parser.add_argument("--model", default="retinanet", type=str)
    parser.add_argument("--batch_size",
                        default=4,
                        type=int,
                        help="Batch size used during training.")
    parser.add_argument("--epochs",
                        default=2.0,
                        type=float,
                        help="(Fraction of) epochs to train.")
    parser.add_argument("--optimizer",
                        default="random",
                        type=str,
                        help="Optimizer used in hyperparameter optimization")
    parser.add_argument("--acq_func",
                        default="gp_hedge",
                        type=str,
                        help="Acquisition function for GP optimization")
    parser.add_argument("--calls",
                        default=50,
                        type=int,
                        help="Number of hyperparameter search evaluations.")
    args = parser.parse_args()

    train_root = os.path.join(args.path_prefix, args.train_dir)
    val_root = os.path.join(args.path_prefix, args.val_dir)
    dataset_names = load_datasets(train_root, val_root)

    if args.model == "retinanet":
        base_config = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    elif args.model == "faster_rcnn":
        base_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    elif args.model == "mask_rcnn":
        base_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    else:
        base_config = args.model

    output_dir = os.path.join(args.path_prefix, args.out_dir)
    train_datasets = parse_data(args.data, dataset_names)
    space = get_space()

    @skopt.utils.use_named_args(dimensions=space)
    def objective(**params: Any) -> Union[float, Tuple[float, float]]:
        start = time.time()
        if "random_data" in params:
            if params["random_data"] and len(train_datasets) > 1:
                train_ds = random.sample(train_datasets,
                                         random.randint(1, len(train_datasets)))
        else:
            train_ds = train_datasets
        if "batch_size" in params:
            batch_size = int(params["batch_size"])
        else:
            batch_size = args.batch_size
        if "epochs" in params:
            epochs = int(params["epochs"])
        else:
            epochs = args.epochs
        if "model" in params:
            if params["model"] == "retinanet":
                base_config = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
            elif params["model"] == "faster_rcnn":
                base_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
            elif params["model"] == "mask_rcnn":
                base_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            else:
                base_config = params["model"]
        cfg = build_config(train_ds, base_config, output_dir, batch_size,
                           epochs)
        set_cfg_values(cfg, params)
        print(
            tabulate.tabulate(np.expand_dims(list(params.values()), axis=0),
                              headers=list(params.keys())))

        if args.optimizer == "gp" and args.acq_func in ["EIps", "PIps"]:
            return_time = True
        else:
            return_time = False

        try:
            ap, it = train_eval(cfg)
            if np.any(np.isnan(ap)) or np.nanmax(ap) < 1.0:
                if return_time:
                    return 100.0, start - time.time()
                else:
                    return 100.0
            result = np.vstack([ap, it]).T
            result = result[result[:, 0].argmax()]
        except:
            if return_time:
                return 100.0, start - time.time()
            else:
                return 100.0
        ap = result[0]

        print("AP:", ap, "time:", start - time.time())
        if return_time:
            return 100.0, start - time.time()
        else:
            return 100.0

    if args.optimizer == "gp":
        res = skopt.gp_minimize(func=objective,
                                dimensions=space,
                                n_calls=args.calls,
                                acq_func=args.acq_func,
                                random_state=seed,
                                verbose=True)
    elif args.optimizer == "random":
        res = skopt.dummy_minimize(func=objective,
                                   dimensions=space,
                                   n_calls=args.calls,
                                   random_state=seed,
                                   verbose=True)

    res_sorted = np.concatenate(
        [np.expand_dims(100. - res.func_vals, axis=1), res.x_iters], axis=1)
    table_values = res_sorted[res_sorted[:, 0].argsort()[::-1]]
    table_headers = ["AP"] + get_param_names(space)
    print()
    print("SEED:", seed)
    print(tabulate.tabulate(table_values, headers=table_headers))

    if args.data in ['best', 'all']:
        results_filename = f"hyperopt_{args.model}_{args.data}_{seed}.txt"
    else:
        results_filename = f"hyperopt_{args.model}_{train_datasets[0] if len(train_datasets) == 1 else '_'.join(train_datasets)}_{seed}.txt"
    with open(os.path.join(output_dir, results_filename), 'w') as f:
        f.write(tabulate.tabulate(table_values, headers=table_headers))

    try:
        skopt.dump(res, results_filename.replace('txt', 'pkl'))
    except:
        print("Trying to store the result without the objective.")
        skopt.dump(res,
                   results_filename.replace('txt', 'pkl'),
                   store_objective=False)
    finally:
        print("Deleting the objective.")
        del res.specs['args']['func']
        skopt.dump(res, results_filename.replace('txt', 'pkl'))


if __name__ == "__main__":
    main(seed=np.random.randint(2**31 - 1))
