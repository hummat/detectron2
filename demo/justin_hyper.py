import os
import random

import skopt
import tabulate
import numpy as np
import argparse

from utils import get_space, get_param_names, set_cfg_values, parse_data
from justin import set_all_seeds, load_datasets, build_config, train_eval


def main(seed=None):
    if seed is not None:
        set_all_seeds(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, nargs='+', help="List of datasets used for training.")
    parser.add_argument("--path_prefix", default="/home/matthias/Data/Ubuntu/data", type=str)
    parser.add_argument("--train_dir", default="datasets/case", type=str)
    parser.add_argument("--val_dir", default="datasets/justin", type=str)
    parser.add_argument("--out_dir", default="justin_training", type=str)
    parser.add_argument("--model", default="retinanet", type=str)
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size used during training.")
    parser.add_argument("--epochs", default=2.0, type=float, help="(Fraction of) epochs to train.")
    parser.add_argument("--calls", default=50, type=int, help="Number of hyperparameter search evaluations.")
    args = parser.parse_args()

    train_root = os.path.join(args.path_prefix, args.train_dir)
    val_root = os.path.join(args.path_prefix, args.val_dir)
    dataset_names = load_datasets(train_root, val_root)

    if args.model == "retinanet":
        base_config = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    elif args.model == "mask_rcnn":
        base_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    else:
        base_config = args.model
    output_dir = os.path.join(args.path_prefix, args.out_dir)

    train_datasets = parse_data(args.data, dataset_names)
    space = get_space()

    @skopt.utils.use_named_args(dimensions=space)
    def objective(**params):
        if "random_data" in params:
            if params["random_data"] and len(train_datasets) > 1:
                train_ds = random.sample(train_datasets, random.randint(1, len(train_datasets)))
        else:
            train_ds = train_datasets
        if "batch_size" in params:
            batch_size = params["batch_size"]
        else:
            batch_size = args.batch_size
        if "epochs" in params:
            epochs = params["epochs"]
        else:
            epochs = args.epochs
        cfg = build_config(train_ds, base_config, output_dir, batch_size, epochs)
        set_cfg_values(cfg, params)
        print(tabulate.tabulate(np.expand_dims(list(params.values()), axis=0), headers=list(params.keys())))
        try:
            ap, it = train_eval(cfg)
            if np.any(np.isnan(ap)):
                return 100.
            elif ap.max() < 1.:
                return 100.
            result = np.vstack([ap, it]).T
            result = result[result[:, 0].argmax()]
        except:
            return 100.
        ap = result[0]
        print("AP:", ap)
        return 100. if np.isnan(ap) else 100. - ap

    res = skopt.dummy_minimize(func=objective, dimensions=space, n_calls=args.calls, random_state=seed, verbose=True)
    res_sorted = np.concatenate([np.expand_dims(100. - res.func_vals, axis=1), res.x_iters], axis=1)
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
        skopt.dump(res, results_filename.replace('txt', 'pkl'), store_objective=False)
    finally:
        print("Deleting the objective.")
        del res.specs['args']['func']
        skopt.dump(res, results_filename.replace('txt', 'pkl'))


if __name__ == "__main__":
    main(seed=np.random.randint(2 ** 31 - 1))

"""SEED: 1330231260 | data: case_new_cam_tless_new, case_image, case_room
      AP    learning_rate  random_data      epochs    batch_size  weights      photometric  random_types    cam_noise    noise      motion_blur     cutout    cutout_sizes     sharpen      clahe    channel_dropout     invert       hist  vignette    chromatic      warmup_fraction    reduce_lr
--------  ---------------  -------------  --------  ------------  ---------  -------------  --------------  -----------  -------  -------------  ---------  --------------  ----------  ---------  -----------------  ---------  ---------  ----------  -----------  -----------------  -----------
59.6911       3.64182e-05  False                 5            16  none         1.34933      True            True         True         0.35648    0.349892               10  0.899466    1.9614            1.34076     1.17482    0.907061   False       True                       0           0.9
"""

"""SEED: 466711279 | data: case_new_cam_tless_new, case_image, case_room, case_no_alpha, case_still
      AP    learning_rate  random_data      epochs    batch_size  weights      photometric  random_types    cam_noise    noise      motion_blur     cutout    cutout_sizes     sharpen       clahe    channel_dropout     invert        hist  vignette    chromatic      warmup_fraction    reduce_lr
--------  ---------------  -------------  --------  ------------  ---------  -------------  --------------  -----------  -------  -------------  ---------  --------------  ----------  ----------  -----------------  ---------  ----------  ----------  -----------  -----------------  -----------
58.932        7.81261e-05  False                 3             2  none          1.57246     False           False        True        0.692054    1.29552               200  0.411191    1.44028           0.0204539    1.3208     0.38159     False       False                      0.3         0.99
"""

"""
      AP    learning_rate    photometric  random_types    photometric_types                                       cam_noise    motion_blur    cutout    cutout_sizes    sharpen     clahe    channel_dropout    grayscale     invert    hist_fda  vignette    chromatic      weight_decay    warmup_fraction  clip_gradients      reduce_lr
--------  ---------------  -------------  --------------  ----------------------------------------------------  -----------  -------------  --------  --------------  ---------  --------  -----------------  -----------  ---------  ----------  ----------  -----------  --------------  -----------------  ----------------  -----------
52.058        4.28226e-05      1.89366    True            ('brightness',)                                         0.103469       0.622739   0.887532             100   1.26211   0.923281           0.962356    1.18521    1.24233      1.56282   True        False           1.94036e-13                0.2  False                    0
46.8321       6.87235e-05      0.99484    False           ('brightness', 'contrast', 'saturation', 'lighting')    0.0815834     0.447135    0.360628               50  1.24189    0.254711           1.06683      1.46498    1.30233     2.49435    True        True            9.20867e-13                0    True                     0.9
45.6293       0.000290654      1.31423    True            ('contrast',)                                           0.31747        0.440812   0.90653               100  1.15749    1.03711          0.429949       1.77727    0.421703    1.84105    False       False           1.18207e-14                0.2  False                    0.99
45.0714       6.38393e-05      1.36208    True            ('brightness',)                                         0.0544735      0.390955   0.844674              100  0.0418989  0.527211           1.85124      1.06547    1.17996      2.08706   True        True            4.27169e-13                0.1  False                    0.9
"""

"""
      AP    learning_rate    batch_size    weight_decay  lr_scheduler         warmup_fraction  clip_gradients      clip_value  clip_type      norm_type    reduce_lr
--------  ---------------  ------------  --------------  -----------------  -----------------  ----------------  ------------  -----------  -----------  -----------
50.8236       0.000112884             4     2.4338e-11   WarmupMultiStepLR                0.1  True                    1e-05   norm                   2         0.9
49.4218       0.000200714             2     3.37867e-08  WarmupMultiStepLR                0.2  True                    0.0005  norm                   1         0
48.3921       0.000400831             2     8.42407e-08  WarmupCosineLR                   0.3  True                    0.001   value                inf         0
45.5131       8.96462e-05             4     1.19641e-08  WarmupMultiStepLR                0    True                    0.005   norm                   2         0.99
44.5976       0.000351911             4     1.25598e-15  WarmupMultiStepLR                0.3  True                    0.5     value                  1         0.1
"""

"""seed 13
      AP    learning_rate    batch_size  optimizer      weight_decay    momentum  lr_scheduler         warmup_fraction  clip_gradients      reduce_lr
--------  ---------------  ------------  -----------  --------------  ----------  -----------------  -----------------  ----------------  -----------
45.4444       4.925e-05               4  ADAM            3.26758e-05        0.9   WarmupMultiStepLR                0.2  False                    0.99
42.4404       0.00018893              2  ADAM            8.61609e-09        0.95  WarmupMultiStepLR                0.2  True                     0.99
42.2611       1.74606e-05             2  ADAM            1.16105e-05        0     WarmupMultiStepLR                0    True                     0
42.0539       0.000107611             8  ADAM            1.79504e-09        0.99  WarmupMultiStepLR                0.1  False                    0.1
41.9136       5.25361e-05             2  ADAM            1.72767e-07        0.99  WarmupCosineLR                   0.2  False                    0
41.8785       0.000286816             2  SGD             1.19338e-07        0.99  WarmupMultiStepLR                0.1  False                    0.9
41.1564       9.31932e-05             2  ADAM            1.17301e-09        0.9   WarmupMultiStepLR                0.1  True                     0
41.0661       3.18662e-05             2  ADAM            2.34281e-08        0.99  WarmupMultiStepLR                0    False                    0
39.3311       1.58885e-05             2  ADAM            8.01104e-09        0.95  WarmupMultiStepLR                0.2  True                     0.9
39.3258       1.67384e-05             8  ADAM            2.66187e-05        0     WarmupMultiStepLR                0    False                    0
39.0861       0.000122106             2  ADAM            1.66605e-09        0.99  WarmupMultiStepLR                0.2  False                    0.9
38.8237       3.78651e-05             4  ADAM            1.09702e-05        0.9   WarmupCosineLR                   0.1  False                    0.99
38.2017       0.000286083             2  SGD             7.9831e-07         0.95  WarmupCosineLR                   0.2  False                    0.1
37.0004       1.59702e-05             2  ADAM            1.83059e-10        0.95  WarmupMultiStepLR                0.2  True                     0
36.7239       0.000101445             4  ADAM            1.44761e-05        0.99  WarmupMultiStepLR                0.1  True                     0.1
36.4215       0.000124969             4  ADAM            0.000297016        0.9   WarmupMultiStepLR                0    False                    0.99
35.0634       7.81416e-05             4  SGD             2.09212e-07        0.99  WarmupCosineLR                   0    False                    0
32.7209       0.000308701             4  ADAM            8.31087e-07        0.99  WarmupCosineLR                   0.2  False                    0
32.1562       0.000461665             4  SGD             0.000290648        0.9   WarmupMultiStepLR                0    False                    0.9
30.4118       0.000233193             4  ADAM            2.32092e-07        0.9   WarmupMultiStepLR                0    False                    0
30.4005       1.01869e-05             8  ADAM            3.03709e-08        0.99  WarmupMultiStepLR                0.1  True                     0.9
30.0356       5.18214e-05             2  SGD             1.98044e-10        0.9   WarmupMultiStepLR                0    False                    0.99
29.9443       0.000110647             2  ADAM            0.000374754        0.95  WarmupCosineLR                   0.1  False                    0.1
29.4267       1.279e-05               4  ADAM            2.40339e-10        0     WarmupMultiStepLR                0.1  True                     0.9
29.1358       2.00217e-05             4  ADAM            2.41694e-06        0     WarmupCosineLR                   0    True                     0.99
28.9278       5.51032e-05             2  SGD             2.59898e-06        0.9   WarmupCosineLR                   0.2  False                    0
27.2923       1.25456e-05             4  SGD             1.9909e-05         0.99  WarmupMultiStepLR                0    False                    0.9
27.2509       1.22961e-05             8  ADAM            2.38683e-09        0.95  WarmupMultiStepLR                0.1  False                    0.9
27.1691       0.000222722             8  ADAM            5.29012e-07        0.9   WarmupMultiStepLR                0.2  True                     0.1
26.7044       1.38205e-05             8  ADAM            8.39372e-06        0.9   WarmupCosineLR                   0.2  False                    0.99
25.9286       8.44888e-05             8  ADAM            1.39439e-07        0.99  WarmupMultiStepLR                0.2  False                    0
25.198        1.08956e-05             2  SGD             6.4671e-05         0.99  WarmupCosineLR                   0    False                    0
20.5911       4.52048e-05             4  SGD             2.72824e-08        0.95  WarmupCosineLR                   0    False                    0.1
19.6199       9.90983e-05             4  ADAM            0.000642097        0.95  WarmupMultiStepLR                0.2  True                     0.99
18.4168       6.92224e-05             4  SGD             2.05293e-10        0.95  WarmupCosineLR                   0.1  False                    0.99
18.1889       3.69874e-05             4  SGD             0.000109849        0.9   WarmupMultiStepLR                0    False                    0.1
17.7828       0.000314975             2  ADAM            2.40808e-10        0.99  WarmupMultiStepLR                0.2  True                     0.1
12.8886       6.10468e-05             8  SGD             0.000110976        0.9   WarmupMultiStepLR                0.1  False                    0.1
 1.65695      0.000107763             2  SGD             5.1073e-07         0     WarmupMultiStepLR                0    True                     0.1
 0            2.76347e-05             8  SGD             1.5429e-08         0     WarmupMultiStepLR                0.1  False                    0.9
 0            1.71503e-05             2  SGD             1.30161e-09        0     WarmupMultiStepLR                0    False                    0.9
 0            1.155e-05               8  SGD             3.46116e-05        0.99  WarmupMultiStepLR                0.2  True                     0.99
 0            1.69344e-05             2  SGD             6.18992e-08        0     WarmupCosineLR                   0.2  True                     0
 0            0.000153363             2  SGD             7.08727e-08        0.99  WarmupCosineLR                   0.2  True                     0.9
 0            2.24206e-05             8  SGD             0.000935443        0.95  WarmupCosineLR                   0    True                     0.1
 0            1.41607e-05             4  SGD             9.08819e-09        0     WarmupMultiStepLR                0.1  True                     0.9
 0            8.62619e-05             8  SGD             1.5441e-09         0.95  WarmupCosineLR                   0.2  True                     0.99
 0            1.74358e-05             2  SGD             0.000922939        0     WarmupMultiStepLR                0.2  True                     0
 0            0.000178316             2  SGD             0.000325052        0.99  WarmupMultiStepLR                0.2  True                     0.9
 0            3.34331e-05             4  SGD             1.42461e-10        0.99  WarmupCosineLR                   0.1  True                     0.1
"""

"""seed 456
     AP    photometric    gaussian_blur      cutout    cutout_sizes    sharpen
-------  -------------  ---------------  ----------  --------------  ---------
47.6718      0.391352         0.684079   0.123766               400  0.602619
47.6555      0.0521781        0.836699   0.886323                50  0.869857
46.6261      0.360316         0.794388   0.942531                10  0.79767
46.3668      0.617159         0.172799   0.136288               100  0.319687
44.6408      0.694609         0.524897   0.599631               200  0.395456
42.4104      0.666694         0.528346   0.484681                10  0.32306
42.2171      0.416628         0.133426   0.73587                200  0.839281
42.1322      0.106345         0.323523   0.9507                 200  0.52263
40.8937      0.248148         0.647961   0.0755293              400  0.939661
40.8196      0.536815         0.83145    0.259321                10  0.142047
40.6766      0.613538         0.655372   0.247293                10  0.23265
39.634       0.790727         0.265893   0.423685                50  0.153933
38.723       0.125755         0.777846   0.665006                10  0.944333
38.552       0.641235         0.761714   0.305478                50  0.752109
38.1894      0.522091         0.995362   0.0113984              400  0.0813109
37.9618      0.191977         0.398148   0.00848701             400  0.0411088
37.8238      0.425695         0.849157   0.875452                10  0.992641
37.4074      0.274854         0.756963   0.860107               200  0.451059
37.0909      0.980484         0.151541   0.334689                50  0.154655
36.3732      0.394841         0.485588   0.218625               200  0.87429
35.4496      0.584185         0.0877286  0.831251               200  0.748846
35.2201      0.312053         0.12136    0.817656                10  0.428006
35.1723      0.0293108        0.0962411  0.820894               100  0.280302
34.7683      0.865383         0.389009   0.891398               200  0.0137622
34.4596      0.331411         0.677455   0.712772                10  0.0312674
34.3323      0.333305         0.220546   0.458818               100  0.453526
32.4813      0.98032          0.693218   0.733472                10  0.87145
32.1217      0.0708158        0.200801   0.366426                50  0.904949
31.7011      0.46227          0.735729   0.619827               200  0.880661
31.3267      0.79784          0.55275    0.531386               200  0.312855
31.1601      0.184818         0.429563   0.305742               100  0.3208
31.1305      0.680753         0.282114   0.243947                50  0.84951
30.7565      0.868307         0.134769   0.165863               400  0.454134
30.7005      0.913739         0.360682   0.834478                50  0.973926
30.6465      0.495163         0.997809   0.123065                10  0.134492
30.6193      0.553912         0.920063   0.863384               100  0.906683
30.4984      0.407036         0.198821   0.0760762              100  0.687189
30.4428      0.275229         0.219841   0.369948               200  0.261663
29.4814      0.475411         0.382916   0.855581                50  0.697195
29.3058      0.23476          0.0722691  0.306293               100  0.412031
28.1088      0.0309322        0.976      0.781599               200  0.912199
27.5902      0.436619         0.561662   0.174402               400  0.92049
27.5882      0.985102         0.784919   0.767327               400  0.306598
27.2409      0.563789         0.117603   0.799007               400  0.0451456
27.0114      0.974688         0.0666434  0.254669               400  0.394266
26.7636      0.676244         0.529054   0.911943               200  0.483364
25.2462      0.577891         0.475403   0.290822               200  0.723981
24.3893      0.784362         0.877849   0.626355               100  0.343764
24.0149      0.481738         0.0157863  0.40284                400  0.301869
21.1582      0.348714         0.893581   0.753106               200  0.457822
"""

"""seed 568
     AP    photometric  random_types    photometric_types
-------  -------------  --------------  ----------------------------------------------------
50.3808      0.938287   False           ('brightness',)
46.2192      0.434478   False           ('contrast', 'lighting')
45.5453      0.561451   False           ('brightness', 'contrast')
42.547       0.239212   False           ('brightness', 'contrast')
40.9507      0.770509   False           ('brightness', 'contrast', 'lighting')
40.1592      0.458025   False           ('contrast', 'saturation', 'lighting')
39.7995      0.653815   False           ('contrast', 'saturation', 'lighting')
39.7672      0.0692201  False           ('saturation',)
39.7084      0.250586   False           ('brightness', 'contrast')
38.5622      0.193841   False           ('brightness', 'lighting')
37.9569      0.665721   False           ('brightness', 'contrast', 'lighting')
37.7101      0.620765   False           ('contrast', 'saturation')
37.6784      0.241455   False           ('brightness',)
37.6027      0.373491   False           ('lighting',)
36.7325      0.370889   False           ('brightness', 'contrast', 'saturation', 'lighting')
36.6054      0.558385   False           ('brightness', 'lighting')
36.5633      0.306084   False           ('brightness', 'contrast', 'saturation', 'lighting')
36.5609      0.349449   False           ('contrast', 'saturation', 'lighting')
36.2782      0.64058    False           ('contrast', 'lighting')
36.261       0.0485784  False           ('brightness', 'lighting')
36.0272      0.616056   False           ('saturation',)
35.4899      0.166924   False           ('brightness', 'saturation')
35.4523      0.400202   False           ('brightness', 'contrast', 'lighting')
35.3004      0.85455    False           ('brightness', 'contrast', 'saturation', 'lighting')
34.5487      0.509084   False           ('saturation',)
34.0365      0.377225   False           ('brightness', 'lighting')
33.0432      0.440097   False           ('brightness', 'lighting')
32.8274      0.832337   False           ('saturation', 'lighting')
32.6283      0.0803756  False           ('brightness', 'lighting')
32.3933      0.27999    False           ('saturation', 'lighting')
31.7158      0.113077   False           ('brightness', 'contrast', 'lighting')
31.5849      0.459882   False           ('contrast', 'saturation', 'lighting')
31.4547      0.0268349  False           ('brightness', 'contrast', 'lighting')
31.2562      0.413525   False           ('saturation', 'lighting')
30.823       0.764495   False           ('brightness', 'saturation')
30.7541      0.770246   False           ('contrast', 'saturation')
30.4536      0.622146   False           ('saturation',)
30.3322      0.0454054  False           ('saturation',)
30.2096      0.291754   False           ('brightness', 'contrast', 'saturation')
29.0729      0.618571   False           ('brightness', 'saturation', 'lighting')
27.9778      0.96675    False           ('brightness', 'contrast', 'saturation')
27.9476      0.78355    False           ('contrast', 'saturation')
26.9346      0.25938    False           ('contrast', 'lighting')
26.4497      0.663983   False           ('brightness', 'lighting')
26.1339      0.299219   False           ('brightness', 'contrast')
24.5771      0.882639   False           ('brightness', 'contrast')
23.612       0.454495   False           ('brightness', 'saturation')
22.8792      0.838457   False           ('brightness',)
20.7769      0.668497   False           ('contrast', 'lighting')
 0           0.266978   False           ('contrast', 'saturation')
"""


"""seed 345
     AP    photometric       noise    cam_noise    motion_blur    gaussian_blur     cutout    cutout_sizes    sharpen       clahe    channel_dropout    grayscale     invert    hist_fda    vignette    chromatic
-------  -------------  ----------  -----------  -------------  ---------------  ---------  --------------  ---------  ----------  -----------------  -----------  ---------  ----------  ----------  -----------
50.1101     0.64762     0.0155589     0.158973       0.196311        0.0374315   0.426811               10  0.0642329  0.454142            0.760862     0.244381   0.963937     1.60953            0            1
49.4836     0.571483    0.0869895     0.359013       0.167855        0.282007    0.823422               50  0.918644   0.57938             0.22176      0.226049   0.844182     2.60871            1            1
44.3714     0.787317    0.0510128     0.465235       0.0217274       0.384135    0.114888              200  0.470868   0.335263            0.684837     0.161935   0.183628     1.47092            1            1
44.271      0.907203    0.0154492     0.487386       0.364851        0.454143    0.470683               10  0.0367454  0.0686518           0.095975     0.320022   0.460831     0.961556           1            0
43.8143     0.180885    0.0495502     0.157421       0.0901788       0.233356    0.279724              100  0.887491   0.158981            0.168288     0.504966   0.401002     1.64493            1            0
42.0635     0.655225    0.0397788     0.182794       0.41172         0.190241    0.729819               10  0.471309   0.171777            0.435274     0.700776   0.348621     0.538032           0            0
41.8299     0.0607333   0.0844071     0.382018       0.411924        0.466671    0.266453               50  0.442687   0.748495            0.585849     0.229839   0.344499     1.0517             0            1
41.625      0.299199    0.0783162     0.39043        0.412544        0.318805    0.988387              200  0.913208   0.345976            0.782861     0.442353   0.726419     2.58342            0            0
40.6867     0.00858756  0.0332451     0.470503       0.365388        0.421584    0.572281               50  0.877356   0.833097            0.657615     0.675566   0.807107     0.334753           0            0
39.6472     0.34935     0.026314      0.399013       0.45864         0.162705    0.949279               10  0.909692   0.102333            0.953535     0.703561   0.779137     1.66043            0            1
39.6169     0.935855    0.032454      0.124179       0.368381        0.192615    0.647349              200  0.554211   0.131766            0.372834     0.477147   0.464126     1.94512            0            1
39.2522     0.915366    0.0191038     0.107472       0.194208        0.174096    0.785682               10  0.132018   0.682976            0.151683     0.807905   0.376638     0.278018           0            0
38.657      0.194688    0.0152317     0.239088       0.0973221       0.144111    0.529673               10  0.426959   0.04684             0.367502     0.752635   0.758657     1.3826             0            1
36.5155     0.668986    0.0828998     0.12571        0.031065        0.261232    0.0399532             100  0.153848   0.732869            0.841607     0.13664    0.73562      0.455118           1            0
35.6129     0.360568    0.010458      0.223024       0.223009        0.0629757   0.453127               10  0.799441   0.302479            0.87032      0.216562   0.844967     1.27442            0            0
35.6107     0.577416    0.0582562     0.468391       0.241126        0.258049    0.674811              200  0.206129   0.0097517           0.0634354    0.243606   0.818364     0.918893           0            1
35.3845     0.194284    0.0537923     0.170298       0.250429        0.346933    0.952692              100  0.875793   0.360842            0.195387     0.268543   0.223101     0.36467            1            0
33.2629     0.637785    0.00453624    0.133183       0.0325637       0.0161265   0.819831               50  0.204427   0.178701            0.149997     0.266447   0.191532     1.7037             0            1
32.7848     0.349553    0.0211012     0.385818       0.266338        0.486613    0.0649301             100  0.300721   0.700133            0.0981155    0.443813   0.504504     0.648452           0            1
32.1788     0.461952    0.0754743     0.18206        0.481384        0.359825    0.197387              200  0.471312   0.147348            0.670518     0.559457   0.404528     0.416235           0            0
31.2653     0.0936744   0.061049      0.0995024      0.191225        0.254667    0.758616              200  0.0848305  0.669529            0.726272     0.0452587  0.631264     2.11684            0            1
31.1386     0.652992    0.0101824     0.401474       0.215582        0.314195    0.961848              200  0.0884198  0.506018            0.989642     0.109817   0.0927999    0.956785           0            0
31.034      0.935795    0.0950487     0.49252        0.241359        0.00830058  0.0743997             200  0.99631    0.576717            0.338832     0.986805   0.733426     1.56246            0            0
31.0055     0.111164    0.0427352     0.213474       0.141056        0.322565    0.380998              100  0.208841   0.182273            0.645618     0.851649   0.790803     1.76047            0            1
30.846      0.0619881   0.0835877     0.451141       0.29881         0.135729    0.766439               50  0.27024    0.891366            0.644681     0.961442   0.586755     0.914068           0            1
29.6969     0.490474    0.0705023     0.0420874      0.250875        0.304388    0.392797               10  0.356571   0.499966            0.766701     0.663902   0.366183     0.886738           1            1
29.5906     0.276208    0.0270236     0.44641        0.0599309       0.495823    0.671236              100  0.0873302  0.0946382           0.607707     0.719834   0.574447     2.2087             0            1
29.2265     0.0964503   0.097967      0.319799       0.167927        0.358166    0.181084               50  0.505659   0.91839             0.330511     0.93067    0.78715      0.957634           0            0
28.7742     0.822705    0.0282113     0.0286902      0.29363         0.466043    0.77634                50  0.522609   0.23953             0.810271     0.843963   0.181872     0.80359            1            1
28.0995     0.107129    0.0643731     0.203274       0.110479        0.294059    0.0697643             100  0.202703   0.0677041           0.923094     0.750574   0.0706052    1.59465            0            1
27.5313     0.673267    0.0675582     0.385632       0.0116889       0.422898    0.131887               10  0.825356   0.320059            0.811432     0.190032   0.588318     1.04584            0            1
27.4364     0.613435    0.0699498     0.423225       0.375793        0.225124    0.440537              100  0.695473   0.444646            0.810386     0.664698   0.509023     2.44899            0            0
27.1972     0.721443    0.0243327     0.37752        0.3366          0.414288    0.945655              200  0.294698   0.731195            0.165884     0.19138    0.381858     0.206773           0            1
27.0343     0.0309677   0.0971587     0.429152       0.174757        0.459733    0.124301              200  0.851104   0.343532            0.184061     0.648791   0.321636     0.577914           1            0
26.4568     0.118196    0.00287842    0.256231       0.110085        0.390444    0.149862               50  0.0799797  0.395351            0.455377     0.748312   0.0875875    1.01859            0            1
26.0061     0.619836    0.0412296     0.0663801      0.237776        0.227476    0.469149               10  0.163523   0.820726            0.355484     0.663594   0.922623     0.501968           0            1
25.8765     0.701046    0.0953246     0.0509932      0.340544        0.265867    0.155448               10  0.743783   0.469533            0.201086     0.956771   0.844389     0.479117           1            0
24.8155     0.448681    0.0200893     0.401855       0.459024        0.123235    0.385512              100  0.701893   0.948529            0.793852     0.79664    0.810251     2.23837            0            0
24.4614     0.722361    0.0587681     0.38509        0.380912        0.376205    0.287765              100  0.79167    0.0249325           0.812665     0.588407   0.888991     2.38146            1            1
24.1983     0.294858    0.0220708     0.449124       0.0560543       0.393956    0.325794               50  0.360707   0.54761             0.436328     0.465104   0.632187     2.40343            0            0
24.0862     0.978913    0.0370076     0.392897       0.0511985       0.057685    0.847395               10  0.348092   0.775124            0.229622     0.714885   0.0873197    0.930779           0            0
23.9616     0.31427     0.0178765     0.0326241      0.215793        0.25352     0.62288               100  0.12171    0.272677            0.0353205    0.376612   0.0558647    2.61941            1            0
23.6286     0.517026    0.0981572     0.29362        0.315839        0.138141    0.162769               10  0.398309   0.162273            0.0614621    0.896754   0.027873     0.448894           1            0
23.0534     0.237167    0.021702      0.149441       0.140476        0.209631    0.431081               10  0.276496   0.485999            0.164391     0.857181   0.24957      2.87468            0            1
22.3794     0.695418    0.028043      0.480833       0.430141        0.222367    0.730049               10  0.826437   0.808348            0.766704     0.380912   0.917024     0.47208            0            1
21.6822     0.594682    0.0503755     0.41902        0.463015        0.413471    0.756079               10  0.601289   0.955577            0.919999     0.503516   0.0504325    2.75891            1            1
20.5031     0.517485    0.0101771     0.371301       0.373899        0.0876058   0.305629               10  0.923887   0.00465943          0.515057     0.889123   0.319415     1.62222            0            0
17.691      0.576721    0.0376602     0.223862       0.49081         0.482725    0.116682               50  0.862173   0.972583            0.891624     0.97842    0.882632     1.80285            0            0
12.1703     0.182772    0.00931149    0.180983       0.129917        0.406136    0.302822               10  0.524993   0.919905            0.863932     0.641827   0.380998     0.627577           0            1
12.1085     0.401018    0.0895861     0.403525       0.269877        0.304171    0.977037              200  0.0745726  0.0609651           0.493017     0.0653144  0.784286     2.01084            0            0
"""


"""seed 42
      AP    photometric       noise    cam_noise    motion_blur    gaussian_blur     cutout    cutout_sizes    sharpen      clahe    channel_dropout    grayscale     invert    hist_fda    vignette    chromatic    denoise    interp
--------  -------------  ----------  -----------  -------------  ---------------  ---------  --------------  ---------  ---------  -----------------  -----------  ---------  ----------  ----------  -----------  ---------  --------
52.8156      0.580687    0.0372283    0.470067        0.486832        0.14196     0.305364               50  0.448424   0.994457           0.175925     0.0180754  0.493894    0.536468            1            0          0         0
50           0.665922    0.0591298,   0.137361        0.280622        0.19146     0.971712              200  0.721730   0.235985           0.256068     0.0404336  0.710663    0.332672            1            1          0         1
48.5632      0.00753436  0.0225333    0.182678        0.243905        0.425409    0.0878876             200  0.0556535  0.842314           0.0516355    0.0182425  0.696961    2.99177             0            0          0         2
45.8097      0.796543    0.0183435    0.389846        0.298425        0.222916    0.0999749              50  0.333709   0.142867           0.650888     0.0564116  0.721999    2.81566             1            0          0         3
39.7566      0.599029    0.0826799    0.479537        0.171262        0.113675    0.423597               50  0.61495    0.911852           0.139116     0.100795   0.256016    2.17829             0            1          0         2
36.3527      0.106699    0.0850727    0.372987        0.204259        0.466469    0.990929               10  0.379229   0.926449           0.721597     0.0480946  0.781514    2.48382             0            0          0         0
35.2579      0.655723    0.0385397    0.340807        0.170312        0.130347    0.496037              100  0.348337   0.936648           0.0391863    0.417946   0.967581    1.64392             1            0          0         2
35.2296      0.0468965   0.0268672    0.0110924       0.249083        0.238105    0.831371               50  0.816386   0.967974           0.0884082    0.791818   0.589956    1.44014             1            0          0         2
35.2053      0.516636    0.0260829    0.498127        0.48271         0.279147    0.882636               10  0.278871   0.700358           0.846661     0.856324   0.404508    2.66331             0            0          0         2
34.0422      0.235692    0.0633759    0.453933        0.158098        0.294153    0.682982               50  0.713782   0.899667           0.624102     0.539781   0.438745    1.73246             1            1          0         2
33.222       0.013672    0.00753591   0.345857        0.267173        0.374955    0.913166              100  0.726121   0.757081           0.377851     0.241085   0.205045    0.754327            1            1          0         2
32.9015      0.607894    0.0379306    0.372125        0.10279         0.393895    0.603722               10  0.414505   0.863518           0.92296      0.4657     0.480837    2.75536             0            1          0         0
32.8988      0.767195    0.082279     0.372106        0.34052         0.118753    0.400223               50  0.0828913  0.52837            0.436336     0.802109   0.977901    1.66802             1            1          0         3
32.4832      0.220241    0.071115     0.404751        0.174333        0.0480883   0.940523               50  0.517751   0.83771            0.67569      0.735216   0.209072    1.62434             0            1          1         3
32.2481      0.0156364   0.0423401    0.197441        0.146744        0.00703991  0.198842              100  0.790176   0.60596            0.926301     0.651077   0.91496     2.55012             1            1          1         2
31.567       0.0230226   0.0651367    0.385623        0.187218        0.0344608   0.0773211              10  0.84044    0.910686           0.122811     0.235906   0.165521    0.558963            0            1          1         0
31.3496      0.00706631  0.00230624   0.262387        0.19993         0.0233328   0.973756               10  0.0906064  0.618386           0.382462     0.983231   0.466763    2.57982             0            1          1         3
30.4272      0.577632    0.0165513    0.0169396       0.155737        0.390248    0.277587               10  0.212681   0.515174           0.975541     0.458989   0.557305    2.5819              0            1          1         0
30.2392      0.806913    0.0346304    0.232337        0.324887        0.0240295   0.949146              200  0.260894   0.0153045          0.933436     0.50104    0.539377    2.05189             0            0          0         3
30.2255      0.768918    0.0066236    0.0229306       0.310403        0.173707    0.209131              100  0.341563   0.537263           0.460119     0.584766   0.4003      2.093               1            0          1         3
30.0006      0.714064    0.0493981    0.377318        0.051458        0.268241    0.378822               50  0.603958   0.502288           0.539861     0.486357   0.408955    2.31565             1            0          0         2
29.0204      0.636404    0.0800949    0.338584        0.286684        0.0642502   0.811204              200  0.62594    0.820427           0.651485     0.206684   0.273961    0.643767            1            1          0         1
28.4475      0.863139    0.088036     0.118343        0.453847        0.295944    0.350218              100  0.481667   0.377988           0.705084     0.248724   0.330253    1.30336             1            1          0         2
27.9055      0.665922    0.0591298    0.137361        0.280622        0.191463    0.971712              200  0.72173    0.235985           0.256068     0.0404336  0.710663    0.332672            1            1          0         1
26.9484      0.515236    0.097311     0.300968        0.111925        0.410895    0.345083               50  0.0318047  0.548715           0.534424     0.355991   0.894217    0.386245            1            1          1         1
26.3551      0.646848    0.0388268    0.114697        0.132969        0.180173    0.25995                50  0.032316   0.279764           0.411207     0.602782   0.270958    0.399555            1            0          1         3
25.956       0.919177    0.00827484   0.438331        0.275794        0.0824171   0.411255              200  0.48037    0.985286           0.376739     0.749578   0.392989    2.48749             0            1          1         2
25.2209      0.903151    0.0617264    0.490231        0.304044        0.318322    0.554816               10  0.726397   0.547446           0.45091      0.910471   0.297959    1.57081             0            0          1         3
24.5775      0.229025    0.0542849    0.215764        0.166408        0.365275    0.693718               10  0.878629   0.495424           0.741461     0.573151   0.997693    2.25721             0            0          1         0
24.4988      0.860405    0.0250251    0.0194174       0.151633        0.268541    0.326651              200  0.271543   0.965252           0.457265     0.842023   0.19438     1.23406             0            1          1         3
24.3634      0.638271    0.0516696    0.328556        0.217836        0.36502     0.0477161             100  0.158646   0.120165           0.34188      0.0917991  0.094157    0.93424             0            1          1         2
23.0151      0.324345    0.0122088    0.178149        0.453414        0.136066    0.64769                10  0.352569   0.304781           0.164656     0.534089   0.48483     2.07731             1            1          1         0
21.9179      0.563288    0.0385417    0.00798313      0.115447        0.120513    0.683264              100  0.833195   0.173365           0.391061     0.182236   0.755361    1.27547             1            0          1         3
21.123       0.0558712   0.0737036    0.272958        0.352916        0.484326    0.68803               200  0.866869   0.838481           0.426091     0.222576   0.396652    2.67569             1            0          1         3
21.1083      0.449754    0.039515     0.463329        0.363636        0.16327     0.570444              100  0.961172   0.844534           0.74732      0.539692   0.586751    2.89577             0            1          1         2
19.7751      0.25299     0.0695411    0.0377173       0.0831077       0.108405    0.294494              200  0.696925   0.384202           0.737101     0.915254   0.958702    0.173592            1            1          1         0
18.8247      0.975067    0.0490749    0.361448        0.410431        0.359229    0.535037               50  0.838583   0.205078           0.967994     0.710952   0.199507    2.20874             0            0          0         2
18.4015      0.506104    0.0932014    0.160321        0.296942        0.184615    0.454268              100  0.548922   0.20173            0.684572     0.0878681  0.138825    0.0081327           1            1          0         2
18.3223      0.611514    0.00815942   0.00259243      0.313947        0.097137    0.0709409              50  0.0507685  0.886617           0.0276168    0.578865   0.438474    2.01608             1            1          0         3
17.8473      0.54254     0.0508814    0.318166        0.125231        0.294935    0.978893               50  0.906099   0.434394           0.350078     0.645103   0.668924    2.5925              1            1          0         2
17.8085      0.447783    0.0552893    0.296348        0.0404267       0.184827    0.24216               200  0.470301   0.983423           0.398824     0.816432   0.798345    0.452153            0            0          0         0
16.4871      0.394691    0.0844213    0.465008        0.0352081       0.104459    0.671144               50  0.254164   0.295291           0.322551     0.84867    0.136621    2.12673             0            1          1         0
16.3167      0.847143    0.0354905    0.4784          0.338385        0.24126     0.493026               10  0.0917041  0.602441           0.553703     0.212728   0.946195    2.34389             1            0          0         3
15.6454      0.687785    0.0511657    0.0784888       0.188643        0.00129751  0.868301               10  0.597278   0.986257           0.536591     0.924042   0.236117    2.27987             0            0          1         2
15.5912      0.47397     0.0355104    0.324411        0.239791        0.2921      0.736822              100  0.586535   0.564459           0.378773     0.337447   0.899647    1.82267             1            1          1         3
13.7873      0.133117    0.0687166    0.42222         0.374808        0.0152361   0.867215               50  0.397164   0.104869           0.737405     0.182284   0.563965    2.52213             1            0          1         1
13.2046      0.558102    0.0403836    0.0324461       0.126958        0.123438    0.696304              100  0.148087   0.99774            0.266781     0.976615   0.411037    0.0991522           1            0          0         3
11.9328      0.0436038   0.0994551    0.234972        0.13978         0.441747    0.747719              200  0.33075    0.552765           0.572292     0.980332   0.0753463   0.917091            1            1          1         1
 5.73545     0.563276    0.0695516    0.0696657       0.302209        0.269921    0.203061              200  0.598865   0.694785           0.880468     0.624354   0.295634    0.316483            1            1          1         3
 4.94876     0.12769     0.0250016    0.290272        0.433558        0.280933    0.238597              100  0.739909   0.238236           0.377729     0.534327   0.496561    1.16885             1            1          1         3
 2.33316     0.714595    0.00410675   0.19941         0.21676         0.372021    0.250861               10  0.080873   0.428314           0.6885       0.0581936  0.915214    1.32706             1            1          1         3
"""
