import os
import glob
import random
import csv
import logging
import multiprocessing
import time
import sys
import warnings

import tabulate
import yaml
import cv2
import torch
from typing import List, Dict, Any, Set
import numpy as np
from PIL import Image, ImageStat
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma
from tqdm import tqdm
import albumentations as A
from imgaug.random import seed as imgaug_seed
from scipy.stats import ttest_ind, sem, t
from colour_demosaicing import (demosaicing_CFA_Bayer_bilinear, mosaicing_CFA_Bayer, demosaicing_CFA_Bayer_Malvar2004,
                                demosaicing_CFA_Bayer_Menon2007)
from colour import cctf_encoding
from colour.utilities import as_float_array

from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.structures import BoxMode
from detectron2.data import (DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader,
                             DatasetMapper, transforms, detection_utils as utils)
from detectron2.data.transforms import Transform, Augmentation
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.solver.build import maybe_add_gradient_clipping, build_optimizer
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger

# from .eval_loss_hook import LossEvalHook


class ReturnTransform(Transform):
    def __init__(self, image=None):
        super().__init__()
        self.image = image

    def apply_image(self, img):
        return img if self.image is None else self.image

    def apply_coords(self, coords):
        return coords

    def inverse(self):
        return self


class Mosaic(Augmentation):
    def __init__(self, pattern='RGGB'):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        img = image.copy().astype(float)[:, :, ::-1] / 255.
        cfa = mosaicing_CFA_Bayer(img, self.pattern)
        img = np.clip(as_float_array(cctf_encoding(demosaicing_CFA_Bayer_bilinear(cfa, self.pattern))), 0, 1)[:, :, ::-1]
        return ReturnTransform(image=(img * 255.).astype(np.uint8))


class Denoise(Augmentation):
    def __init__(self, mode='wavelet'):
        super().__init__()
        self._init(locals())

    def _run(self, img):
        img = img.copy().astype(float) / 255.
        if self.mode == 'wavelet':
            img = denoise_wavelet(img, multichannel=True, rescale_sigma=True)
        return (img * 255.).astype(np.uint8)

    def get_transform(self, image):
        return ReturnTransform(image=self._run(image))


class AddNoise(Augmentation):
    def __init__(self, types, amount, random_types=True):
        super().__init__()
        self._init(locals())
        if isinstance(amount, list) or isinstance(amount, tuple):
            self.gaussian = A.GaussNoise(var_limit=(amount[0], amount[1]))
        else:
            self.gaussian = A.GaussNoise(var_limit=(amount, amount))

    def _get_amount(self):
        if isinstance(self.amount, list) or isinstance(self.amount, tuple):
            return random.uniform(self.amount[0], self.amount[1])
        else:
            return self.amount

    def _run(self, img):
        noise = img.copy().astype(float) / 255.
        if self.random_types:
            types = random.sample(self.types, random.randint(0, len(self.types)))
        else:
            types = self.types

        for t in types:
            if t == 'gaussian':
                noise = self.gaussian(image=noise)['image']
            elif t == 'speckle':
                noise = random_noise(noise, t, var=self._get_amount())
            elif t == 'poisson':
                w = min(1., 1000. * self._get_amount())
                noise = random_noise(noise, t) * w + (1. - w) * noise
            elif t == 's&p':
                noise = random_noise(noise, t, amount=.1 * self._get_amount())
            else:
                raise ValueError("Unknown noise type: {}".format(t))
        return (noise * 255.).astype(np.uint8)

    def get_transform(self, image):
        return ReturnTransform(image=self._run(image))


class Photometric(Augmentation):
    def __init__(self, types, amount, random_types=True):
        super().__init__()
        self._init(locals())

    def _get_amount(self):
        if isinstance(self.amount, list) or isinstance(self.amount, tuple):
            return random.uniform(self.amount[0], self.amount[1])
        else:
            return self.amount

    def get_transform(self, image):
        composition = list()
        if self.random_types:
            types = random.sample(self.types, random.randint(0, len(self.types)))
        else:
            types = self.types

        if 'brightness' in types:
            composition.append(transforms.RandomBrightness(1. - self._get_amount(), 1. + self._get_amount()))
        if 'contrast' in types:
            composition.append(transforms.RandomContrast(1. - self._get_amount(), 1. + self._get_amount()))
        if 'lighting' in types:
            composition.append(transforms.RandomLighting(self._get_amount()))
        if 'saturation' in types:
            composition.append(transforms.RandomSaturation(1. - self._get_amount(), 1. + self._get_amount()))

        random.shuffle(composition)
        augs = transforms.AugmentationList(composition)
        return augs(transforms.AugInput(image))


class Cutout(Augmentation):
    def __init__(self, holes=(8, None), size=(8, 8, None, None), fill_value=0, p=0.5):
        super().__init__()
        self._init(locals())

        self.cutout = A.CoarseDropout(max_holes=holes[0],
                                      min_holes=holes[1],
                                      max_height=size[0],
                                      max_width=size[1],
                                      min_height=size[2],
                                      min_width=size[3],
                                      fill_value=fill_value,
                                      p=p)

    def get_transform(self, image):
        return ReturnTransform(image=self.cutout(image=image, fill_value=random.randint(0, 255))['image'])


class Flip(Augmentation):
    def __init__(self, flip_vertically=True, flip_horizontally=True, p=0.5):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        do = self._rand_range() < self.p
        if do:
            if self.flip_vertically and self.flip_horizontally:
                flips = [transforms.HFlipTransform(w), transforms.VFlipTransform(h)]
                return transforms.TransformList(random.sample(flips, random.randint(1, 2)))
            elif self.flip_horizontally:
                return transforms.HFlipTransform(w)
            elif self.flip_vertically:
                return transforms.VFlipTransform(h)
        else:
            return transforms.NoOpTransform()


class Vignetting(Augmentation):
    def __init__(self, ratio_min_dist=0.2, strength=(0.2, 0.8), random_sign=False):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        img = image.copy().astype(float) / 255.
        h, w = image.shape[:2]
        min_dist = np.array([h, w]) / 2 * np.random.random() * self.ratio_min_dist

        # create matrix of distance from the center on the two axis
        x, y = np.meshgrid(np.linspace(-w / 2, w / 2, w), np.linspace(-h / 2, h / 2, h))
        x, y = np.abs(x), np.abs(y)

        # create the vignette mask on the two axis
        x = (x - min_dist[0]) / (np.max(x) - min_dist[0])
        x = np.clip(x, 0, 1)
        y = (y - min_dist[1]) / (np.max(y) - min_dist[1])
        y = np.clip(y, 0, 1)

        # then get a random intensity of the vignette
        if isinstance(self.strength, list) or isinstance(self.strength, tuple):
            strength = np.random.uniform(*self.strength)
        else:
            strength = self.strength
        vignette = (x + y) / 2 * strength
        vignette = np.tile(vignette[..., None], [1, 1, 3])
        sign = 2 * (np.random.random() < 0.5) * self.random_sign - 1
        img = img * (1 + sign * vignette)

        return ReturnTransform(image=(img * 255.).astype(np.uint8))


class ChromaticAberration(Augmentation):
    def __init__(self, strength=0.1):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        img = Image.fromarray(image)
        r, g, b = img.split()
        rdata = np.asarray(r)

        rfinal = r
        gfinal = g
        bfinal = b

        if isinstance(self.strength, list) or isinstance(self.strength, tuple):
            strength = random.uniform(*self.strength)
        else:
            strength = self.strength
        gfinal = gfinal.resize((round((1 + random.uniform(0, 0.018) * strength) * rdata.shape[1]),
                                round((1 + random.uniform(0, 0.018) * strength) * rdata.shape[0])), Image.LANCZOS)
        bfinal = bfinal.resize((round((1 + random.uniform(0, 0.044) * strength) * rdata.shape[1]),
                                round((1 + random.uniform(0, 0.044) * strength) * rdata.shape[0])), Image.LANCZOS)

        rwidth, rheight = rfinal.size
        gwidth, gheight = gfinal.size
        bwidth, bheight = bfinal.size
        rhdiff = (bheight - rheight) // 2
        rwdiff = (bwidth - rwidth) // 2
        ghdiff = (bheight - gheight) // 2
        gwdiff = (bwidth - gwidth) // 2

        img = Image.merge("RGB", (rfinal.crop((-rwdiff, -rhdiff, bwidth - rwdiff, bheight - rhdiff)),
                                  gfinal.crop((-gwdiff, -ghdiff, bwidth - gwdiff, bheight - ghdiff)),
                                  bfinal))

        return ReturnTransform(image=np.asarray(img.crop((rwdiff, rhdiff, rwidth + rwdiff, rheight + rhdiff))))


class ApplyAEAug(Augmentation):
    def __init__(self, ae_aug, **kwargs):
        super().__init__()
        self._init(locals())

        self.aug = ae_aug(**kwargs)

    def get_transform(self, image):
        return ReturnTransform(image=self.aug(image=image, **self.kwargs)['image'])


class Trainer(DefaultTrainer):
    @classmethod
    def build_optimizer(cls, cfg, model):
        if cfg.OPTIMIZER.lower() == "adam":
            norm_module_types = (
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
                torch.nn.SyncBatchNorm,
                torch.nn.GroupNorm,
                torch.nn.InstanceNorm1d,
                torch.nn.InstanceNorm2d,
                torch.nn.InstanceNorm3d,
                torch.nn.LayerNorm,
                torch.nn.LocalResponseNorm,
            )
            params: List[Dict[str, Any]] = []
            memo: Set[torch.nn.parameter.Parameter] = set()
            for module in model.modules():
                for key, value in module.named_parameters(recurse=False):
                    if not value.requires_grad:
                        continue
                    if value in memo:
                        continue
                    memo.add(value)
                    lr = cfg.SOLVER.BASE_LR
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY
                    if isinstance(module, norm_module_types):
                        weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
                    elif key == "bias":
                        lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                        weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
                    params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

            optimizer = torch.optim.Adam(params, cfg.SOLVER.BASE_LR)
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
            return optimizer
        elif cfg.OPTIMIZER in ["SGD", "sgd"]:
            return build_optimizer(cfg, model)
        else:
            raise ValueError

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(cfg.DATASETS.TEST[0], output_dir=cfg.OUTPUT_DIR)

    """
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks
    """

    @classmethod
    def build_train_loader(cls, cfg):
        augmentations = [transforms.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                                                       max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                                                       sample_style="range" if len(
                                                           cfg.INPUT.MIN_SIZE_TRAIN) == 2 else "choice",
                                                       interp=cfg.INTERP)]
        if cfg.FLIP:
            if cfg.FLIP in ["vertical", "vertically"]:
                flips = [True, False]
            elif cfg.FLIP in ["horizontal", "horizontally"]:
                flips = [False, True]
            else:
                flips = [True, True]
                warnings.filterwarnings("ignore", category=UserWarning)
            augmentations.append(Flip(*flips))
        if cfg.ROTATE:
            augmentations.append(transforms.RandomRotation([90, 180, 270, -90, -180, -270], sample_style="choice"))
        if cfg.INVERT:
            augmentations.append(ApplyAEAug(A.InvertImg, p=cfg.INVERT))
        if cfg.CHANNEL_DROPOUT:
            augmentations.append(ApplyAEAug(A.ChannelDropout, channel_drop_range=(1, 2), p=cfg.CHANNEL_DROPOUT))
        if cfg.GRAYSCALE:
            augmentations.append(ApplyAEAug(A.ToGray, always_apply=True))
        if cfg.CUTOUT:
            augmentations.append(Cutout(holes=cfg.CUTOUT_HOLES, size=cfg.CUTOUT_SIZES, p=cfg.CUTOUT))
        if cfg.PHOTOMETRIC:
            augmentations.append(
                Photometric(types=cfg.PHOTOMETRIC_TYPES, amount=cfg.PHOTOMETRIC, random_types=cfg.RANDOM_TYPES))
        if cfg.CLAHE:
            augmentations.append(ApplyAEAug(A.CLAHE, p=cfg.CLAHE))
        if cfg.CHROMATIC_ABERRATION:
            augmentations.append(ChromaticAberration(strength=cfg.CHROMATIC_ABERRATION))
        if cfg.VIGNETTE:
            augmentations.append(Vignetting(strength=cfg.VIGNETTE))
        if cfg.MOTION_BLUR:
            augmentations.append(ApplyAEAug(A.MotionBlur, blur_limit=cfg.MB_KERNEL_SIZE, p=cfg.MOTION_BLUR))
        if cfg.NOISE:
            augmentations.append(AddNoise(types=cfg.NOISE_TYPES, amount=cfg.NOISE, random_types=cfg.RANDOM_TYPES))
        if cfg.CAM_NOISE:
            augmentations.append(ApplyAEAug(A.ISONoise, color_shift=cfg.CAM_NOISE_SHIFT, intensity=cfg.CAM_NOISE,
                                            always_apply=True))
        if cfg.MOSAIC:
            augmentations.append(Mosaic())
        if cfg.SHARPEN:
            augmentations.append(ApplyAEAug(A.IAASharpen, p=cfg.SHARPEN))
        if cfg.DENOISE:
            augmentations.append(Denoise())
        if cfg.FDA:
            augmentations.append(ApplyAEAug(A.FDA, reference_images=cfg.REFERENCE, beta_limit=cfg.FDA,
                                            always_apply=True))
        if cfg.HISTOGRAM:
            augmentations.append(ApplyAEAug(A.HistogramMatching, reference_images=cfg.REFERENCE,
                                            blend_ratio=cfg.HISTOGRAM, always_apply=True))
        if cfg.JPEG:
            augmentations.append(ApplyAEAug(A.JpegCompression, always_apply=True))
        mapper = DatasetMapper(is_train=True, augmentations=augmentations, image_format='BGR')
        return build_detection_train_loader(cfg, mapper, num_workers=cfg.DATALOADER.NUM_WORKERS)


def get_justin_dicts(directory: str):
    img_paths = sorted(glob.glob(os.path.join(directory, "*/*.png"), recursive=True))
    info_paths = sorted(glob.glob(os.path.join(directory, "*/*.yaml"), recursive=True))

    data_dicts = []
    for index, (img_path, info_path) in enumerate(zip(img_paths, info_paths)):
        with open(info_path, 'r') as f:
            img_info = yaml.safe_load(f)

        record = {
            "file_name": img_path,
            "image_id": index,
            "height": img_info["meta_data"]["original_height"],
            "width": img_info["meta_data"]["original_width"]
        }

        labels = img_info["labels"]
        objects = []
        for label in labels:
            bbox = label["bbox"]
            obj = {
                "bbox": [round(bbox["minx"] * record["width"]),
                         round(bbox["miny"] * record["height"]),
                         round(bbox["maxx"] * record["width"]),
                         round(bbox["maxy"] * record["height"])],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
            }
            objects.append(obj)
        record["annotations"] = objects
        data_dicts.append(record)
    return data_dicts


def save_dicts_as_csv(data_dicts, path=""):
    with open(os.path.join(path, "data.csv") if path else "data.csv", 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for d in data_dicts:
            for obj in d["annotations"]:
                csvwriter.writerow([d["file_name"],
                                    int(obj["bbox"][0]),
                                    int(obj["bbox"][1]),
                                    int(obj["bbox"][2]),
                                    int(obj["bbox"][3]),
                                    "case"])


def load_data(data_root, splits=("train", "val", "test", "misc", "justin")):
    coco = {0: u'__background__',
            1: u'person',
            2: u'bicycle',
            3: u'car',
            4: u'motorcycle',
            5: u'airplane',
            6: u'bus',
            7: u'train',
            8: u'truck',
            9: u'boat',
            10: u'traffic light',
            11: u'fire hydrant',
            12: u'stop sign',
            13: u'parking meter',
            14: u'bench',
            15: u'bird',
            16: u'cat',
            17: u'dog',
            18: u'horse',
            19: u'sheep',
            20: u'cow',
            21: u'elephant',
            22: u'bear',
            23: u'zebra',
            24: u'giraffe',
            25: u'backpack',
            26: u'umbrella',
            27: u'handbag',
            28: u'tie',
            29: u'suitcase',
            30: u'frisbee',
            31: u'skis',
            32: u'snowboard',
            33: u'sports ball',
            34: u'kite',
            35: u'baseball bat',
            36: u'baseball glove',
            37: u'skateboard',
            38: u'surfboard',
            39: u'tennis racket',
            40: u'bottle',
            41: u'wine glass',
            42: u'cup',
            43: u'fork',
            44: u'knife',
            45: u'spoon',
            46: u'bowl',
            47: u'banana',
            48: u'apple',
            49: u'sandwich',
            50: u'orange',
            51: u'broccoli',
            52: u'carrot',
            53: u'hot dog',
            54: u'pizza',
            55: u'donut',
            56: u'cake',
            57: u'chair',
            58: u'couch',
            59: u'potted plant',
            60: u'bed',
            61: u'dining table',
            62: u'toilet',
            63: u'tv',
            64: u'laptop',
            65: u'mouse',
            66: u'remote',
            67: u'keyboard',
            68: u'cell phone',
            69: u'microwave',
            70: u'oven',
            71: u'toaster',
            72: u'sink',
            73: u'refrigerator',
            74: u'book',
            75: u'clock',
            76: u'vase',
            77: u'scissors',
            78: u'teddy bear',
            79: u'hair drier',
            80: u'toothbrush'}

    for split in splits:
        DatasetCatalog.register(f"justin_{split}", lambda split=split: get_justin_dicts(os.path.join(data_root, split)))
        MetadataCatalog.get(f"justin_{split}").set(thing_classes=["1"])


def test_data(name):
    metadata = MetadataCatalog.get(name)
    data_dicts = DatasetCatalog.get(name)

    print(f"Found {len(data_dicts)} images. Showing 10:")
    for d in random.sample(data_dicts, 10):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow(d["file_name"], out.get_image()[:, :, ::-1])
    cv2.waitKey()


def visualize_results(predictor, dataset="", image_path=""):
    if dataset:
        data_dicts = DatasetCatalog.get(dataset)
        metadata = MetadataCatalog.get('justin_test')

        for d in random.sample(data_dicts, 10):
            img = cv2.imread(d["file_name"])
            outputs = predictor(img[:, :, ::-1])
            v = Visualizer(img[:, :, ::-1], metadata=metadata)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow(str(d["image_id"]), out.get_image()[:, :, ::-1])
        cv2.waitKey()
    elif image_path:
        img = cv2.imread(image_path)
        outputs = predictor(img[:, :, ::-1])
        v = Visualizer(img[:, :, ::-1])
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow(image_path, out.get_image()[:, :, ::-1])
        cv2.waitKey()


def visualize_data(cfg, data_loader):
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    for batch in data_loader:
        for per_image in batch:
            img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
            img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)

            visualizer = Visualizer(img, metadata=metadata, scale=1.0)
            target_fields = per_image["instances"].get_fields()
            labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
            vis = visualizer.overlay_instances(
                labels=labels,
                boxes=target_fields.get("gt_boxes", None),
                masks=target_fields.get("gt_masks", None),
                keypoints=target_fields.get("gt_keypoints", None),
            )
            cv2.imshow(str(per_image['image_id']), vis.get_image()[:, :, ::-1])
            cv2.waitKey()


def set_seed(seed):
    imgaug_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def build_config(train_datasets, base_config, output_dir, epochs=2):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(base_config))

    cfg.DATASETS.TRAIN = train_datasets
    cfg.DATASETS.TEST = ("justin_test",)

    cfg.DATALOADER.NUM_WORKERS = multiprocessing.cpu_count() - 2

    cfg.INPUT.MIN_SIZE_TRAIN = (800,)  # (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.CROP.ENABLED = False
    cfg.INPUT.CROP.SIZE = [.9, .9]

    cfg.SOLVER.IMS_PER_BATCH = 4

    cfg.NUM_BATCHES = 0
    for ds in cfg.DATASETS.TRAIN:
        cfg.NUM_BATCHES += len(DatasetCatalog.get(ds))
    cfg.NUM_BATCHES = cfg.NUM_BATCHES // cfg.SOLVER.IMS_PER_BATCH
    cfg.EPOCHS = epochs

    cfg.SOLVER.BASE_LR = .0001
    cfg.SOLVER.MAX_ITER = cfg.EPOCHS * cfg.NUM_BATCHES
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg.NUM_BATCHES  # Checkpoint every epoch
    cfg.SOLVER.WEIGHT_DECAY = 0.
    cfg.SOLVER.MOMENTUM = .9
    cfg.SOLVER.NESTEROV = False
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    # Linear warm up to base learning rate within one 10th of all iterations
    cfg.SOLVER.WARMUP_ITERS = 1.  # int(round(cfg.SOLVER.MAX_ITER * 0.2))
    cfg.SOLVER.WARMUP_FACTOR = 1. / float(cfg.SOLVER.WARMUP_ITERS)
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = False
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = .001
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.
    # Reduce learning rate by a factor of 10 every 2 epochs
    cfg.SOLVER.STEPS = (1,)  # list(cfg.NUM_BATCHES * np.arange(1, cfg.EPOCHS, 2))
    cfg.SOLVER.GAMMA = 1.

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_config)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.5

    cfg.TEST.EVAL_PERIOD = 100
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.OPTIMIZER = "ADAM"
    cfg.LOGGER = True

    cfg.ROTATE = False
    cfg.PHOTOMETRIC = []
    cfg.PHOTOMETRIC_TYPES = ['brightness', 'contrast', 'saturation', 'lighting']
    cfg.INTERP = Image.BILINEAR
    cfg.NOISE = []
    cfg.NOISE_TYPES = ['poisson', 'gaussian', 'speckle', 's&p']
    cfg.CAM_NOISE = []  # [0.1, 0.5]
    cfg.CAM_NOISE_SHIFT = []  # [0.01, 0.05]
    cfg.RANDOM_TYPES = True
    cfg.DENOISE = False
    cfg.CUTOUT = 0.
    cfg.CUTOUT_SIZES = [100, 100, None, None]
    cfg.CUTOUT_HOLES = [5, 1]
    cfg.MOTION_BLUR = 0.
    cfg.MB_KERNEL_SIZE = 51
    cfg.FLIP = "both"
    cfg.INVERT = 0.
    cfg.GRAYSCALE = False
    cfg.CHANNEL_DROPOUT = 0.
    cfg.HISTOGRAM = []
    cfg.FDA = 0.
    cfg.REFERENCE = [sample['file_name'] for sample in random.sample(DatasetCatalog.get(cfg.DATASETS.TEST[0]), 100)]
    cfg.SHARPEN = 0.
    cfg.CLAHE = 0.
    cfg.MOSAIC = False
    cfg.JPEG = False
    cfg.VIGNETTE = 0.  # (0.3, 0.8)
    cfg.CHROMATIC_ABERRATION = 0.  # (0., 0.2)

    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def get_model(cfg):
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    return model


def train_eval(cfg, resume=False):
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume)

    trainer.train()
    ap_it = np.array(trainer.storage.history(name='bbox/AP').values())
    ap = ap_it[:, 0]
    it = ap_it[:, 1]
    del trainer
    torch.cuda.empty_cache()
    return ap, it


def evaluate(cfg, model=None):
    if model is None:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        model = get_model(cfg).eval()
    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], num_workers=cfg.DATALOADER.NUM_WORKERS)
    return inference_on_dataset(model, val_loader, evaluator)


def predict(cfg, dataset="", image_path=""):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)
    if dataset:
        visualize_results(predictor, dataset=dataset)
    elif image_path:
        visualize_results(predictor, image_path=image_path)
    else:
        visualize_results(predictor, dataset=cfg.DATASETS.TEST[0])


def load_datasets(train_root, eval_root):
    load_data(eval_root, splits=['val', 'test', 'train'])

    top = os.path.join(train_root, "case")
    for dirpath, dirnames, filenames in os.walk(top):
        for name in dirnames:
            register_coco_instances(name, {}, os.path.join(top, name, "coco_annotations.json"), os.path.join(top, name))
        break


def compute_data_statistics(loader):
    box_color = []
    box_area = []
    brightness = []
    num_images = 0
    for batch in loader:
        for per_image in batch:
            img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
            img = utils.convert_image_to_rgb(img, 'BGR')

            gray = Image.fromarray(img).convert('L')
            brightness.append(ImageStat.Stat(gray).mean[0])
            gray.close()

            target_fields = per_image["instances"].get_fields()
            boxes = target_fields.get("gt_boxes", None)
            for box in boxes:
                if box is not None:
                    x_vals = np.random.randint(box[0], box[2], 10)
                    y_vals = np.random.randint(box[1], box[3], 10)
                    for x, y in zip(x_vals, y_vals):
                        box_color.append(img[y, x, :])

                    box_area.append((box[2] - box[0]) * (box[3] - box[1]))
        num_images += len(batch)
        if num_images == 1000:
            break
    # Color
    # Train mean: [60 81 70] [ 0.23680145  0.31585956  0.27415616]
    # Val mean: [103 130 121] [ 0.40543296  0.5110567   0.47583678]
    # Test mean: [103 128 119] [ 0.40352941  0.50206041  0.46821198]

    # Box Area
    # justin_train: mean: 6220.56 median: 3755.0 min/max: 340.0 34933.0
    # justin_test: mean: 3518.72 median: 3150.0 min/max: 1404.0 10980.0
    # case_color: mean: 5374.07 median: 4139.0 min/max: 22.0 22610.0
    # case_far: mean: 2852.87 median: 2597.0 min/max: 5.0 9991.0
    # case_less_light: mean: 5609.78 median: 4204.5 min/max: 45.0 35475.0
    # case_two_lights: mean: 5260.08 median: 4212.0 min/max: 1.0 24549.0
    # case_no_point_light: mean: 5168.95 median: 4109.0 min/max: 12.0 24332.0
    # case_point_lights_only: mean: 5770.44 median: 4657.5 min/max: 12.0 22240.0

    # Brightness
    # justin_train: mean: 112.037592849 median: 109.869568885 min/max: 55.1408388004 188.965364339
    # justin_test: mean: 178.212240754 median: 178.790208529 min/max: 119.67949508 232.593474695
    # case_color: mean: 142.690784431 median: 144.398138472 min/max: 27.8768802718 242.947983833
    # case_far: mean: 148.495684196 median: 152.700051546 min/max: 22.223881209 242.785112465
    # case_less_light: mean: 105.731650155 median: 107.670424086 min/max: 8.8248559044 201.306090675
    # case_two_lights: mean: 153.63752325 median: 161.519141284 min/max: 17.7638484067 245.784960169
    # case_no_point_light: mean: 101.867872943 median: 96.3478830834 min/max: 13.4076148079 232.4852015
    # case_point_lights_only: mean: 126.674849234 median: 123.876030928 min/max: 21.6544857076 234.427679241

    print("Mean value of bbox center pixels:",
          np.round(np.mean(box_color, axis=0)).astype(np.uint8),
          np.mean(box_color, axis=0) / 255.)
    print("Std:", np.std(box_color, axis=0).astype(np.uint8))

    print("Box Area")
    print("mean:", np.mean(box_area), "median:", np.median(box_area), "min/max:", np.min(box_area),
          np.max(box_area))
    print("histogram:")
    print(np.histogram(box_area)[0])

    print("Brightness")
    print("mean:", np.mean(brightness), "median:", np.median(brightness), "min/max:", np.min(brightness),
          np.max(brightness))
    print("histogram:")
    print(np.histogram(brightness)[0])


def main(visualize=False):
    start = time.time()
    set_seed(42)
    setup_logger(name="case")

    train_root = "/home/matthias/Data/Ubuntu/data/datasets"
    eval_root = "/home/matthias/Data/Ubuntu/data/datasets/justin"
    load_datasets(train_root, eval_root)

    base_config = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    output_dir = "/home/matthias/Data/Ubuntu/data/justin_training"
    cfg = build_config(train_datasets=("case_smooth",), base_config=base_config, output_dir=output_dir, epochs=2)

    if visualize:
        trainer = Trainer(cfg)
        loader = trainer.build_train_loader(cfg)
        # compute_data_statistics(loader)
        visualize_data(cfg, loader)

    results = list()
    for seed in np.arange(5):
        set_seed(seed)
        ap, it = train_eval(cfg, resume=False)
        result = np.vstack([ap, it]).T
        result = result[result[:, 0].argmax()].tolist()
        result.append(seed)
        results.append(result)
    results = np.array(results)

    ap = results[:, 0]
    print(tabulate.tabulate(results[ap.argsort()[::-1]], headers=["AP", "iter", "seed"]))
    print()
    print("Mean AP:", ap.mean())
    print("95% conf. interv.:", t.interval(0.95, len(ap) - 1, loc=np.mean(ap), scale=sem(ap)))
    print("99% conf. interv.:", t.interval(0.99, len(ap) - 1, loc=np.mean(ap), scale=sem(ap)))
    print("Copy:", np.sort(ap)[::-1])
    print()

    res_dict = {"case_color": [40.20318213, 35.35496166, 31.02521989, 29.54828771, 27.14293436,
                               27.09203161, 26.40146892, 23.90050957, 23.23194967, 22.82641386],  # checked
                "case_green": [42.07711803, 33.84280772, 33.08787521, 31.47131132, 30.18678932],  # checked
                "case_white": [21.0838, 17.2644, 12.5009, 11.3386, 6.06284],
                "case_flat": [37.91878455, 30.64923742, 28.30067853, 27.75824945, 27.34716009],  # checked
                "case_smooth": [30.82600167, 30.42064908, 27.06868854, 25.34457596, 20.87147743],  # checked
                "case_refined": [22.3332, 21.3385, 21.0649, 18.4229, 14.5228],
                "case_cam_k": [29.96887896, 26.99492782, 22.50186606, 16.9611616, 15.7358362],  # checked
                "case_all_white": [12.8596, 12.1132, 10.5135, 10.4775, 8.79655],
                "case_50_samples": [29.6049, 23.6589, 23.5817, 21.2605, 20.5096],
                "case_512_samples": [26.1091, 23.8615, 22.8284, 21.5225, 9.46198],
                "case_2048_samples": [25.51789534, 23.77423735, 22.73706366, 22.58643221, 20.10152691],  # checked
                "case_color_texture": [42.27489632, 33.6542957, 32.44540635, 30.33389404, 20.45308856],  # checked
                "case_texture": [30.93216935, 30.5514081, 28.04813986, 24.42527646, 19.33343886],  # checked
                "case_more_bounces": [33.83921761, 29.46399028, 27.03319279, 26.84496801, 25.03445091],  # checked
                "case_area_light": [28.64929296, 26.14490815, 21.8578116, 19.71479663, 17.28886026],  # checked
                "case_cam_uniform3d_inplane": [31.96673795, 31.26197291, 30.49044743, 24.15643791, 23.57493626],  # checked
                "case_cam_uniform3d": [26.97584852, 26.62969497, 26.14628782, 21.09606741, 20.16900372],  # checked
                "case_cam_poi": [10.86430437, 7.79265517, 7.42004987, 6.64692292, 6.02255621],  # checked
                "case_white_light": [31.87034131, 29.22695543, 26.59384891, 23.18313712, 20.93746476],  # checked
                "case_no_bounces": [32.4185546, 28.64617559, 25.05437995, 23.88907582, 19.67199348],  # checked
                "case_uniform_clutter": [33.40435191, 31.27687028, 26.87701708, 25.97052168, 23.90796012],  # checked
                "case_clutter": [36.49185575, 30.14689522, 25.91438832, 25.32848971, 20.45497139],  # checked
                "case_high_res": [32.10037995, 27.17417729, 17.91426037, 14.40626157, 12.24492306],  # checked
                "case_point_lights_only": [27.49361244, 25.31737831, 21.3253789, 14.1208087, 11.48722047],  # checked
                "case_less_light": [26.20951466, 23.10560782, 21.613864, 20.07971469, 4.9723414],  # checked
                "case_two_lights": [26.07114511, 19.92342411, 11.68455528, 9.60110245, 3.3309312],  # checked
                "case_even_less_cam_rotation": [41.14436033, 37.58032475, 35.9072579, 34.13007927, 30.49871135],  # checked
                "case_fov": [36.76167819, 35.08760321, 31.27037743, 30.37089514, 26.70382605],  # checked
                "case_less_cam_rotation": [25.7747, 18.8846, 18.2181, 17.4002, 10.1219],  # checked
                "case_train_green": [36.01153481, 34.57256129, 31.54693327, 30.22182292, 22.28776532],  # checked
                "case_test_green": [27.6192, 21.9087, 21.3022, 20.4311, 17.6845],  # checked
                "case_no_specular": [19.30362239, 18.9428661, 17.37443433, 13.01278383, 11.54669053],  # checked
                "case_no_denoiser": [30.0072, 27.3804, 25.6204, 22.7732, 19.3679],  # checked
                "case_no_glossy": [35.6989, 29.8466, 29.3413, 25.6552, 17.015],  # checked
                "case_no_rough": [35.69471537, 33.93573952, 30.49634283, 30.43674663, 24.45885994],  # checked
                "case_no_point_light": [40.83368553, 34.71397511, 32.09706752, 27.477971, 25.45296477]}  # checked

    for k, v in res_dict.items():
        # The probability to obtain the current result by chance (assuming the same data generating process) is ...
        print(f"Statistics for {cfg.DATASETS.TRAIN[0]} vs {k}:")
        print(ttest_ind(ap, v, equal_var=False))
        print(f"99% conf. interv. of {k}:", t.interval(0.99, len(v) - 1, loc=np.mean(v), scale=sem(v)))
        print()
        # ... or less.

    print("Runtime:", (time.time() - start) / 60)


if __name__ == "__main__":
    main(visualize=False)
