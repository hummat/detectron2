import numpy as np
import cv2
import tabulate
import torch
import yaml
from PIL import Image, ImageStat
from scipy.stats import sem, t, ttest_ind

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data import detection_utils as utils
from detectron2.data import transforms
from detectron2.data.datasets import register_coco_instances
from detectron2.data.transforms import Augmentation, Transform
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.solver.build import build_optimizer, maybe_add_gradient_clipping
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

import albumentations as A
import argparse
import csv
import glob
import logging
import os
import random
import skopt
import time
import warnings
from autoaugment import ImageNetPolicy
from colour import cctf_encoding
from colour.utilities import as_float_array
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear, mosaicing_CFA_Bayer
from imgaug.random import seed as imgaug_seed
from skimage.restoration import denoise_wavelet
from skimage.util import random_noise
from typing import Any, Dict, List, Set, Tuple, Union
from utils import get_param_names, get_results_dict, get_space, parse_data, set_cfg_values

# from .eval_loss_hook import LossEvalHook


class ReturnTransform(Transform):

    def __init__(self, image: np.ndarray = None):
        super().__init__()
        self.image = image

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return img if self.image is None else self.image

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def inverse(self):
        return transforms.NoOpTransform()


class AutoAugment(Augmentation):

    def __init__(self):
        super().__init__()
        self._init(locals())
        self.policy = ImageNetPolicy()

    def get_transform(self, image):
        return ReturnTransform(
            image=np.asarray(self.policy(Image.fromarray(image))))


class Mosaic(Augmentation):

    def __init__(self, pattern: str = "RGGB"):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        img = image.copy().astype(float)[:, :, ::-1] / 255.0
        cfa = mosaicing_CFA_Bayer(img, self.pattern)
        img = np.clip(
            as_float_array(
                cctf_encoding(demosaicing_CFA_Bayer_bilinear(cfa,
                                                             self.pattern))),
            0,
            1,
        )[:, :, ::-1]
        return ReturnTransform(image=(img * 255.0).astype(np.uint8))


class Denoise(Augmentation):

    def __init__(self, mode="wavelet"):
        super().__init__()
        self._init(locals())

    def _run(self, img):
        img = img.copy().astype(float) / 255.0
        if self.mode == "wavelet":
            img = denoise_wavelet(
                img,
                multichannel=len(img.shape) == 3 and img.shape[2] == 3,
                rescale_sigma=True,
            )
        return (img * 255.0).astype(np.uint8)

    def get_transform(self, image):
        return ReturnTransform(image=self._run(image))


class AddNoise(Augmentation):

    def __init__(
        self,
        types: Union[List, Tuple],
        amount: Union[List, Tuple, float],
        random_types: bool = True,
    ):
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
        noise = img.copy().astype(float) / 255.0
        if self.random_types:
            types = random.sample(self.types,
                                  random.randint(0, len(self.types)))
        else:
            types = self.types

        for _type in types:
            if _type == "gaussian":
                noise = self.gaussian(image=noise)["image"]
            elif _type == "speckle":
                noise = random_noise(noise, _type, var=self._get_amount())
            elif _type == "poisson":
                w = min(1.0, 1000.0 * self._get_amount())
                noise = random_noise(noise, _type) * w + (1.0 - w) * noise
            elif _type == "s&p":
                noise = random_noise(noise,
                                     _type,
                                     amount=0.1 * self._get_amount())
            else:
                raise ValueError("Unknown noise type: {}".format(_type))
        return (noise * 255.0).astype(np.uint8)

    def get_transform(self, image):
        return ReturnTransform(image=self._run(image))


class Photometric(Augmentation):

    def __init__(self, types, amount, random_types=True):
        super().__init__()
        self._init(locals())
        if isinstance(amount, (list, tuple)):
            assert len(types) == len(amount)

    def _get_amount(self, idx):
        if isinstance(self.amount, (list, tuple)):
            if isinstance(self.amount[idx], (list, tuple)):
                return random.uniform(self.amount[idx][0], self.amount[idx][1])
            else:
                return self.amount[idx]
        else:
            return self.amount

    def get_transform(self, image):
        composition = list()
        if self.random_types:
            types = random.sample(self.types,
                                  random.randint(0, len(self.types)))
        else:
            types = list(self.types)

        if "brightness" in types:
            offset = self._get_amount(self.types.index("brightness"))
            composition.append(
                transforms.RandomBrightness(1.0 - min(offset, 0.5),
                                            1.0 + min(offset, 0.5)))
        if "contrast" in types:
            offset = self._get_amount(self.types.index("contrast"))
            composition.append(
                transforms.RandomContrast(1.0 - offset, 1.0 + offset))
        if "lighting" in types:
            composition.append(
                transforms.RandomLighting(
                    self._get_amount(self.types.index("lighting"))))
        if "saturation" in types:
            offset = self._get_amount(self.types.index("saturation"))
            composition.append(
                transforms.RandomSaturation(1.0 - offset, 1.0 + offset))

        random.shuffle(composition)
        augs = transforms.AugmentationList(composition)
        return augs(transforms.AugInput(image))


class Cutout(Augmentation):

    def __init__(self,
                 holes=(8, None),
                 size=(8, 8, None, None),
                 fill_value=0,
                 p=0.5):
        super().__init__()
        self._init(locals())

        self.cutout = A.CoarseDropout(
            max_holes=holes[0],
            min_holes=holes[1],
            max_height=size[0],
            max_width=size[1],
            min_height=size[2],
            min_width=size[3],
            fill_value=fill_value,
            p=p,
        )

    def get_transform(self, image):
        return ReturnTransform(image=self.cutout(image=image)["image"])


class Flip(Augmentation):

    def __init__(
        self,
        flip_vertically: bool = True,
        flip_horizontally: bool = True,
        p: float = 0.5,
    ):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        do = self._rand_range() < self.p
        if do:
            if self.flip_vertically and self.flip_horizontally:
                flips = [
                    transforms.HFlipTransform(w),
                    transforms.VFlipTransform(h)
                ]
                return transforms.TransformList(
                    random.sample(flips, random.randint(1, 2)))
            elif self.flip_horizontally:
                return transforms.HFlipTransform(w)
            elif self.flip_vertically:
                return transforms.VFlipTransform(h)
        else:
            return transforms.NoOpTransform()


class Vignetting(Augmentation):

    def __init__(
            self,
            ratio_min_dist: float = 0.2,
            strength: Tuple[float, float] = (0.2, 0.8),
            random_sign: bool = False,
    ):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        img = image.copy().astype(float) / 255.0
        h, w = image.shape[:2]
        min_dist = np.array([h, w
                            ]) / 2 * np.random.random() * self.ratio_min_dist

        # create matrix of distance from the center on the two axis
        x, y = np.meshgrid(np.linspace(-w / 2, w / 2, w),
                           np.linspace(-h / 2, h / 2, h))
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

        return ReturnTransform(image=(img * 255.0).astype(np.uint8))


class ChromaticAberration(Augmentation):

    def __init__(self, strength=0.1):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        img = Image.fromarray(image).convert("RGB")
        r, g, b = img.split()
        rdata = np.asarray(r)

        rfinal = r
        gfinal = g
        bfinal = b

        if isinstance(self.strength, list) or isinstance(self.strength, tuple):
            strength = random.uniform(*self.strength)
        else:
            strength = self.strength
        gfinal = gfinal.resize(
            (
                round(
                    (1 + random.uniform(0, 0.018) * strength) * rdata.shape[1]),
                round(
                    (1 + random.uniform(0, 0.018) * strength) * rdata.shape[0]),
            ),
            Image.LANCZOS,
        )
        bfinal = bfinal.resize(
            (
                round(
                    (1 + random.uniform(0, 0.044) * strength) * rdata.shape[1]),
                round(
                    (1 + random.uniform(0, 0.044) * strength) * rdata.shape[0]),
            ),
            Image.LANCZOS,
        )

        rwidth, rheight = rfinal.size
        gwidth, gheight = gfinal.size
        bwidth, bheight = bfinal.size
        rhdiff = (bheight - rheight) // 2
        rwdiff = (bwidth - rwidth) // 2
        ghdiff = (bheight - gheight) // 2
        gwdiff = (bwidth - gwidth) // 2

        img = Image.merge(
            "RGB",
            (
                rfinal.crop(
                    (-rwdiff, -rhdiff, bwidth - rwdiff, bheight - rhdiff)),
                gfinal.crop(
                    (-gwdiff, -ghdiff, bwidth - gwdiff, bheight - ghdiff)),
                bfinal,
            ),
        )
        img = np.asarray(
            img.crop((rwdiff, rhdiff, rwidth + rwdiff,
                      rheight + rhdiff)))[:, :, ::-1]
        return ReturnTransform(image=img)


class ApplyAEAug(Augmentation):

    def __init__(self, ae_aug, **kwargs):
        super().__init__()
        self._init(locals())

        self.aug = ae_aug(**kwargs)

    def get_transform(self, image):
        return ReturnTransform(
            image=self.aug(image=image, **self.kwargs)["image"])


class Trainer(DefaultTrainer):

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        return model

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
                    params += [{
                        "params": [value],
                        "lr": lr,
                        "weight_decay": weight_decay
                    }]

            optimizer = torch.optim.Adam(params, cfg.SOLVER.BASE_LR)
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
            return optimizer
        elif cfg.OPTIMIZER.lower() == "sgd":
            return build_optimizer(cfg, model)
        else:
            raise ValueError

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(
            cfg.DATASETS.TEST[0],
            tasks=("bbox",) if "justin" in cfg.DATASETS.TEST[0] else None,
            output_dir=cfg.OUTPUT_DIR,
        )

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
        augmentations = [
            transforms.ResizeShortestEdge(
                short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                sample_style="range"
                if len(cfg.INPUT.MIN_SIZE_TRAIN) == 2 else "choice",
                interp=cfg.INTERP,
            )
        ]
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
            augmentations.append(
                transforms.RandomRotation([90, 180, 270, -90, -180, -270],
                                          sample_style="choice"))
        if cfg.INVERT:
            augmentations.append(ApplyAEAug(A.InvertImg, p=cfg.INVERT))
        if cfg.CHANNEL_DROPOUT:
            augmentations.append(
                ApplyAEAug(A.ChannelDropout,
                           channel_drop_range=(1, 2),
                           p=cfg.CHANNEL_DROPOUT))
        if cfg.GRAYSCALE:
            augmentations.append(ApplyAEAug(A.ToGray, p=cfg.GRAYSCALE))
        if cfg.PHOTOMETRIC:
            augmentations.append(
                Photometric(
                    types=cfg.PHOTOMETRIC_TYPES,
                    amount=cfg.PHOTOMETRIC,
                    random_types=cfg.RANDOM_TYPES,
                ))
        if cfg.AUTO_AUGMENT:
            augmentations.append(AutoAugment())
        if cfg.CLAHE:
            augmentations.append(ApplyAEAug(A.CLAHE, p=cfg.CLAHE))
        if cfg.CHROMATIC_ABERRATION:
            augmentations.append(
                ChromaticAberration(strength=cfg.CHROMATIC_ABERRATION))
        if cfg.VIGNETTE:
            augmentations.append(Vignetting(strength=cfg.VIGNETTE))
        if cfg.MOTION_BLUR:
            augmentations.append(
                ApplyAEAug(A.MotionBlur,
                           blur_limit=cfg.KERNEL_SIZE,
                           p=cfg.MOTION_BLUR))
        if cfg.GAUSSIAN_BLUR:
            augmentations.append(
                ApplyAEAug(A.GaussianBlur,
                           blur_limit=cfg.KERNEL_SIZE,
                           p=cfg.GAUSSIAN_BLUR))
        if cfg.NOISE:
            augmentations.append(
                AddNoise(
                    types=cfg.NOISE_TYPES,
                    amount=cfg.NOISE,
                    random_types=cfg.RANDOM_TYPES,
                ))
        if cfg.CAM_NOISE:
            augmentations.append(
                ApplyAEAug(
                    A.ISONoise,
                    color_shift=cfg.CAM_NOISE_SHIFT,
                    intensity=cfg.CAM_NOISE,
                    p=cfg.CAM_NOISE[1],
                ))
        if cfg.MOSAIC:
            augmentations.append(Mosaic())
        if cfg.SHARPEN:
            augmentations.append(
                ApplyAEAug(A.IAASharpen, alpha=cfg.SHARPEN_RANGE,
                           p=cfg.SHARPEN))
        if cfg.DENOISE:
            augmentations.append(Denoise())
        if cfg.FDA:
            augmentations.append(
                ApplyAEAug(A.FDA, reference_images=cfg.REFERENCE, p=cfg.FDA))
        if cfg.HISTOGRAM:
            augmentations.append(
                ApplyAEAug(
                    A.HistogramMatching,
                    reference_images=cfg.REFERENCE,
                    blend_ratio=cfg.HISTOGRAM,
                    p=cfg.HISTOGRAM[1],
                ))
        if cfg.JPEG:
            augmentations.append(ApplyAEAug(A.JpegCompression, p=cfg.JPEG))
        if cfg.CUTOUT:
            augmentations.append(
                Cutout(holes=cfg.CUTOUT_HOLES,
                       size=cfg.CUTOUT_SIZES,
                       p=cfg.CUTOUT))
        mapper = DatasetMapper(
            is_train=True,
            use_instance_mask=cfg.MODEL.MASK_ON,
            instance_mask_format="bitmask",
            augmentations=augmentations,
            image_format="BGR",
        )
        return build_detection_train_loader(
            cfg, mapper=mapper, num_workers=cfg.DATALOADER.NUM_WORKERS)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DatasetMapper(
            is_train=False,
            use_instance_mask=cfg.MODEL.MASK_ON,
            instance_mask_format="bitmask",
            augmentations=utils.build_augmentation(cfg, is_train=False),
            image_format="BGR",
        )
        return build_detection_test_loader(
            dataset=DatasetCatalog.get(dataset_name),
            mapper=mapper,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
        )


def get_justin_dicts(directory: str):
    img_paths = sorted(
        glob.glob(os.path.join(directory, "*/*.png"), recursive=True))
    info_paths = sorted(
        glob.glob(os.path.join(directory, "*/*.yaml"), recursive=True))

    data_dicts = []
    for index, (img_path, info_path) in enumerate(zip(img_paths, info_paths)):
        with open(info_path, "r") as f:
            img_info = yaml.safe_load(f)

        record = {
            "file_name": img_path,
            "image_id": index,
            "height": img_info["meta_data"]["original_height"],
            "width": img_info["meta_data"]["original_width"],
        }

        labels = img_info["labels"]
        objects = []
        for label in labels:
            bbox = label["bbox"]
            obj = {
                "bbox": [
                    round(bbox["minx"] * record["width"]),
                    round(bbox["miny"] * record["height"]),
                    round(bbox["maxx"] * record["width"]),
                    round(bbox["maxy"] * record["height"]),
                ],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
            }
            objects.append(obj)
        record["annotations"] = objects
        data_dicts.append(record)
    return data_dicts


def save_dicts_as_csv(data_dicts, path=""):
    with open(os.path.join(path, "data.csv") if path else "data.csv",
              "w",
              newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        for d in data_dicts:
            for obj in d["annotations"]:
                csvwriter.writerow([
                    d["file_name"],
                    int(obj["bbox"][0]),
                    int(obj["bbox"][1]),
                    int(obj["bbox"][2]),
                    int(obj["bbox"][3]),
                    "case",
                ])


def load_data(data_root, splits=("train", "val", "test", "misc", "justin")):
    coco = {
        0: "__background__",
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        12: "stop sign",
        13: "parking meter",
        14: "bench",
        15: "bird",
        16: "cat",
        17: "dog",
        18: "horse",
        19: "sheep",
        20: "cow",
        21: "elephant",
        22: "bear",
        23: "zebra",
        24: "giraffe",
        25: "backpack",
        26: "umbrella",
        27: "handbag",
        28: "tie",
        29: "suitcase",
        30: "frisbee",
        31: "skis",
        32: "snowboard",
        33: "sports ball",
        34: "kite",
        35: "baseball bat",
        36: "baseball glove",
        37: "skateboard",
        38: "surfboard",
        39: "tennis racket",
        40: "bottle",
        41: "wine glass",
        42: "cup",
        43: "fork",
        44: "knife",
        45: "spoon",
        46: "bowl",
        47: "banana",
        48: "apple",
        49: "sandwich",
        50: "orange",
        51: "broccoli",
        52: "carrot",
        53: "hot dog",
        54: "pizza",
        55: "donut",
        56: "cake",
        57: "chair",
        58: "couch",
        59: "potted plant",
        60: "bed",
        61: "dining table",
        62: "toilet",
        63: "tv",
        64: "laptop",
        65: "mouse",
        66: "remote",
        67: "keyboard",
        68: "cell phone",
        69: "microwave",
        70: "oven",
        71: "toaster",
        72: "sink",
        73: "refrigerator",
        74: "book",
        75: "clock",
        76: "vase",
        77: "scissors",
        78: "teddy bear",
        79: "hair drier",
        80: "toothbrush",
    }

    for split in splits:
        DatasetCatalog.register(
            f"justin_{split}",
            lambda split=split: get_justin_dicts(os.path.join(data_root, split)
                                                ),
        )
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
        metadata = MetadataCatalog.get("justin_test")

        for d in data_dicts:
            img = cv2.imread(d["file_name"])
            outputs = predictor(img[:, :, ::-1])
            v = Visualizer(img[:, :, ::-1], metadata=metadata)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow(str(d["image_id"]), out.get_image()[:, :, ::-1])
            cv2.waitKey(0)
    elif image_path:
        img = cv2.imread(image_path)
        outputs = predictor(img[:, :, ::-1])
        v = Visualizer(img[:, :, ::-1])
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow(image_path, out.get_image()[:, :, ::-1])
        cv2.waitKey(0)


def visualize_data(cfg, data_loader):
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    for batch in data_loader:
        for per_image in batch:
            img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
            img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)

            visualizer = Visualizer(img, metadata=metadata, scale=1.0)
            target_fields = per_image["instances"].get_fields()
            labels = [
                metadata.thing_classes[i] for i in target_fields["gt_classes"]
            ]
            vis = visualizer.overlay_instances(
                labels=labels,
                boxes=target_fields.get("gt_boxes", None),
                masks=target_fields.get("gt_masks", None),
                keypoints=target_fields.get("gt_keypoints", None),
            )
            cv2.imshow(str(per_image["image_id"]), vis.get_image()[:, :, ::-1])
            cv2.waitKey()


def set_all_seeds(seed: int):
    seed = int(seed)
    imgaug_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg):
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    return model


def path_to_checkpoint(cfg, max_ap: float, it_at_max_ap: int) -> str:
    dataset = cfg.DATASETS.TRAIN[0] if len(
        cfg.DATASETS.TRAIN) == 1 else '_'.join(cfg.DATASETS.TRAIN)
    path = os.path.join(
        cfg.OUTPUT_DIR,
        f"{cfg.BASE_CONFIG.split('/')[-1].strip('.yaml')}_{dataset}_mAP{max_ap:.2f}@it{it_at_max_ap}.pth"
    )
    return path


def extract_ap(chkpt: str) -> float:
    return float(chkpt.split("mAP")[1].split("@")[0])


def is_best_checkpoint(cfg, chkpt: str) -> bool:
    model = cfg.BASE_CONFIG.split('/')[-1].strip('.yaml')
    dataset = cfg.DATASETS.TRAIN[0] if len(
        cfg.DATASETS.TRAIN) == 1 else '_'.join(cfg.DATASETS.TRAIN)

    # Find current maximum mAP
    max_ap = 0
    for file_name in os.listdir(cfg.OUTPUT_DIR):
        if file_name.lower().endswith("pth"):
            if model in file_name and dataset in file_name:
                ap = extract_ap(file_name)
                max_ap = ap if ap > max_ap else max_ap

    # Compare to new mAP
    return extract_ap(chkpt) > max_ap


def find_best_checkpoint(cfg) -> str:
    model = cfg.BASE_CONFIG.split('/')[-1].strip('.yaml')
    dataset = cfg.DATASETS.TRAIN[0] if len(
        cfg.DATASETS.TRAIN) == 1 else '_'.join(cfg.DATASETS.TRAIN)

    max_ap = 0
    best_weights = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    for file_name in os.listdir(cfg.OUTPUT_DIR):
        if file_name.lower().endswith("pth"):
            if model in file_name and dataset in file_name:
                ap = extract_ap(file_name)
                if ap > max_ap:
                    best_weights = file_name

    logging.getLogger(name=__file__).info(f"Using {best_weights}")
    return os.path.join(cfg.OUTPUT_DIR, best_weights)


def train_eval(cfg, resume=False):
    try:
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume)
        trainer.train()
    except:
        del trainer
        torch.cuda.empty_cache()
        raise

    ap_it = np.array(trainer.storage.history(name="bbox/AP").values())
    ap = ap_it[:, 0]
    it = ap_it[:, 1]

    # Store best checkpoint if better than previous
    it_at_max_ap = it[ap.argmax()].astype(int)
    all_checkpoints = trainer.checkpointer.get_all_checkpoint_files()
    for chkpt in all_checkpoints:
        if chkpt.split('_')[-1].strip(".pth").lstrip('0') == str(it_at_max_ap):
            path = path_to_checkpoint(cfg, ap.max(), it_at_max_ap)
            if is_best_checkpoint(cfg, path):
                logging.getLogger(
                    name=__file__).info(f"Storing best model at {path}")
                torch.save(torch.load(chkpt), path)

    del trainer
    torch.cuda.empty_cache()
    return ap, it


def evaluate(cfg, model=None):
    if model is None:
        cfg.MODEL.WEIGHTS = find_best_checkpoint(cfg)
        model = get_model(cfg).eval()
    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(
        cfg, cfg.DATASETS.TEST[0], num_workers=cfg.DATALOADER.NUM_WORKERS)
    return inference_on_dataset(model, val_loader, evaluator)


def predict(cfg, dataset="", image_path=""):
    cfg.MODEL.WEIGHTS = find_best_checkpoint(cfg)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.75
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    if dataset:
        visualize_results(predictor, dataset=dataset)
    elif image_path:
        visualize_results(predictor, image_path=image_path)
    else:
        visualize_results(predictor, dataset=cfg.DATASETS.TEST[0])


def load_datasets(train_root, eval_root):
    load_data(eval_root, splits=["val", "test", "train"])

    names = list()
    for dirpath, dirnames, filenames in os.walk(train_root):
        for name in dirnames:
            names.append(name)
            path = os.path.join(train_root, name, "coco_data")
            if not os.path.exists(path):
                path = os.path.join(train_root, name)
            register_coco_instances(name, {},
                                    os.path.join(path, "coco_annotations.json"),
                                    path)
        break
    names.extend(["justin_val", "justin_test", "justin_train"])
    return names


def compute_data_statistics(loader):
    box_color = []
    box_area = []
    brightness = []
    num_images = 0
    for batch in loader:
        for per_image in batch:
            img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
            img = utils.convert_image_to_rgb(img, "BGR")

            gray = Image.fromarray(img).convert("L")
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

    print(
        "Mean value of bbox center pixels:",
        np.round(np.mean(box_color, axis=0)).astype(np.uint8),
        np.mean(box_color, axis=0) / 255.0,
    )
    print("Std:", np.std(box_color, axis=0).astype(np.uint8))

    print("Box Area")
    print(
        "mean:",
        np.mean(box_area),
        "median:",
        np.median(box_area),
        "min/max:",
        np.min(box_area),
        np.max(box_area),
    )
    print("histogram:")
    print(np.histogram(box_area)[0])

    print("Brightness")
    print(
        "mean:",
        np.mean(brightness),
        "median:",
        np.median(brightness),
        "min/max:",
        np.min(brightness),
        np.max(brightness),
    )
    print("histogram:")
    print(np.histogram(brightness)[0])


def load_and_apply_cfg_values(cfg,
                              output_dir,
                              results_name="skopt_results.pkl"):
    res = skopt.load(os.path.join(output_dir, results_name))
    augmentation_values = dict()
    for k, v in zip(get_param_names(get_space()), res.x):
        augmentation_values[k] = v
    set_cfg_values(cfg, values=augmentation_values)


def build_config(
    train_datasets,
    base_config,
    output_dir,
    batch_size: int = 4,
    epochs=2.0,
    weights: str = "coco",
):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(base_config))
    cfg.BASE_CONFIG = base_config

    cfg.DATASETS.TRAIN = train_datasets
    cfg.DATASETS.TEST = ("justin_test",)

    # cfg.DATALOADER.NUM_WORKERS = multiprocessing.cpu_count() - 2

    cfg.INPUT.MIN_SIZE_TRAIN = (800,)  # (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.CROP.ENABLED = False
    cfg.INPUT.CROP.SIZE = [0.9, 0.9]

    cfg.SOLVER.IMS_PER_BATCH = int(batch_size)

    cfg.NUM_BATCHES = 0
    for ds in cfg.DATASETS.TRAIN:
        cfg.NUM_BATCHES += len(DatasetCatalog.get(ds))
    cfg.NUM_BATCHES = cfg.NUM_BATCHES // cfg.SOLVER.IMS_PER_BATCH
    cfg.EPOCHS = epochs

    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = int(cfg.EPOCHS * cfg.NUM_BATCHES)
    cfg.SOLVER.CHECKPOINT_PERIOD = 100
    cfg.SOLVER.WEIGHT_DECAY = 0.0
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.NESTEROV = False
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    # Linear warm up to base learning rate within 20% of all iterations
    cfg.SOLVER.WARMUP_ITERS = 1  # int(cfg.SOLVER.MAX_ITER * .2)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / float(cfg.SOLVER.WARMUP_ITERS)
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = False
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.001
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    cfg.SOLVER.GAMMA = 1.0  # .99
    # [int(fraction * cfg.SOLVER.MAX_ITER) for fraction in np.arange(1 - cfg.SOLVER.GAMMA, 1, 1 - cfg.SOLVER.GAMMA)]
    cfg.SOLVER.STEPS = (1,)

    if weights.lower() == "coco":
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_config)
    elif weights.lower() == "none":
        cfg.MODEL.WEIGHTS = ""
    elif weights.lower() == "imagenet":
        pass
    else:
        raise ValueError(
            f"Weights must be one of 'coco', 'imagenet' or 'none', not {weights}"
        )

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.5

    cfg.TEST.EVAL_PERIOD = cfg.SOLVER.CHECKPOINT_PERIOD
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1

    cfg.OPTIMIZER = "ADAM"
    cfg.LOGGER = True

    cfg.ROTATE = False
    cfg.PHOTOMETRIC = []
    cfg.PHOTOMETRIC_TYPES = ["brightness", "contrast", "saturation", "lighting"]
    cfg.INTERP = Image.BILINEAR
    cfg.NOISE = []
    cfg.NOISE_TYPES = ["poisson", "gaussian", "speckle", "s&p"]
    cfg.CAM_NOISE = []
    cfg.CAM_NOISE_SHIFT = [0.01, 0.1]
    cfg.RANDOM_TYPES = True
    cfg.DENOISE = False
    cfg.CUTOUT = 0.0
    cfg.CUTOUT_SIZES = [100, 100, None, None]
    cfg.CUTOUT_HOLES = [5, 1]
    cfg.MOTION_BLUR = 0.0
    cfg.GAUSSIAN_BLUR = 0.0
    cfg.KERNEL_SIZE = 11
    cfg.FLIP = "horizontal"
    cfg.INVERT = 0.0
    cfg.GRAYSCALE = 0.0
    cfg.CHANNEL_DROPOUT = 0.0
    cfg.HISTOGRAM = []
    cfg.FDA = 0.0
    cfg.REFERENCE = [
        sample["file_name"] for sample in random.sample(
            DatasetCatalog.get(cfg.DATASETS.TEST[0]), 100)
    ]
    cfg.SHARPEN = 0.0
    cfg.SHARPEN_RANGE = [0.2, 0.5]
    cfg.CLAHE = 0.0
    cfg.MOSAIC = False
    cfg.JPEG = False
    cfg.VIGNETTE = 0.0  # (0., 0.8)
    cfg.CHROMATIC_ABERRATION = 0.0  # (0., 0.5)
    cfg.AUTO_AUGMENT = False
    # gaussian blur, contrast, brightness, color and sharpness filters, cutout

    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def main(seed: int = 42) -> None:
    start = time.time()
    set_all_seeds(seed)

    default_values = {
        "learning_rate": 0.0001,
        "batch_size": 4,
        "epochs": 2.0,
        "reduce_lr": 0.0,
        "weight_decay": 0.0,
        "warmup_fraction": 0.0,
        "clip_gradients": False,
        "clip_value": 0.001,
        "clip_type": "value",
        "norm_type": 2.0,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        default=["case_color"],
        type=str,
        nargs="+",
        help="List of datasets used for training.",
    )
    parser.add_argument("--path_prefix",
                        default="/home/matthias/Data/Ubuntu/data",
                        type=str)
    parser.add_argument("--train_dir", default="datasets/case", type=str)
    parser.add_argument("--val_dir", default="datasets/justin", type=str)
    parser.add_argument("--out_dir", default="justin_training", type=str)
    parser.add_argument("--model", default="retinanet", type=str)
    parser.add_argument("--weights", default="coco", type=str)
    parser.add_argument(
        "--batch_size",
        default=default_values["batch_size"],
        type=int,
        help="Batch size used during training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=default_values["learning_rate"],
        type=float,
        help="Learning rate used during training.",
    )
    parser.add_argument(
        "--reduce_lr",
        default=default_values["reduce_lr"],
        type=float,
        help="Multiplied by learning rate each iteration.",
    )
    parser.add_argument(
        "--weight_decay",
        default=default_values["weight_decay"],
        type=float,
        help="Weight decay used during training.",
    )
    parser.add_argument(
        "--warmup_fraction",
        default=default_values["warmup_fraction"],
        type=float,
        help="Learning rate warmup in fraction of total steps.",
    )
    parser.add_argument(
        "--clip_gradients",
        action="store_true",
        help="Gradients are clipped at 'clip_value'.",
    )
    parser.add_argument(
        "--clip_value",
        default=default_values["clip_value"],
        type=float,
        help="Threshold for gradient clipping.",
    )
    parser.add_argument(
        "--clip_type",
        default="value",
        type=str,
        help="Gradients can be clipped at 'clip_value' value or their norm",
    )
    parser.add_argument(
        "--norm_type",
        default=default_values["norm_type"],
        type=float,
        help="Norm type for 'norm' gradient clipping.",
    )
    parser.add_argument(
        "--epochs",
        default=default_values["epochs"],
        type=float,
        help="(Fraction of) epochs to train.",
    )
    parser.add_argument("--visualize",
                        action="store_true",
                        help="Visualize training data.")
    parser.add_argument("--predict",
                        action="store_true",
                        help="Visualize predictions.")
    args = parser.parse_args()

    values = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "reduce_lr": args.reduce_lr,
        "weight_decay": args.weight_decay,
        "warmup_fraction": args.warmup_fraction,
        "clip_gradients": args.clip_gradients,
        "clip_value": args.clip_value,
        "clip_type": args.clip_type,
        "norm_type": args.norm_type,
        "epochs": args.epochs
    }

    if args.model == "retinanet":
        base_config = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    elif args.model == "faster_rcnn":
        base_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    elif args.model == "mask_rcnn":
        base_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        values = {
            "learning_rate": 4.28226e-05,
            "weight_decay": 1.94036e-13,
            "warmup_fraction": 0.2,
            "cam_noise": 0.103469,
            "hist": 0.56282,
            "invert": 1.0,
            "cutout": 0.887532,
            "cutout_sizes": 100,
            "clahe": 0.923281,
            "channel_dropout": 0.962356,
            "vignette": True,
        }
    else:
        base_config = args.model

    output_dir = os.path.join(args.path_prefix, args.out_dir)
    train_root = os.path.join(args.path_prefix, args.train_dir)
    val_root = os.path.join(args.path_prefix, args.val_dir)
    dataset_names = load_datasets(train_root, val_root)
    train_datasets = parse_data(args.data, dataset_names)

    if args.data in ["best", "all"]:
        log_file = f"{args.data}.log"
    else:
        log_file = f"{train_datasets[0] if len(train_datasets) == 1 else '_'.join(train_datasets)}.log"
    logger = setup_logger(output=os.path.join(output_dir, log_file),
                          name=__file__)

    cfg = build_config(
        train_datasets,
        base_config,
        output_dir,
        args.batch_size,
        args.epochs,
        args.weights,
    )
    logger.info(values)
    set_cfg_values(cfg, values)
    # load_and_apply_cfg_values(cfg, output_dir)

    if args.visualize:
        trainer = Trainer(cfg)
        loader = trainer.build_train_loader(cfg)
        # compute_data_statistics(loader)
        visualize_data(cfg, loader)
    elif args.predict:
        predict(cfg)
    else:
        results = list()
        for s in np.arange(5):
            set_all_seeds(s)
            ap, it = train_eval(cfg)
            result = np.vstack([ap, it]).T
            result = result[result[:, 0].argmax()].tolist()
            result.append(s)
            results.append(result)
        results = np.array(results)

        ap = results[:, 0]
        table = tabulate.tabulate(results[ap.argsort()[::-1]],
                                  headers=["AP", "iter", "seed"])

        logger.info(table)
        logger.info(f"Mean AP: {ap.mean()}")
        logger.info(
            f"95% conf. interv.: {t.interval(0.95, len(ap) - 1, loc=np.mean(ap), scale=sem(ap))}"
        )
        logger.info(
            f"99% conf. interv.: {t.interval(0.99, len(ap) - 1, loc=np.mean(ap), scale=sem(ap))}"
        )
        logger.info(f"Copy: {np.sort(ap)[::-1]}")

        for k, v in get_results_dict().items():
            # The probability to obtain the current result by chance (assuming the same data generating process) is ...
            logger.info(f"Statistics for {cfg.DATASETS.TRAIN[0]} vs {k}:")
            logger.info(ttest_ind(ap, v, equal_var=False))
            logger.info(
                f"99% conf. interv. of {k}: {t.interval(0.99, len(v) - 1, loc=np.mean(v), scale=sem(v))}"
            )
            # ... or less.

        logger.info(f"Runtime: {(time.time() - start) / 60}")


if __name__ == "__main__":
    main(seed=42)
