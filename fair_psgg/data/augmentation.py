from typing import Union
from PIL import Image
import torch
import torchvision.transforms as tvtfm
import torchvision.transforms.functional as F


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def bbox_hflip(boxes: torch.Tensor, img_width: int):
    middle = img_width // 2
    x1 = 2 * middle - boxes[..., 2]
    x2 = 2 * middle - boxes[..., 0]
    output = torch.stack((x1, boxes[..., 1], x2, boxes[..., 3]), dim=-1)
    return output


def get_img_shape(img: Union[Image.Image, torch.Tensor]):
    """Returns the shape of the image as `(height, width)`"""
    if isinstance(img, Image.Image):
        return img.height, img.width
    elif isinstance(img, torch.Tensor):
        return img.shape[1], img.shape[2]
    raise ValueError(f"Unsupported img type: {type(img)}")


class BaseAugmentation:
    def __call__(self, img: torch.Tensor, seg: torch.Tensor, bboxes: torch.Tensor):
        raise NotImplementedError()

    def _get_repr_args(self):
        return "?"

    def __repr__(self):
        return f"{self.__class__.__name__}({self._get_repr_args()})"


class SquarePad(BaseAugmentation):
    def __call__(self, img: torch.Tensor, seg: torch.Tensor, bboxes: torch.Tensor):
        img_shape = get_img_shape(img)
        size = max(*img_shape)
        # padding on: left, top, right, bottom
        padding = [0, 0, size - img_shape[1], size - img_shape[0]]
        img = F.pad(img, padding)
        seg = F.pad(seg, padding)
        return img, seg, bboxes

    def _get_repr_args(self):
        return ""


class Resize(BaseAugmentation):
    def __init__(self, size):
        self.img_resize = tvtfm.Resize(size, antialias=True)

    def __call__(self, img: torch.Tensor, seg: torch.Tensor, bboxes: torch.Tensor):
        scale = self.img_resize.size / min(img.shape[1:])
        return self.img_resize(img), self.img_resize(seg), bboxes * scale

    def _get_repr_args(self):
        return f"size={self.img_resize.size}"


class ToTensor(BaseAugmentation):
    def __call__(self, img: torch.Tensor, seg: torch.Tensor, bboxes: torch.Tensor):
        return F.to_tensor(img), seg, bboxes

    def _get_repr_args(self):
        return ""


class RandomHorizontalFlip(BaseAugmentation):
    """Flips the image with a probability of `prob`"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img: torch.Tensor, seg: torch.Tensor, bboxes: torch.Tensor):
        if torch.rand(()) < self.prob:
            img = F.hflip(img)
            seg = F.hflip(seg)
            img_width = get_img_shape(img)[1]
            bboxes = bbox_hflip(bboxes, img_width)
        return img, seg, bboxes

    def _get_repr_args(self):
        return f"prob={self.prob}"


class BoxJitter(BaseAugmentation):
    def __init__(self, max_relative_offset: float = 0.1):
        assert 0 <= max_relative_offset < 1
        self.max_relative_offset = max_relative_offset

    def __call__(self, img: torch.Tensor, seg: torch.Tensor, bboxes: torch.Tensor):
        offsets = (2 * torch.rand(bboxes.shape) - 1) * self.max_relative_offset
        widths_heights = (bboxes - bboxes[:, (2, 3, 0, 1)]).abs()
        bboxes = bboxes + offsets * widths_heights
        return img, seg, bboxes

    def _get_repr_args(self):
        return f"max_relative_offset={self.max_relative_offset}"


class BoxClamp(BaseAugmentation):
    """Clamps boxes to the image size (such that the boxes are always inside the image).
    Note that if you pad the image first, the boxes can also end up in the padding region.
    """

    def __call__(self, img: torch.Tensor, seg: torch.Tensor, bboxes: torch.Tensor):
        h, w = get_img_shape(img)
        return (
            img,
            seg,
            bboxes.clamp(min=torch.tensor(0), max=torch.tensor((w, h, w, h))),
        )

    def _get_repr_args(self):
        return ""


class Identity(BaseAugmentation):
    """Redirects the input without touching it"""

    def __call__(self, img: torch.Tensor, seg: torch.Tensor, bboxes: torch.Tensor):
        return img, seg, bboxes

    def _get_repr_args(self):
        return ""


class Prob(BaseAugmentation):
    """Performs the underlying augmentation `tfm` with a probability of `prob`"""

    def __init__(self, tfm, prob: float = 0.5):
        assert 0 <= prob <= 1, prob
        self.prob = prob
        self.tfm = tfm

    def __call__(self, img: torch.Tensor, seg: torch.Tensor, bboxes: torch.Tensor):
        if torch.rand(()) < self.prob:
            return self.tfm(img, seg, bboxes)
        return img, seg, bboxes

    def _get_repr_args(self) -> str:
        return f"tfm={self.tfm}, prob={self.prob}"


class OnImage:
    def __init__(self, tfm):
        self.tfm = tfm

    def __call__(self, img: torch.Tensor, seg: torch.Tensor, bboxes: torch.Tensor):
        return self.tfm(img), seg, bboxes


def denormalize_imagenet(img: torch.Tensor):
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    return img * std + mean


class Normalize(BaseAugmentation):
    def __init__(self, mean=None, std=None):
        """Default values are ImageNet statistics"""
        if mean is None:
            mean = IMAGENET_MEAN
        if std is None:
            std = IMAGENET_STD
        self.tfm = OnImage(tvtfm.Normalize(mean=mean, std=std))

    def __call__(self, img: torch.Tensor, seg: torch.Tensor, bboxes: torch.Tensor):
        return self.tfm(img, seg, bboxes)

    def _get_repr_args(self):
        return f"mean={self.tfm.tfm.mean}, std={self.tfm.tfm.std}"


class ColorJitter(BaseAugmentation):
    def __init__(
        self,
        brightness: float = 0,
        contrast: float = 0,
        saturation: float = 0,
        hue: float = 0,
    ):
        self.tfm = OnImage(
            tvtfm.ColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
            ),
        )

    def __call__(self, img: torch.Tensor, seg: torch.Tensor, bboxes: torch.Tensor):
        return self.tfm(img, seg, bboxes)

    def _get_repr_args(self):
        cj = self.tfm.tfm
        return f"brightness={cj.brightness}, contrast={cj.contrast}, saturation={cj.saturation}, hue={cj.hue}"


def config_to_aug(config: Union[dict, str]):
    """Builds a data augmentation object based on a configuration.

    If the configuration is a single string, the string will be used to identify the class name.
    The augmentation object is then initialised without any arguments. This only works for augmentation classes with default arguments.

    If the configuration is a dictionary, the `type` key will be used to identify the class name.
    All other keys are forwarded to `kwargs`.
    If an augmentation class requires another augmentation class (e.g. `Prob`), you can specify the nested object with a dictionary:

    ``` json
    {
        "type": "Prob",
        "tfm": {
            "type": "BoxJitter"
        }
    }
    ```

    Just using a string will not work for nested augmentation objects.
    """
    transforms = [
        SquarePad,
        Resize,
        ToTensor,
        RandomHorizontalFlip,
        BoxJitter,
        BoxClamp,
        Identity,
        Prob,
        Normalize,
        ColorJitter,
    ]
    if isinstance(config, str):
        for tfm in transforms:
            if tfm.__name__ == config:
                return tfm()
        raise KeyError(config)

    tfm_cls = None
    for tfm in transforms:
        if tfm.__name__ == config["type"]:
            tfm_cls = tfm
            break
    assert tfm_cls is not None
    kwargs = {}
    for k, v in config.items():
        if k == "type":
            continue
        # it's a nested augmentation!
        if isinstance(v, dict) and "type" in v:
            v = config_to_aug(v)
        kwargs[k] = v
    return tfm_cls(**kwargs)


def get_standard_transforms(is_train: bool, image_size: int = 640, box_jitter=False):
    """Standard transforms that are used for every loader (for now)"""
    colour_augs = [
        tvtfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if is_train:
        colour_augs.insert(
            0, tvtfm.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        )
    return [
        ToTensor(),
        RandomHorizontalFlip() if is_train else Identity(),
        Prob(BoxJitter(), prob=0.7) if box_jitter and is_train else Identity(),
        BoxClamp(),
        SquarePad(),
        Resize(image_size),  # backbone expects 800
        OnImage(tvtfm.Compose(colour_augs)),
    ]
