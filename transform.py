from albumentations.core.transforms_interface import ImageOnlyTransform
from torchvision.transforms import functional as F
import cv2
import numpy as np

from deslant import deslant


class Deslant(ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        bg_color = np.bincount(img[0, :]).argmax()
        return deslant(img, bg_color=int(bg_color)).img


class Binarize(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.3):
        super(Binarize, self).__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # threshold = params['threshold']
        # if threshold:
        #     return (img > threshold) * 255
        _, image = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return image

    def get_transform_init_args_names(self):
        return ()


class Rotate(ImageOnlyTransform):
    def __init__(self, rotate_code=cv2.ROTATE_90_CLOCKWISE, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.rotate_code = rotate_code

    def apply(self, img, **params):
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    def get_transform_init_args_names(self):
        return ("rotate_code",)


class ToTensor(ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        return F.to_tensor(img)

    def get_transform_init_args_names(self):
        return ()
