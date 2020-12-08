#################################################
# Transforms applied to images in this project. #
#################################################

import numpy as np
import cv2
import torch
from PIL import Image


class ToNormalized(object):
    """Perform channel-wise normalization on an RGB image
    """
    def __call__(self, pic):
        """
        :param pic: (numpy.ndarray): Image to be normalized

        :return: normalized_image: (numpy.ndarray): Mean-normalized image
        """
        normalized_image = torch.div(torch.sub(pic, self.mean), self.std)
        return normalized_image

    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean).reshape(3, 1, 1)
        self.std = torch.Tensor(std).reshape(3, 1, 1)


class ToClosed(object):
    """
    Perform morphological closing on an RGB image with a 2,2 kernel
    """
    def __call__(self, img):
        closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, self.kernel)
        closed_img = Image.fromarray(np.uint8(closed_img))
        return closed_img

    def __init__(self):
        self.kernel = np.ones([2, 2], np.uint8)
