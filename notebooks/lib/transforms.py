#################################################
# Transforms applied to images in this project. #
#################################################

import cv2
import numpy as np
from PIL import Image
import torch

class ToNormalized(object):
    """Perform channel-wise normalization on an RGB image."""
    
    def __init__(self, mean, std):
        """Create a normalizing transform.
        
        Args:
            mean: Channel-wise means for the images. 
            std: Channel-wise stds for the images.
        """
        self.mean = torch.Tensor(mean).reshape(3, 1, 1)
        self.std = torch.Tensor(std).reshape(3, 1, 1)
    
    def __call__(self, pic: torch.Tensor) -> torch.Tensor:
        """
        :param pic: (numpy.ndarray): Image to be normalized

        :return: normalized_image: (numpy.ndarray): Mean-normalized image
        """
        if not isinstance(pic, torch.Tensor):
            raise TypeError(f'Transform {self.__class__.__name__} expects a PyTorch tensor.')
        normalized_image = torch.div(torch.sub(pic, self.mean), self.std)
        return normalized_image

class ToClosed(object):
    """Perform morphological closing on an RGB image with a 2,2 kernel."""

    def __call__(self, img: Image.Image or np.ndarray) -> Image.Image:
        if isinstance(img, Image.Image):
            img = np.array(img)
        elif not isinstance(img, np.ndarray):
            raise TypeError(f'Transform {self.__class__.__name__} expects a PIL image or Numpy array.')
        # Convert to BGR format for compatibility with OpenCV
        cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        closed_img = cv2.morphologyEx(cv2_img, cv2.MORPH_CLOSE, self.kernel)
        # Convert back to RGB format.
        closed_img = cv2.cvtColor(closed_img, cv2.COLOR_BGR2RGB)
        # Convert OpenCV matrix to a PIL Image
        closed_img = Image.fromarray(np.uint8(closed_img))
        return closed_img

    def __init__(self):
        self.kernel = np.ones([2, 2], np.uint8)
