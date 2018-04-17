"""Module to process images."""
from numpy import hstack,reshape
from skimage.transform import resize
from skimage.io import imread, imsave
from PIL import Image
import numpy as np

from .color_processor import ColorProcessor
from .texture_processor import TextureProcessor
from .entropy_processor import EntropyProcessor


class ImageProcessor:
    """Class to manupulate image for pre and pos process."""

    def __init__(self):
        """Intance the image processor."""
        self.path = None
        self.image = None

    def open(self, image_path=None):
        """Open an image to be processed.

        Args:
            image_path: path of an image
        """
        try:
            self.image = imread(image_path, mode='RGB')
            self.path = image_path
        except (FileNotFoundError, AttributeError):
            self.image = None
            print(f'image {image_path} not found !!!')

    def split_in_blocks(self, new_size=15):
        """Split current image into 15x15 pixels blocks."""
        image_blocks = []
        for line in range(0, self.image.shape[1], new_size):
            for column in range(0, self.image.shape[0], new_size):
                image_blocks.append(self.image[line:line+new_size,
                                               column:column+new_size])
        return image_blocks

    def extract_features(self, image):
        """Extract all features from image setted.

        This method will serialize the image and get all property for each pixel

        """
        img_hsv = ColorProcessor.convert_rgb_to_hsv(image)

        shape_format = img_hsv.shape[0]*img_hsv.shape[1]

        feature_hsv = img_hsv.reshape((shape_format,3))
        img_gray = ColorProcessor.convert_rgb_to_grey(image)
        feature_gray = img_gray.reshape((shape_format,1))
        feature_entropy = hstack([EntropyProcessor.entropy(image,channel).reshape((shape_format,1))  for channel in range(3)])
        lbp = TextureProcessor.local_binary_pattern(image)
        feature_lbp = lbp.reshape((shape_format,1))
        feature_gabor = TextureProcessor.get_gabor_properties(image)
        return hstack([feature_hsv,feature_gray, feature_entropy, feature_lbp, feature_gabor])

    def save(self, image, path):
        """Save an array as image.

        Args:
            image(ndarray): Array representing an image
            path(string): image path

        """
        imsave(path, image)
