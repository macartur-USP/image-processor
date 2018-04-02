"""Module to process images."""
from numpy import hstack,reshape
from skimage.transform import resize
from skimage.io import imread
from PIL import Image
import numpy as np

from .color_processor import ColorProcessor
from .texture_processor import TextureProcessor


class ImageProcessor:
    """Class to manupulate image for pre and pos process."""

    def __init__(self):
        """Intance the image processor."""
        self.path = None
        self.image = None
        self.features = []
        self.resize_shape = (320, 240, 3)

    def open(self, image_path=None):
        """Open an image to be processed.

        Args:
            image_path: path of an image
        """
        try:
            self.image = imread(image_path)
            self.path = image_path
        except (FileNotFoundError, AttributeError):
            self.image = None
            print(f'image {image_path} not found !!!')

    def pre_process(self):
        """Preprocess the image setted.

        This method will:
        - convert to hsv color space
        - resize the image to 320 x 240 pixels
        """
        resized = Image.fromarray(self.image).resize(self.resize_shape[:2])
        self.image = np.array(resized.convert('RGB'))

    def split_in_blocks(self, new_size=15):
        """Split current image into 15x15 pixels blocks."""
        image_blocks = []
        for line in range(0, self.resize_shape[1], new_size):
            for column in range(0, self.resize_shape[0], new_size):
                image_blocks.append(self.image[line:line+new_size,
                                               column:column+new_size])
        return image_blocks

    def extract_features(self, image):
        """Extract all features from image setted.

        This method will serialize the image and get all property for each pixel

        """
        img_hsv = ColorProcessor.convert_rgb_to_hsv(image)
        feature_hsv = img_hsv.reshape((225,3))
        img_gray = ColorProcessor.convert_rgb_to_grey(image)
        feature_gray = img_gray.reshape((225,1))
        feature_entropy = hstack([TextureProcessor.entropy(image,channel).reshape((225,1))  for channel in range(3)])
        lbp = TextureProcessor.local_binary_pattern(image)
        feature_lbp = lbp.reshape((225,1))
        feature_gabor = TextureProcessor.get_gabor_properties(image)
        return hstack([feature_hsv,feature_gray, feature_entropy, feature_lbp, feature_gabor])

    def get_fill_image(self, image):
        gray = ColorProcessor.convert_rgb_to_grey(image)
        zeros = np.zeros(gray.shape)
        for l in range(gray.shape[0]):
           for x in range(gray.shape[1]):
               if gray[l][x] > 0:
                   zeros[l][x] = 1
        return zeros
