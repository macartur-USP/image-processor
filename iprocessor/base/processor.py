"""Module to process images."""
from skimage.io import imread

from .color_processor import ColorProcessor


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
        self.image.resize(self.resize_shape, refcheck=False)
        self.image = ColorProcessor.convert_rgb_to_hsv(self.image)

    def split_in_blocks(self, new_size=15):
        """Split current image into 15x15 pixels blocks."""
        image_blocks = []
        for line in range(0, self.resize_shape[1], new_size):
            for column in range(0, self.resize_shape[0], new_size):
                image_blocks.append(self.image[line:line+new_size,
                                               column:column+new_size])
        return image_blocks

    def extract_features(self):
        """Extract all features from image setted."""
        pass
