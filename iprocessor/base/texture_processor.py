"""Modulo to process texture."""
from numpy import pi
from skimage.filters import gabor_kernel
from skimage.feature import greycomatrix, local_binary_pattern
from .color_processor import ColorProcessor


class TextureProcessor:
    """Class to process Texture from image."""

    @staticmethod
    def coocorrence_matrix(image, levels=256):
        """Calculate the coocorrence matrix of a image.

        distance = 1
        This process will use 6 orientation: [0, 30, 45, 60, 90, 180]
        """
        image_grey = ColorProcessor.convert_rgb_to_grey(image)
        distances = [1]
        angles = [0, pi/6, pi/4, pi/3, pi/2, pi]
        return greycomatrix(image_grey, distances, angles, levels=levels)

    @staticmethod
    def local_binary_pattern(image):
        """Execute the LBP to extract texture feature."""
        image_grey = ColorProcessor.convert_rgb_to_grey(image)
        return local_binary_pattern(image, P=8, R=1.0, method='default')

