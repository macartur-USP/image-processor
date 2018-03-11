"""Modulo to process texture."""
import numpy
from numpy import pi
from skimage.morphology import disk
from skimage.measure import shannon_entropy
from skimage.filters import gabor
from skimage.feature import greycomatrix, local_binary_pattern
from .color_processor import ColorProcessor


class TextureProcessor:
    """Class to process Texture from image."""

    @staticmethod
    def get_texture_properties(image):
        """Create a texture features.

        This method will create texture features using:

            - coocorrence_matris:
                - entropy (shannon entropy)
                - contrast
                - dissimilarity
                - homogeneity
                - asm
                - energy
                - correlation
            - gabor filter:
                - frequency: 1, 2**(1/5), 2
                - theta = [0, 30, 45, 60, 90, 180]
            - local_binary_pattern:
                - P=8
                - R=1
                - method='method'
        """
        # skimage.feature.greycoprops
        # http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.greycoprops
        pass

    @staticmethod
    def coocorrence_matrix(image, levels=256):
        """Calculate the coocorrence matrix of a image.

        distance = 1,2,3
        This process will use:
            - distance = [1,2,3]
            - orientation = [0, 30, 45, 60, 90, 180] in radian
        """
        image_grey = ColorProcessor.convert_rgb_to_grey(image)
        distances = [1,2,3]
        angles = [0, pi/6, pi/4, pi/3, pi/2, pi]
        return greycomatrix(image_grey, distances, angles, levels=levels)

    @staticmethod
    def local_binary_pattern(image):
        """Execute the LBP to extract texture feature.

        The function LBP will use P=8 and R=1.0
        """
        image_grey = ColorProcessor.convert_rgb_to_grey(image)
        return local_binary_pattern(image, P=8, R=1.0, method='default')

    @staticmethod
    def entropy(image):
        """Calculate the image entropy.

        Args:
            image(ndarray): image representation as array.
        """
        image_grey = ColorProcessor.convert_rgb_to_grey(image)
        return shannon_entropy(image_grey)

    @staticmethod
    def gabor_filter(image, frequency=1, theta=0):
        """Execute Gabor filter to extract texture feature.
        Args:
            frequency(float): frequency to calculate gabor filter method
            theta(float): angle to calculate gabor filter method

        Returns:
            tuple: returns the real and imaginary coordinates

        """
        image_grey = ColorProcessor.convert_rgb_to_grey(image)
        return gabor(image_grey, frequency=frequency, theta=theta)
