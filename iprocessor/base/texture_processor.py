"""Modulo to process texture."""
from numpy import hstack, pi
from skimage.feature import local_binary_pattern
from skimage.filters import gabor, rank
from skimage.morphology import disk
from skimage import util

from .color_processor import ColorProcessor

class TextureProcessor:
    """Class to process Texture from image."""

    @staticmethod
    def get_gabor_properties(image):
        """Get all properties from gabor filter.

        - gabor filter with the parameters:
            - frequency: 1, 2**(1/5), 2
            - theta = [0, 30, 45, 60, 90]
        """
        frequencies = [1, 2**(1/5), 2]
        angles = [0, pi/6, pi/4, pi/3, pi/2, pi]
        values = []
        for freq in frequencies:
            for angle in angles:
                real, img = TextureProcessor.gabor_filter(image, freq,
                                                            angle)
                values.append(real.reshape((225,1)))
        return hstack(values)

    @staticmethod
    def local_binary_pattern(image):
        """Execute the LBP to extract texture features.

        The function LBP will use P=8 and R=1.0
        """
        image_grey = ColorProcessor.convert_rgb_to_grey(image)
        return local_binary_pattern(image_grey, P=8, R=1.0, method='default')

    @staticmethod
    def entropy(image, channel=0):
        """Calculate the image entropy.

        Args:
            image(ndarray): image representation as array.
        """
        return rank.entropy(image[:,:, channel], disk(5))

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
