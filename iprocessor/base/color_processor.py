"""Modulo to process color."""
from skimage import color, util, exposure
from numpy import array
from PIL import Image

class ColorProcessor:
    """Class to process color space from images."""

    @staticmethod
    def convert_rgb_to_hsv(image):
        """Convert a image to hsv color space."""
        return color.rgb2hsv(image)

    @staticmethod
    def convert_rgb_to_grey(image):
        """Convert a image to grey scale color space."""
        gray = Image.fromarray(image)
        return array(gray.convert('L'))

    @staticmethod
    def create_color_histograms(image, nbins=15):
        """Return 3 histograms from given image.

        Returns:
            histogram(Hue, saturation, value): A tuple with 3 histogram

        """
        hue_histogram = ColorProcessor.create_histogram(image, nbins, 0)
        saturation_histogram = ColorProcessor.create_histogram(image, nbins, 1)
        value_histogram = ColorProcessor.create_histogram(image, nbins, 2)

        return (hue_histogram, saturation_histogram, value_histogram)

    @staticmethod
    def create_histogram(image, nbins=15, feature=0):
        """Create a histogram color.

        Args:
            image(ndarray): array of an image
            nbins(int): number of segments
            feature(int): feature used to create the histogram of pixel
                           color(0, 1, 2, 3)
        Returns:
            histogram: tuple with the histogram x, y values.

        """
        image_features = image[0:image.shape[0], 0: image.shape[1], feature]
        return exposure.histogram(image_features, nbins=nbins)
