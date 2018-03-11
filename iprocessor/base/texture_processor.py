"""Modulo to process texture."""
from numpy import pi, hstack, ravel
from skimage.morphology import disk
from skimage.filters import gabor, rank
from skimage.feature import greycomatrix, local_binary_pattern, greycoprops
from .color_processor import ColorProcessor


class TextureProcessor:
    """Class to process Texture from image."""

    @staticmethod
    def get_texture_features(image):
        """Create a texture features.

        This method will create texture features using:

            - local_binary_pattern:
            - entropy
            - coocorrence matrix
            - gabor filter
        """
        entropy = TextureProcessor.entropy(image)
        lbp = TextureProcessor.local_binary_pattern(image)
        features = TextureProcessor.get_coocorrence_matrix_properties(image)
        gabor = TextureProcessor.get_gabor_properties(image)

        return hstack([features, entropy, lbp, gabor])

    @staticmethod
    def get_gabor_properties(image):
        """ Get all properties from gabor filter.

        - gabor filter with the parameters:
            - frequency: 1, 2**(1/5), 2
            - theta = [0, 30, 45, 60, 90]

        """
        frequencies = [1, 2**(1/5), 2]
        theta = [0, pi/6, pi/4, pi/3, pi/2, pi]
        gabor = []
        for freq in frequencies:
            for t in  theta:
                gabor.extend(TextureProcessor.gabor_filter(image,freq, t))
        return hstack(gabor)

    @staticmethod
    def get_coocorrence_matrix_properties(image):
        """Create features based on image gray level.

        The properties extracted are:
            - contrast
            - dissimilarity
            - homogeneity
            - ASM (Angular Second Moment)
            - energy
            - correlation
        """
        image_grey = ColorProcessor.convert_rgb_to_grey(image)
        props = ['contrast', 'homogeneity','dissimilarity',
                 'ASM', 'correlation', 'energy']
        matrix = TextureProcessor.coocorrence_matrix(image_grey)
        return hstack([greycoprops(matrix, prop) for prop in props])

    @staticmethod
    def coocorrence_matrix(image, levels=256):
        """Calculate the coocorrence matrix of a image.

        distance = 1,2,3
        This process will use:
            - distance = [1,2,3,4]
            - orientation = [0, 30, 45, 60, 90, 180] in radian
        """
        image_grey = ColorProcessor.convert_rgb_to_grey(image)
        distances = [x for x in range(1,image.shape[0]+1) ]
        angles = [0, pi/6, pi/4, pi/3, pi/2, pi]
        return greycomatrix(image_grey, distances, angles, levels=levels)

    @staticmethod
    def local_binary_pattern(image):
        """Execute the LBP to extract texture features.

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
        return rank.entropy(image_grey, disk(5))

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
