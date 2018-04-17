"""Modulo to process entropy."""
from skimage.filters import rank
from skimage.morphology import disk


class EntropyProcessor:

    @staticmethod
    def entropy(image, channel=0):
        """Calculate the image entropy.

        Args:
            image(ndarray): image representation as array.

        """
        return rank.entropy(image[:,:, channel], disk(5))
