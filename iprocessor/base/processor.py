"""Module to process images."""
from skimage import color, exposure
from skimage.io import imread


class ImageProcessor:
    """Class to manupulate image for pre and pos process."""

    def __init__(self):
        """Intance the image processor."""
        self.path = None
        self.image = None
        self.image_blocks = []
        self.feactures = []
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
            # print(f'image {image_path} not found !!!')

    def pre_process(self):
        """Preprocess the image setted.

        This method will:
        - convert to hsv color space
        - resize the image to 320 x 240 pixels
        - update image_blocks with 15x15 pixels of the current image
        - extract feactures from each block
        """
        self.convert_to_hsv()
        self.resize_image()
        self.split_into_blocks()

    def resize_image(self):
        """Resize the current image."""
        self.image.resize(self.resize_shape, refcheck=False)

    def convert_to_hsv(self):
        """Convert the current image to hsv color space."""
        self.image = color.rgb2hsv(self.image[0:320, 0:240, 0:3])

    def split_into_blocks(self, new_size=15):
        """Split current image into 15x15 pixels blocks."""
        for line in range(0, self.resize_shape[1], new_size):
            for column in range(0, self.resize_shape[0], new_size):
                self.image_blocks.append(self.image[line:line+new_size,
                                         column:column+new_size])

    def extract_feactures(self):
        """Extract all feactures from image setted."""
        pass

    @staticmethod
    def create_color_histograms(image, nbins=15):
        """Return 3 histograms from given image.

        Returns:
            histogram(Hue, saturation, value): A tuple with 3 histogram

        """
        hue_histogram = ImageProcessor.create_histogram(image, nbins, 0)
        saturation_histogram = ImageProcessor.create_histogram(image, nbins, 1)
        value_histogram = ImageProcessor.create_histogram(image, nbins, 2)

        return (hue_histogram, saturation_histogram, value_histogram)

    @staticmethod
    def create_histogram(image, nbins=15, feacture=0):
        """Create a histogram color.

        Args:
            image(ndarray): array of an image
            nbins(int): number of segments
            feacture(int): feacture used to create the histogram of pixel
                           color(0, 1, 2, 3)
        Returns:
            histogram: tuple with the histogram x, y values.

        """
        image_feactures = image[0:image.shape[0], 0: image.shape[1], feacture]
        return exposure.histogram(image_feactures, nbins=nbins)
