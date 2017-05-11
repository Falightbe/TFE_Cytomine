import os
from abc import ABCMeta, abstractmethod
from operator import mul

import numpy as np
from functools import reduce
from PIL import Image

try:
    from sklearn.model_selection import StratifiedShuffleSplit
except ImportError:
    from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state


def list_files(path):
    """
    Get list all the files from the given folder
    Suppose a one-level directory structure: path should contain class folders (integer names) and those folders
    should themselves contain the image files
    """
    folders = list(os.walk(path))
    # Init X and y
    X, y = list(), list()
    for folder, subdir, subfiles in folders[1:]:
        folder_name = os.path.basename(folder)
        if folder_name.isdigit():
            y.extend([int(folder_name)] * len(subfiles))
            X.extend([os.path.join(folder, file) for file in subfiles])
    X = np.array(X)
    y = np.array(y)
    return X, y


def list_and_merge_files(paths):
    """Same as list_files except that file listed from several different path are returned in a same list"""
    result = [list_files(path) for path in paths]
    Xs, ys = zip(*result)
    return np.hstack(Xs), np.hstack(ys)


def _extract_dataset(path):
    """"""
    folders = list(os.walk(path))
    X, y = list(), list()
    for folder, subdir, subfiles in folders[1:]:
        folder_name = os.path.basename(folder)
        if folder_name.isdigit():
            y.extend([int(folder_name)] * len(subfiles))
            X.extend([os.path.join(folder, file) for file in subfiles])
    return np.array(X), np.array(y)


class ImageDataset(metaclass=ABCMeta):
    def __init__(self, image_folder, dirs=None):
        """
        Parameters
        ----------
        image_folder: basestring
            Path the the folder containing the image dataset
        dirs: list
            Name of the folders containing splitted dataset. If none, the subfolders are considered to be classes.
        """
        self._image_folder = image_folder
        self._directories = dirs
        # If directories are defined, _x and _y maps directory names with numpy arrays representing respectively
        # inputs (image files' full path) and outputs for this dataset (format not specified by ImageDataset class).
        # If there is no directories, _x contains directly the inputs (image files' full path), and _y the outputs.
        self._x, self._y = dict(), dict()
        self._init()

    @property
    def image_folder(self):
        return self._image_folder

    @property
    def root_folder(self):
        return self.image_folder

    def _has_dirs(self):
        return self._directories is not None

    def _directory_path(self, directory):
        return os.path.join(self._image_folder, directory)

    def _init(self):
        if self._has_dirs():
            for directory in self._directories:
                self._x[directory], self._y[directory] = self._load(self._directory_path(directory))
        else:
            self._x, self._y = self._load(self._directory_path(self._image_folder))

    @abstractmethod
    def _load(self, directory):
        """Extract the dataset part from the given directory"""
        pass

    def _gather(self, directories=None):
        """Get all the dataset data from the given directories"""
        if self._directories is None:  # no directories
            return self._x, self._y
        if directories is None:
            directories = self._directories
        unknown_directories = set(directories).difference(self._directories)
        if len(unknown_directories):
            raise ValueError("Unknown directories: {}".format(unknown_directories))
        x = [_x for directory in directories for _x in self._x[directory]]
        y = [_y for directory in directories for _y in self._y[directory]]
        return np.array(x), np.array(y)

    @abstractmethod
    def _process(self, x, y, **kwargs):
        """Process and selects the inputs and outputs based on criterion depending on the underlying
        structure of the data
        Returns
        -------
        x: ndarray
            Processed and selected input data
        y: ndarray
            Processed and selected output data
        """
        pass

    def raw_all(self, dirs=None):
        """Return all the unprocessed data from the given directories"""
        return self._gather(directories=dirs)

    def all(self, dirs=None, **kwargs):
        """Return all the data from the given directories"""
        x, y = self.raw_all(dirs=dirs)
        return self._process(x, y, **kwargs)

    def batch(self, size, dirs=None, replace=False, random_state=None, **kwargs):
        """Get a batch of data from the given directories"""
        x, y = self.all(dirs=dirs, **kwargs)
        idx = check_random_state(random_state).choice(x.shape[0], size=(size,), replace=replace)
        return x[idx], y[idx]

    def size(self, dirs=None):
        x, _ = self._gather(directories=dirs)
        return x.shape[0]

    def __len__(self):
        return self.size()

    @staticmethod
    def one_hot_encode(y, classes=None):
        """Perform one hot encoding of the classes in the y. The classes array can be any dimensions.
        The classes to encode shouldn't have their dedicated dimension in the array but should lie in the last one.

        Parameters
        ----------
        y: ndarray
            The classes data
        classes: ndarray
            The actual classes
        """
        reshaped = y.reshape((reduce(mul, y.shape), 1))
        encoder = OneHotEncoder(sparse=False)
        if classes is None:
            encoded = encoder.fit_transform(reshaped)
        else:
            classes = np.array(classes)
            n_classes = classes.shape[0]
            encoder.fit(classes.reshape((n_classes, 1)))
            encoded = encoder.transform(reshaped)
        return encoded.reshape(list(y.shape) + [encoded.shape[1]])

    @staticmethod
    def load_images(files, open_fn=None):
        """Load the images contained in the file in files.

        Parameters
        ----------
        files: iterable
            Iterable containing the image files names
        open_fn: callable
            A function taking as parameter the file name and returning the image (by default, set to None. This
            means that opencv will be used)
        """
        if open_fn is None:
            import cv2
            open_fn = cv2.imread
        images = list()
        for _file in files:
            images.append(np.asarray(open_fn(_file)))
        return images

    @staticmethod
    def make_binary(y, positive):
        """All non-positive are negative"""
        pos = np.in1d(y, positive).reshape(y.shape)
        new_y = np.zeros(y.shape, dtype=np.int)
        new_y[pos] = 1
        return new_y


class ImageClassificationDataset(ImageDataset):
    """Dataset mapping images with classes"""
    def __init__(self, image_folder, dirs=None, classes=None, encode_classes=False):
        """

        Parameters
        ----------
        image_folder:
        dirs:
        classes: list
            List of classes for this problem. If None, the classes are deduced from the subfolders of the first
            directory
        encode_classes: bool
            True for encoding classes as integers in [0, n_classes[
        """
        self._encode_classes = encode_classes
        self._class_map = None
        self._classes = classes
        super(ImageClassificationDataset, self).__init__(image_folder, dirs=dirs)

    def _process(self, x, y, **kwargs):
        """
        Parameters
        ----------
        x: ndarray
        y: ndarray
        classes: list (type: int)
            Classes that should be kept, other classes are discarded (exclusive with negative and positive).
            If all positive, negative and classes are None, then all the classes are taken.
        positive:
            Enables binary classification. Positive classes (exclusive with classes, negative must be passed)
        negative:
            Enables binary classification. Negative classes (exclusive with classes, positive must be passed)
        one_hot: bool
            True for encoding output using one-hot encoding
        kwargs: dict
            Other parameters
        """
        # Extract parameters
        classes = kwargs.get("classes")
        positive = kwargs.get("positive")
        negative = kwargs.get("negative")
        one_hot = kwargs.get("one_hot", False)

        # Process desired classes
        binary = False
        if classes is not None:
            desired_classes = classes
        elif positive is not None and negative is not None:
            binary = True
            desired_classes = positive + negative
        else:
            desired_classes = None

        if self._encode_classes and desired_classes is not None:
            desired_classes = [self._class_map[cls] for cls in desired_classes]

        # Filter classes
        if desired_classes is not None:
            selected = np.in1d(y, desired_classes)
            x, y = x[selected], y[selected]

        if binary:
            y = self.make_binary(y, positive)

        if one_hot:
            y = self.one_hot_encode(y, classes=desired_classes)

        return x, y

    def _set_classes(self, classes):
        self._classes = list(np.sort(classes))
        self._class_map = {cls: i for i, cls in enumerate(self._classes)}

    def _load(self, directory):
        x, y = _extract_dataset(directory)
        if self._classes is None:
            self._set_classes(np.unique(y))
        if self._encode_classes:
            y = np.array([self._class_map[_y] for _y in y])
        return x, y


class ImageSegmentationDataset(ImageDataset):
    """For image segmentation datasets (input image -> output mask)
    Output images are expected to be a grey level image (the level being the class)
    """
    def __init__(self, image_folder, dirs=None, input_dir="images", output_dir="masks", ignore_ext=True):
        """Constructor
        Parameters
        ----------
        image_folder: Root folder
        dirs: list
            Dataset partitions directories
        input_dir: str
            Name of the folder containing the input images (inside a partition folder or in the main folder)
        output_dir: str
            Name of the folder containing the segmentation mask
        ignore_ext: bool
            True for ignoring file extension when matching input output files
        """
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._ignore_ext = ignore_ext
        super(ImageSegmentationDataset, self).__init__(image_folder, dirs=dirs)

    def _load(self, directory):
        # Load images
        input_path = os.path.join(directory, self._input_dir)
        x = next(os.walk(input_path))[2]

        # Load segmentation masks
        output_path = os.path.join(directory, self._output_dir)
        if self._ignore_ext:
            output_files = next(os.walk(output_path))[2]
            output_names = [file.rsplit(".", 1)[0] for file in output_files]
            input_names_set = {file.rsplit(".", 1)[0] for file in x}
            y = [out_file for i, out_file in enumerate(output_files) if output_names[i] in input_names_set]
        else:
            y = [os.path.join(output_path, _x) for _x in x]

        # Build full path
        x = [os.path.join(input_path, _x) for _x in x]
        y = [os.path.join(output_path, _y) for _y in y]
        return np.array(x), np.array(y)

    def _process(self, x, y, **kwargs):
        return x, y

    def load_masks(self, y, encode_classes=False, one_hot=False, classes=None, open_fn=None):
        """Load segmentation masks and optionnaly encode the classes
        Patterns
        --------
        y: iterable
            Mask files
        encode_classes: bool
            True for encoding the classes (classes parameters must be provided)
        one_hot: bool
            True for one-hot encoding classes (classes parameters must be provided)
        classes: list
            List of classes to expect in the mask (classes parameters)
        open_fn: callable
            A function for opening the image files
        """
        masks = self.load_images(y, open_fn=open_fn)
        if encode_classes and not one_hot:  # not need for encoding a class if one_hot is requested
            mapping = {cls: i for i, cls in enumerate(classes)}
            masks = [self.encode_mask(mask, mapping) for mask in masks]
        if one_hot:
            masks = [self.one_hot_encode(mask, classes=classes) for mask in masks]
        return masks

    def _window(self, x, y, dims, random_state):
        """Extract a subwindow with given dimensions from given input image and output mask """
        height, width = dims

        # Compute image and mask dimensions
        if x.ndim == 3:
            x_height, x_width, x_channels = x.shape
            # discard alpha
            if x_channels == 4:
                x = x[:, :, :-1]
        else:
            x_height, x_width = x.shape

        if y.ndim == 3:
            y_height, y_width, _ = y.shape
        else:
            y_height, y_width = y.shape

        if y_height != x_height or y_width != x_width:
            raise ValueError(
                "Invalid size between y ({}) and x ({})".format(y.shape, x.shape))

        if x_height < height or x_width < width:
            raise ValueError("Windows size {} inconsistent with image dimensions {}".format(dims, x.shape))

        # extract window
        height_offset = random_state.randint(x_height - height)
        width_offset = random_state.randint(x_width - width)
        height_end_idx = height_offset + height
        width_end_idx = width_offset + width

        x_window = x[height_offset:height_end_idx, width_offset:width_end_idx]
        y_window = y[height_offset:height_end_idx, width_offset:width_end_idx]
        return x_window, y_window

    def windows_batch(self, size, dims, dirs=None, random_state=None, one_hot=False, encode_classes=False, classes=None, open_fn=None, dtype=np.uint8):
        """
        size: int
            Batch size (number of windows)
        dims: tuple
            Windows dimensions (height, width, channels)
        dirs: iterable
            Partition folders from the images must extracted
        random_state: RandomState
            A random state for random sampling
        one_hot: bool
            True for applying one hot encoding on the output images
        encode_classes: bool
            True for encoding the classes
        classes: iterable
            The classes to be expected in the extracted mask images
        open_fn: callable
            A function for opening the files
        dtype: int
            The window image type
        """
        random_state = check_random_state(random_state)
        inputs, outputs = self.batch(size, dirs=dirs, replace=True, random_state=random_state)

        # TODO avoid duplicate loading
        x = self.load_images(inputs, open_fn=open_fn)
        y = self.load_masks(outputs, encode_classes=encode_classes, one_hot=one_hot, classes=classes, open_fn=open_fn)

        height, width = dims
        x_channels = x[0].shape[2] if x[0].ndim == 3 else 1
        y_channels = y[0].shape[2] if y[0].ndim == 3 else 1
        x_windows = np.zeros([size] + [height, width, x_channels], dtype=dtype)
        y_windows = np.zeros([size] + [height, width, y_channels], dtype=dtype)

        for i, (img, seg) in enumerate(zip(x, y)):
            x_windows[i], y_windows[i] = self._window(img, seg, dims, random_state=random_state)

        return np.squeeze(x_windows), np.squeeze(y_windows)

    @staticmethod
    def encode_mask(mask, mapping):
        cpy_mask = np.copy(mask)
        for cls, mapped in mapping.keys():
            cpy_mask[mask == cls] = mapped
        return cpy_mask


class ImageProvider(object):
    """A class encapsulating """
    def __init__(self, dataset, dirs=None, **kwargs):
        self._kwargs = kwargs
        self._dataset = dataset
        self._dirs = dirs

    def all(self):
        return self._dataset.all(dirs=self._dirs, **self._kwargs)

    def batch(self, size, random_state=None):
        return self._dataset.batch(size, dirs=self._dirs, random_state=random_state, **self._kwargs)

    def __len__(self):
        """Total number of images in the dataset"""
        return self._dataset.size(dirs=self._dirs)
