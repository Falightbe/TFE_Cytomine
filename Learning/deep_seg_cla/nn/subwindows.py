import numpy as np
import sys

from joblib import Parallel
from joblib import delayed

try:
    import Image
except ImportError:
    from PIL import Image

from sklearn.externals.joblib import cpu_count
from sklearn.utils import check_random_state

MAX_INT = np.iinfo(np.int32).max

INTERPOLATION_NEAREST = 1
INTERPOLATION_BILINEAR = 2
INTERPOLATION_CUBIC = 3
INTERPOLATION_ANTIALIAS = 4

COLORSPACE_RGB = 0
COLORSPACE_TRGB = 1
COLORSPACE_HSV = 2
COLORSPACE_GRAY = 3


def _raw_to_rgb(raw):
    return raw.flatten()


def _raw_to_trgb(raw):
    assert raw.shape[1] == 3

    mean = np.atleast_1d(np.mean(raw, axis=0))
    std = np.atleast_1d(np.std(raw, axis=0))

    trgb = np.zeros(raw.shape)

    for i, s in enumerate(std):
        if np.abs(s) > 10E-9:  # Do to divide by zero
            trgb[:, i] = (raw[:, i] - mean[i]) / s

    return trgb.flatten()


def _raw_to_hsv(raw):
    assert raw.shape[1] == 3

    # Min/Max/Diff
    dim = raw.shape[0]
    fmin = np.min(raw, axis=1)
    fmax = np.max(raw, axis=1)
    diff = fmax - fmin

    # Value
    value = np.asarray(fmax, dtype=np.float32)

    # Sat
    sat = np.zeros(dim, dtype=np.float32)
    mask = fmax > 0.0
    sat[mask] = diff[mask] / fmax[mask]

    # Hue
    hue = np.zeros(dim, dtype=np.float32)
    mask = sat > 0.0

    mask_r = mask & (raw[:, 0] == fmax)
    mask_g = mask & (raw[:, 1] == fmax)
    mask_b = mask & (raw[:, 2] == fmax)

    hue[mask_r] = (raw[mask_r, 1] - raw[mask_r, 2]) / diff[mask_r]
    hue[mask_g] = (raw[mask_g, 2] - raw[mask_g, 0]) / diff[mask_g]
    hue[mask_g] += 2.0
    hue[mask_b] = (raw[mask_b, 0] - raw[mask_b, 1]) / diff[mask_b]
    hue[mask_b] += 4.0

    hue *= 60.0
    hue[hue < 0.0] += 360.0
    hue[hue > 360.0] -= 360.

    return np.hstack((hue[:, np.newaxis], sat[:, np.newaxis], value[:, np.newaxis])).flatten()


def _raw_to_gray(raw):
    # print "raw shape: %d" %raw.shape[1]
    return 1.0 * np.sum(raw, axis=1) / raw.shape[1]


# Random subwindows extraction (Maree et al., 2014). It extracts subwindows of random sizes at random locations in
# images (fully contains in the image)
def _random_window(image, min_size, max_size, target_width, target_height, interpolation, transpose, colorspace, fixed_target_window = False, random_state=None):
    random_state = check_random_state(random_state)

    # Draw a random window
    width, height = image.size

    # if true, we don't select randomly the size of the randow window but we use target sizes instead
    if fixed_target_window:
        crop_width = target_width
        crop_height = target_height
        # if crop_width > width or crop_height > height:
        #    print "Warning: crop larger than image"

    # Rectangular subwindows
    elif width < height:
        ratio = 1. * target_height / target_width
        min_width = min_size * width
        max_width = max_size * width

        if min_width * ratio > height:
            raise ValueError

        if max_width * ratio > height:
            max_width = height / ratio

        crop_width = min_width + random_state.rand() * (max_width - min_width)
        crop_height = ratio * crop_width

    # Square subwindows
    else:
        ratio = 1. * target_width / target_height
        min_height = min_size * height
        max_height = max_size * height

        if min_height * ratio > width:
            raise ValueError

        if max_height * ratio > width:
            max_height = width / ratio

        crop_height = min_height + random_state.rand() * (max_height - min_height)
        crop_width = ratio * crop_height

    if crop_width == 0:
        crop_width = 1
    if crop_height == 0:
        crop_height = 1

    # Draw a random position (subwindow fully contain in the image)
    px = int(random_state.rand() * (width - crop_width))
    py = int(random_state.rand() * (height - crop_height))

    # Crop subwindow
    box = (px, py, int(px + crop_width), int(py + crop_height))

    if interpolation == INTERPOLATION_NEAREST:
        pil_interpolation = Image.NEAREST
    elif interpolation == INTERPOLATION_BILINEAR:
        pil_interpolation = Image.BILINEAR
    elif interpolation == INTERPOLATION_CUBIC:
        pil_interpolation = Image.CUBIC
    elif interpolation == INTERPOLATION_ANTIALIAS:
        pil_interpolation = Image.ANTIALIAS
    else:
        pil_interpolation = Image.BILINEAR

    if fixed_target_window:
        if crop_width > width or crop_height > height:
            # subwindow larger than image, so we simply resize original image to target sizes
            sub_window = image.resize((target_width, target_height), pil_interpolation)
        else:
            sub_window = image.crop(box)

    # Rescaling of random size subwindows to fixed-size (target) using interpolation method
    else:
        sub_window = image.crop(box).resize((target_width, target_height), pil_interpolation)

    # Rotate/transpose subwindow
    # We choose randomly a right angle rotation
    if transpose:
        if np.random.rand() > 1.0 / 6:
            sub_window.transpose((Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180,
                                  Image.ROTATE_270)[np.random.randint(5)])

    return sub_window, box


def _get_image_data(sub_window, colorspace):
    # Convert colorpace
    raw = np.array(sub_window.getdata(), dtype=np.float32)

    # print "raw ndim: %d" %raw.ndim

    if raw.ndim == 1:
        raw = raw[:, np.newaxis]

    # print "raw ndmin after newaxis: %d" %raw.ndim
    if colorspace == COLORSPACE_RGB:
        return np.asarray(sub_window)
    elif colorspace == COLORSPACE_TRGB:
        data = _raw_to_trgb(raw)
    elif colorspace == COLORSPACE_HSV:
        data = _raw_to_hsv(raw)
    elif colorspace == COLORSPACE_GRAY:
        data = _raw_to_gray(raw)
    else:
        raise ValueError("Invalid colorspace.")

    return np.reshape(data, sub_window.shape)


# To work on images in parallel
def _partition_images(n_jobs, n_images):
    if n_jobs == -1:
        n_jobs = min(cpu_count(), n_images)

    else:
        n_jobs = min(n_jobs, n_images)

    counts = [n_images // n_jobs] * n_jobs

    for i in range(n_images % n_jobs):
        counts[i] += 1

    starts = [0] * (n_jobs + 1)

    for i in range(1, n_jobs + 1):
        starts[i] = starts[i - 1] + counts[i - 1]

    return n_jobs, counts, starts


# Output Class is the directory from which the image comes from (used in classification)
def _get_output_from_directory(target, sub_window):
    return target, sub_window.convert('RGB')


# Output class is the class of the central pixel (used in single output segmentation, see Dumont et al., 2009)
def _get_output_from_central_pixel(target, sub_window):
    assert(sub_window.mode == "RGBA")
    width, height = sub_window.size
    pixel = sub_window.getpixel(width / 2, height / 2)
    alpha = pixel[3]
    if alpha == 0:
        target = 0
    return target, sub_window.convert('RGB')


# Output classes are the classes of all output pixels (used in Segmentation, see Dumont et al., 2009)
def _get_output_from_mask(target, sub_window):
    assert(sub_window.mode == "RGBA")
    mask = np.array(sub_window.split()[3].getdata())
    y = np.zeros(mask.shape)
    y[mask == 255] = target
    return y, sub_window.convert('RGB')


# Parallel extraction of subwindows
def _parallel_make_subwindows(X, y, n_subwindows, dtype=np.float32, min_size=0.0, max_size=1.0, target_width=32,
                              target_height=32, interpolation=INTERPOLATION_BILINEAR, transpose=False,
                              colorspace=COLORSPACE_RGB, fixed=False, seed=None, verbose=False):
    random_state = check_random_state(seed)

    if colorspace == COLORSPACE_GRAY:
        dim = 1
    else:
        dim = 3  # default

    _X = np.zeros((len(X) * n_subwindows, target_height, target_width, dim), dtype=dtype)
    if len(y.shape) > 1 and y.shape[1] > 1:
        _y = np.zeros((n_subwindows * len(X), y.shape[1]), dtype=y.dtype)
    else :
        _y = np.zeros((n_subwindows * len(X), ), dtype=y.dtype)  # single output

    i = 0

    for filename, target in zip(X, y):
        if verbose > 0:
            sys.stdout.write(".")
            sys.stdout.flush()

        image = Image.open(filename)

        if image.mode == "P":
            image = image.convert("RGB")

        for w in range(n_subwindows):
            try:
                sub_window, box = _random_window(image, min_size, max_size, target_width, target_height, interpolation,
                                                 transpose, colorspace, fixed, random_state=random_state)
                data = _get_image_data(sub_window, colorspace)
                _X[i, :, :, :] = data

            except:
                print()
                print("Expected dim = {}".format(_X.shape[1]))
                print("Got {}".format(data.shape))
                print(filename)
                raise

            _y[i] = target
            i += 1

    return _X, _y


def random_subwindows(X, y, n_subwindows, random_state=None, min_size=0.01, max_size=0.99, target_width=32,
                      target_height=32, n_jobs=1, colorspace=COLORSPACE_HSV, backend="multiprocessing"):
    # init parameters
    random_state = check_random_state(random_state)
    n_jobs, _, starts = _partition_images(n_jobs, len(X))

    # compute windows in parallel
    all_data = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_parallel_make_subwindows)(
            X[starts[i]:starts[i+1]],
            y[starts[i]:starts[i+1]],
            n_subwindows,
            min_size=min_size,
            max_size=max_size,
            target_height=target_height,
            target_width=target_width,
            seed=random_state.randint(MAX_INT),
            colorspace=colorspace
        ) for i in range(n_jobs))

    # Reduce
    _X = np.vstack(X for X, _ in all_data)
    _y = np.concatenate([y for _, y in all_data])
    return _X, _y