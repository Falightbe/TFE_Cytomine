import numpy as np
import math
import sys

try:
    import Image
except:
    from PIL import Image

from scipy.sparse import csr_matrix
from scipy.stats.mstats import mode

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals.joblib import Parallel, delayed, cpu_count
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
from sklearn.utils import check_random_state

random_state = check_random_state(None)
MAX_INT = np.iinfo(np.int32).max


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

def _get_image_data(sub_window, colorspace):
    # Convert colorpace
    raw = np.array(sub_window.getdata(), dtype=np.float32)

    #print "raw ndim: %d" %raw.ndim

    if raw.ndim == 1:
        raw = raw[:, np.newaxis]

    #print "raw ndmin after newaxis: %d" %raw.ndim

    if colorspace == COLORSPACE_RGB:
        data = _raw_to_rgb(raw)
    elif colorspace == COLORSPACE_TRGB:
        data = _raw_to_trgb(raw)
    elif colorspace == COLORSPACE_HSV:
        data = _raw_to_hsv(raw)
    elif colorspace == COLORSPACE_GRAY:
        data = _raw_to_gray(raw)

    return data


#Random subwindows extraction (Maree et al., 2014). It extracts subwindows of random sizes at random locations in images (fully contains in the image)
def _random_window(image, min_size, max_size, target_width, target_height, interpolation, transpose, colorspace, fixed_target_window = False, random_state=None):
    random_state = check_random_state(random_state)

    # Draw a random window
    width, height = image.size

    if fixed_target_window: #if true, we don't select randomly the size of the randow window but we use target sizes instead
        crop_width = target_width
        crop_height = target_height
        #if crop_width > width or crop_height > height:
        #    print "Warning: crop larger than image"

    #Rectangular subwindows
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

    #Square subwindows
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
            #subwindow larger than image, so we simply resize original image to target sizes
            sub_window = image.resize((target_width, target_height), pil_interpolation)
        else:
            sub_window = image.crop(box)

    #Rescaling of random size subwindows to fixed-size (target) using interpolation method
    else:
        sub_window = image.crop(box).resize((target_width, target_height), pil_interpolation)

    # Rotate/transpose subwindow
    # We choose randomly a right angle rotation
    if transpose:
        if np.random.rand() > 1.0 / 6:
            sub_window.transpose((Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270)[np.random.randint(5)])

    return sub_window, box

def _get_output_from_mask(target, sub_window):
    assert(sub_window.mode == "RGBA")
    
    mask = np.array(sub_window.split()[3].getdata()) 
    	#im.split() ⇒ sequence
		#Returns a tuple of individual image bands from an image. For example, splitting an “RGB” image creates three new images each containing a copy of one of the original bands (red, green, blue).
    y = np.zeros(mask.shape)
    print "Mask"
    print mask
    print 
    print y
    print y[mask == 255]
    y[mask == 255] = target
    print y
    return y, sub_window.convert('RGB')

#Parallel extraction of subwindows
def _parallel_make_subwindows(X, y, dtype, n_subwindows, min_size, max_size, target_width, target_height, interpolation, transpose, colorspace, fixed, seed, verbose, get_output):
    random_state = check_random_state(seed)

    if colorspace == COLORSPACE_GRAY:
        dim = 1
    else:
        dim = 3 # default

    _X = np.zeros((len(X) * n_subwindows, dim * target_width * target_height), dtype=dtype)
    if get_output == _get_output_from_mask:
        _y = np.zeros((len(X) * n_subwindows, target_width * target_height), dtype=np.int32) #multiple output
    else :
        _y = np.zeros((len(X) * n_subwindows), dtype=np.int32) #single output


    i = 0

    for filename, target in zip(X, y):
        if verbose > 0:
            sys.stdout.write(".")
            sys.stdout.flush()

        image = Image.open(filename)

        if image.mode == "P":
            image = image.convert("RGB")

        for w in xrange(n_subwindows):
            try:
                sub_window, box = _random_window(image, min_size, max_size, target_width, target_height, interpolation, transpose, colorspace, fixed, random_state=random_state)

                output, sub_window = get_output(target, sub_window)
                data = _get_image_data(sub_window, colorspace)
                _X[i, :] = data

            except:
                print
                print "Expected dim =", _X.shape[1]
                print "Got"
                print filename
                raise

            _y[i] = output
            i += 1

    return _X, _y

    
    
    
    

filename = "tmp/82731537/zoom_level/2/20202/82838080_86714800.png"
_X, _y = _parallel_make_subwindows([filename], [1],np.float32, 10, 0.0,1.0,24,24,1,True,2,False,random_state.randint(MAX_INT),True,get_output = _get_output_from_mask)
print "I should have 10*24*24*3 = 17280 in _X and 10*24*24 = 5760 in _y"

print "_X"
print type(_X)
print _X.shape
print _X[0]
print "_y"
print type(_y)
print _y.shape
print sum(sum(_y))

