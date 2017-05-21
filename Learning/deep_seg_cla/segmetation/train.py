from __future__ import print_function

from skimage.transform import resize
from keras.models import Model as Model_keras
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from data import *
try:
	import Image, ImageStat
	print("try")
except:
	from PIL import Image, ImageStat
	print("except")

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import pickle
from progressbar import *
import os, optparse
from time import localtime, strftime
import socket

import cytomine

# from cytomine.models import *

from pyxit.data import build_from_dir
from pyxit.estimator import _get_image_data, _partition_images, PyxitClassifier, MAX_INT, _get_output_from_mask

from cytomine import cytomine, models
from cytomine_utilities.wholeslide import WholeSlide
from cytomine_utilities.reader import Bounds, CytomineReader

# from extract_tile_subwindow import *

import time
import sys
import copy
from time import localtime, strftime
import socket

import shapely
from shapely.geometry.polygon import Polygon
from sklearn.externals.joblib import Parallel, delayed, cpu_count
import shapely.wkt
from shapely.wkt import loads

from shapely.geometry import box
import numpy as np
import matplotlib.path
from matplotlib.path import Path
import scipy.ndimage

import cv2
import math

import cytomine
from pyxit.data import build_from_dir
from pyxit.estimator import _get_image_data, _partition_images
from cytomine import cytomine, models
from cytomine_utilities.wholeslide import WholeSlide
from cytomine_utilities.objectfinder import ObjectFinder
from cytomine_utilities.reader import Bounds, CytomineReader
from cytomine_utilities.utils import Utils
from cytomine.models import ImageInstanceCollection

K.set_image_data_format('channels_last')  # TF dimension ordering in this code


smooth = 1.


# For parallel extraction of subwindows in current tile
def _parallel_crop_boxes (y_roi, x_roi, image_filename, half_width, half_height, pyxit_colorspace):
	try:
		import Image
	except:
		from PIL import Image

	_X = []
	boxes = np.empty((len(x_roi)*len(y_roi), 4),dtype=np.int)
	i = 0
	image = Image.open(image_filename)
	for y in y_roi:
		for x in x_roi:
			min_x = int(x - half_width)
			min_y = int(y - half_height)
			max_x = int(x + half_width)
			max_y = int(y + half_height)
			boxes[i] = min_x, min_y, max_x, max_y
			sub_window = image.crop(boxes[i])
			sub_window_data = _get_image_data(sub_window, pyxit_colorspace)
			_X.append(sub_window_data)
			i += 1
	return boxes, _X


# For parallel construction of confidence map in current tile
def _parallel_confidence_map(pixels, _Y, offset, boxes, tile_width, tile_height, n_classes, pyxit_target_width, pyxit_target_height):
	votes_class = np.zeros((tile_width, tile_height, n_classes))

	for i in pixels:
		inc_x = i % pyxit_target_width
		inc_y = i / pyxit_target_height

		for box_index, probas in enumerate(_Y[i-offset]):
			px = boxes[box_index][0] + inc_x
			py = boxes[box_index][1] + inc_y
			votes_class[py, px, :] += probas

	return votes_class


# To convert a polyogn into a list of components
def polygon_2_component(polygon):
	exterior = list(polygon.exterior.coords)
	interiors = []
	for interior in polygon.interiors:
		interiors.append(list(interior.coords))
	return (exterior, interiors)


# To convert a union of roi polygons into a rasterized mask
def rasterize_tile_roi_union(nx, ny, points, local_tile_component, roi_annotations_union, whole_slide, reader):
	tile_component = whole_slide.convert_to_real_coordinates(whole_slide, [local_tile_component], reader.window_position, reader.zoom)[0]
	tile_polygon = shapely.geometry.Polygon(tile_component[0], tile_component[1])
	tile_roi_union = tile_polygon.intersection(roi_annotations_union)

	tile_roi_union_components = []
	if (tile_roi_union.geom_type == "Polygon"):
		tile_roi_union_components.append(polygon_2_component(tile_roi_union))
	if (tile_roi_union.geom_type == "MultiPolygon"):
		for geom in tile_roi_union.geoms:
			tile_roi_union_components.append(polygon_2_component(geom))

	local_tile_roi_union_components = whole_slide.convert_to_local_coordinates(whole_slide, tile_roi_union_components, reader.window_position, reader.zoom)
	local_tile_roi_union_polygons = [shapely.geometry.Polygon(component[0], component[1]) for component in local_tile_roi_union_components]

	local_tile_roi_union_raster = np.zeros((ny, nx), dtype=np.bool)
	for polygon in local_tile_roi_union_polygons:
		vertices = np.concatenate([np.asarray(polygon.exterior)] + [np.asarray(r) for r in polygon.interiors])
		#grid = points_inside_poly(points, vertices) #deprecated > matplotlib 1.2
		path = Path(vertices)
		grid = path.contains_points(points)
		grid = grid.reshape((ny,nx))
		local_tile_roi_union_raster |= grid

	return local_tile_roi_union_raster


# To remove unvalid polygon patterns
def process_mask(mask):
	# remove down-left to up-right diagonal pattern
	structure1 = np.zeros((3,3))
	structure1[0,2] = 1
	structure1[1,1] = 1
	structure2 = np.zeros((3,3))
	structure2[0,1] = 1
	structure2[1,2] = 1
	pattern_mask = scipy.ndimage.morphology.binary_hit_or_miss(mask, structure1 = structure1, structure2 = structure2).astype(np.uint8)
	pattern_mask[pattern_mask==1] = 255
	pattern_mask[pattern_mask==0] = 0

	mask = mask - pattern_mask

	# remove up-left to down-right diagonal pattern
	structure1 = np.zeros((3,3))
	structure1[0,0] = 1
	structure1[1,1] = 1
	structure2 = np.zeros((3,3))
	structure2[0,1] = 1
	structure2[1,0] = 1
	pattern_mask = scipy.ndimage.morphology.binary_hit_or_miss(mask, structure1 = structure1, structure2 = structure2).astype(np.uint8)
	pattern_mask[pattern_mask==1] = 255
	pattern_mask[pattern_mask==0] = 0

	mask = mask - pattern_mask
	#TODO the question is :
	# Does removing the second pattern can recreate the first one ? If so, how to avoid it? (iterative way?)


	# remove up line
	structure1 = np.zeros((3,3))
	structure1[2,1] = 1
	structure1[1,1] = 1
	structure2 = np.zeros((3,3))
	structure2[1,0] = 1
	structure2[1,2] = 1
	pattern_mask = scipy.ndimage.morphology.binary_hit_or_miss(mask, structure1 = structure1, structure2 = structure2).astype(np.uint8)
	pattern_mask[pattern_mask==1] = 255
	pattern_mask[pattern_mask==0] = 0
	mask = mask - pattern_mask

	# remove down line
	structure1 = np.zeros((3,3))
	structure1[0,1] = 1
	structure1[1,1] = 1
	structure2 = np.zeros((3,3))
	structure2[1,0] = 1
	structure2[1,2] = 1
	pattern_mask = scipy.ndimage.morphology.binary_hit_or_miss(mask, structure1 = structure1, structure2 = structure2).astype(np.uint8)
	pattern_mask[pattern_mask==1] = 255
	pattern_mask[pattern_mask==0] = 0
	mask = mask - pattern_mask

	# remove left line
	structure1 = np.zeros((3,3))
	structure1[1,1] = 1
	structure1[1,2] = 1
	structure2 = np.zeros((3,3))
	structure2[0,1] = 1
	structure2[2,1] = 1
	pattern_mask = scipy.ndimage.morphology.binary_hit_or_miss(mask, structure1 = structure1, structure2 = structure2).astype(np.uint8)
	pattern_mask[pattern_mask==1] = 255
	pattern_mask[pattern_mask==0] = 0
	mask = mask - pattern_mask

	# remove right line
	structure1 = np.zeros((3,3))
	structure1[1,1] = 1
	structure1[1,0] = 1
	structure2 = np.zeros((3,3))
	structure2[0,1] = 1
	structure2[2,1] = 1
	pattern_mask = scipy.ndimage.morphology.binary_hit_or_miss(mask, structure1 = structure1, structure2 = structure2).astype(np.uint8)
	pattern_mask[pattern_mask==1] = 255
	pattern_mask[pattern_mask==0] = 0
	mask = mask - pattern_mask

	return mask


def str2bool(v):
		return v.lower() in ("yes", "true", "t", "1")


def dice_coef(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)


def get_unet(imgs_width, imgs_height):
	inputs = Input((imgs_width, imgs_height, 3))
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

	up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

	up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

	up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

	up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

	conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

	model = Model_keras(inputs=[inputs], outputs=[conv10])

	model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

	return model


def preprocess(imgs, imgs_width, imgs_height):
	imgs_p = np.ndarray((imgs.shape[0], imgs_width, imgs_height, 3), dtype=np.uint8)
	for i in range(imgs.shape[0]):
		imgs_p[i] = resize(imgs[i], (imgs_width, imgs_height, 3), preserve_range=True)
	imgs_p = imgs_p[..., np.newaxis]
	return imgs_p


def preprocess_mask(imgs, imgs_width, imgs_height):
	imgs_p = np.ndarray((imgs.shape[0], imgs_width, imgs_height), dtype=np.uint8)
	for i in range(imgs.shape[0]):
		imgs_p[i] = resize(imgs[i], (imgs_width, imgs_height), preserve_range=True)
	imgs_p = imgs_p[..., np.newaxis]
	return imgs_p


def train(imgs_train, imgs_mask_train, model_weights_filename, imgs_width, imgs_height, batch_size = 64, epochs = 30, shuffle = True, validation_split = 0.2):
	# create_test_data('/home/falight/TFE_Cytomine/Learning/tmp/deep_segm/images')
	#
	# create_train_data('/home/falight/TFE_Cytomine/Learning/tmp/deep_segm/images')

	# print('-'*30)
	# print('Loading and preprocessing train data...')
	# print('-'*30)
	# imgs_train, imgs_mask_train = load_train_data()
	# n_train = len(imgs_train)
	# imgs_train = preprocess(imgs_train)
	# imgs_mask_train = preprocess_mask(imgs_mask_train)
	# imgs_train = np.reshape(imgs_train, (n_train, img_rows, img_cols, 3))
	# imgs_mask_train = np.reshape(imgs_mask_train, (n_train, img_rows, img_cols, 1))
	# # we create two instances with the same arguments
	# data_gen_args = dict(featurewise_center = True,
	#                      featurewise_std_normalization = True,
	#                      rotation_range = 90.,
	#                      width_shift_range = 0.1,
	#                      height_shift_range = 0.1,
	#                      zoom_range = 0.2)
	# image_datagen = ImageDataGenerator(**data_gen_args)
	# mask_datagen = ImageDataGenerator(**data_gen_args)
	#
	# # Provide the same seed and keyword arguments to the fit and flow methods
	# seed = 1
	# image_datagen.fit(imgs_train, augment = True, seed = seed)
	# mask_datagen.fit(imgs_mask_train, augment = True, seed = seed)
	#
	#
	#
	# image_generator = image_datagen.flow(imgs_train, imgs_mask_train, batch_size = 32, shuffle = True, seed = seed,
	#                                      save_to_dir = '/home/falight/TFE_Cytomine/Learning/tmp/deep_segm/images/augmented_data',
	#                                      save_format = 'png')
	#
	#
	#
	# print('-'*30)
	# print('Creating and compiling model...')
	# print('-'*30)
	# model = get_unet()
	# model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
	#
	# print('-'*30)
	# print('Fitting model...')
	# print('-'*30)
	# model.fit_generator(
	#     image_generator,
	#     steps_per_epoch = n_train/32,
	#     epochs = 20)
	#
	# print('-'*30)
	# print('Loading and preprocessing test data...')
	# print('-'*30)
	# imgs_test, imgs_id_test = load_test_data()
	#
	# print('-'*30)
	# print('Loading saved weights...')
	# print('-'*30)
	# model.load_weights('weights.h5')
	#
	# print('-'*30)
	# print('Predicting masks on test data...')
	# print('-'*30)
	# imgs_mask_test = model.predict(imgs_test, verbose=1)
	# np.save('imgs_mask_test.npy', imgs_mask_test)
	#
	# print('-' * 30)
	# print('Saving predicted masks to files...')
	# print('-' * 30)
	# pred_dir = '/home/falight/TFE_Cytomine/Learning/tmp/deep_segm/images/preds_2'
	# if not os.path.exists(pred_dir):
	#     os.mkdir(pred_dir)
	# for image, image_id in zip(imgs_mask_test, imgs_id_test):
	#     imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

	print("Data shape :")
	print(imgs_train.shape)
	print(imgs_mask_train.shape)

	imgs_train = imgs_train.astype('float32')
	mean = np.mean(imgs_train)  # mean for data centering
	std = np.std(imgs_train)  # std for data normalization

	imgs_train -= mean
	imgs_train /= std
	imgs_mask_train = imgs_mask_train.astype('float32')

	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	model = get_unet(imgs_width, imgs_height)
	model_checkpoint = ModelCheckpoint(model_weights_filename, monitor='val_loss', save_best_only=True)

	print('-'*30)
	print('Fitting model...')
	print('-'*30)
	n_samples = len(imgs_train)
	print(imgs_train.shape)
	print(imgs_mask_train.shape)
	imgs_mask_train = np.reshape(imgs_mask_train, (n_samples, imgs_width, imgs_height, 1))

	model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=shuffle,
			  validation_split=validation_split,
			  callbacks=[model_checkpoint])

	return mean, std


def predict(imgs_test, model_weights_filename, imgs_width, imgs_height, mean, std):
	print("Data shape :")
	print(imgs_test.shape)


	imgs_test = imgs_test.astype('float32')
	imgs_test -= mean
	imgs_test /= std

	print('-' * 30)
	print('Creating and compiling model...')
	print('-' * 30)
	model = get_unet(imgs_width, imgs_height)
	model_checkpoint = ModelCheckpoint(model_weights_filename, monitor = 'val_loss', save_best_only = True)

	print('-'*30)
	print('Loading saved weights...')
	print('-'*30)
	model.load_weights(model_weights_filename)

	print('-'*30)
	print('Predicting masks on test data...')
	print('-'*30)
	imgs_mask_test = model.predict(imgs_test, verbose=1)
	# np.save('imgs_mask_test.npy', imgs_mask_test)
	#
	#
	# imgs_mask_test = np.load('imgs_mask_test.npy')
	# print('-' * 30)
	# print('Saving predicted masks to files...')
	# print('-' * 30)
	# pred_dir = '/home/falight/TFE_Cytomine/Learning/tmp/deep_segm/images/preds'
	# if not os.path.exists(pred_dir):
	# 	os.mkdir(pred_dir)
	# for image, image_id in zip(imgs_mask_test, imgs_id_test):
	# 	image = (image[:, :, 0] * 255.).astype(np.uint8)
	# 	imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

	return imgs_mask_test


def main(argv):
	# Define command line options
	p = optparse.OptionParser(description='Cytomine Segmentation prediction', prog='Cytomine segmentation prediction', version='0.1')

	p.add_option('--cytomine_host', type="string", default = 'beta.cytomine.be', dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
	p.add_option('--cytomine_public_key', type="string", default = '', dest="cytomine_public_key", help="Cytomine public key")
	p.add_option('--cytomine_private_key',type="string", default = '', dest="cytomine_private_key", help="Cytomine private key")
	p.add_option('--cytomine_base_path', type="string", default = '/api/', dest="cytomine_base_path", help="Cytomine base path")
	p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")
	p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path", help="The working directory (eg: /tmp)")
	p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")
	p.add_option('--cytomine_union', type="string", default="0", dest="cytomine_union", help="Turn on union of geometries")
	p.add_option('--cytomine_postproc', type="string", default="0", dest="cytomine_postproc", help="Turn on postprocessing")
	p.add_option('--cytomine_count', type="string", default="0", dest="cytomine_count", help="Turn on object counting")

	p.add_option('--cytomine_min_size', type="int", default=0, dest="cytomine_min_size", help="minimum size (area) of annotations")
	p.add_option('--cytomine_max_size', type="int", default=10000000000, dest="cytomine_max_size", help="maximum size (area) of annotations")
	p.add_option('--cytomine_mask_internal_holes', type='string', default="0", dest="cytomine_mask_internal_holes", help="Turn on precise hole finding")

	p.add_option('-i', '--cytomine_id_image', type='int', dest='cytomine_id_image', help="image id from cytomine", metavar='IMAGE')
	p.add_option('-z', '--cytomine_zoom_level', type='int', dest='cytomine_zoom_level', help="working zoom level")
	p.add_option('--cytomine_tile_size', type='int', dest='cytomine_tile_size', help="sliding tile size")
	p.add_option('--cytomine_tile_min_stddev', type='int', default=5, dest='cytomine_tile_min_stddev', help="tile minimum standard deviation")
	p.add_option('--cytomine_tile_max_mean', type='int', default=250, dest='cytomine_tile_max_mean', help="tile maximum mean")
	p.add_option('--cytomine_tile_overlap', type = 'int', default = 250, dest = 'cytomine_tile_overlap', help = "tile overlap")
	p.add_option('--cytomine_union_min_length', type='int', default=5, dest='cytomine_union_min_length', help="union")
	p.add_option('--cytomine_union_bufferoverlap', type='int', default=5, dest='cytomine_union_bufferoverlap', help="union")
	p.add_option('--cytomine_union_area', type='int', default=5, dest='cytomine_union_area', help="union")
	p.add_option('--cytomine_union_min_point_for_simplify', type='int', default=5, dest='cytomine_union_min_point_for_simplify', help="union")
	p.add_option('--cytomine_union_min_point', type='int', default=5, dest='cytomine_union_min_point', help="union")
	p.add_option('--cytomine_union_max_point', type='int', default=5, dest='cytomine_union_max_point', help="union")
	p.add_option('--cytomine_union_nb_zones_width', type='int', default=5, dest='cytomine_union_nb_zones_width', help="union")
	p.add_option('--cytomine_union_nb_zones_height', type='int', default=5, dest='cytomine_union_nb_zones_height', help="union")
	p.add_option('-j', '--nb_jobs', type='int', dest='nb_jobs', help="number of parallel jobs")
	p.add_option('--startx', type='int', default=0, dest='cytomine_startx', help="start x position")
	p.add_option('--starty', type='int', default=0, dest='cytomine_starty', help="start y position")
	p.add_option('--endx', type='int', dest='cytomine_endx', help="end x position")
	p.add_option('--endy', type='int', dest='cytomine_endy', help="end y position")
	p.add_option('--cytomine_predict_terms', type='str', dest='cytomine_predict_terms', help="term id of predicted terms (binary mode)")
	p.add_option('--cytomine_roi_term', type='string', dest='cytomine_roi_term', help="term id of region of interest where to count)")
	p.add_option('--cytomine_reviewed_roi', type='string', default="0", dest="cytomine_reviewed_roi", help="Use reviewed roi only")

	p.add_option('--pyxit_target_width', type='int', dest='pyxit_target_width', help="pyxit subwindows width")
	p.add_option('--pyxit_target_height', type='int', dest='pyxit_target_height', help="pyxit subwindows height")
	p.add_option('--cytomine_predict_step', type='int', dest='cytomine_predict_step', help="pyxit step between successive subwindows")
	p.add_option('--pyxit_post_classification', type="string", default="0", dest="pyxit_post_classification", help="pyxit post classification of candidate annotations")
	p.add_option('--pyxit_post_classification_save_to', type='string', default = "", dest='pyxit_post_classification_save_to', help="pyxit post classification model file") #future: get it from server db
	p.add_option('--pyxit_save_to', type = 'string', default = "", dest = 'pyxit_save_to', help = "pyxit model file")
	p.add_option('--pyxit_colorspace', type='int', dest='pyxit_colorspace', help="pyxit colorspace encoding")
	p.add_option('--pyxit_nb_jobs', type='int', dest='pyxit_nb_jobs', help="pyxit number of jobs for trees")
	p.add_option('--pyxit_fixed_size', type = 'string', default = "0", dest = "pyxit_fixed_size", help = "extract fixed size subwindows")
	p.add_option('--pyxit_transpose', type = 'string', default = "0", dest = "pyxit_transpose", help = "transpose subwindows")
	p.add_option('--pyxit_n_subwindows', type='int', default="10", dest="pyxit_n_subwindows", help="number of subwindows")
	p.add_option('--pyxit_interpolation', default = 2, type = "int", dest = "pyxit_interpolation", help = "interpolation method 1,2,3,4")
	p.add_option('--pyxit_min_size', default = 0.5, type = "float", dest = "pyxit_min_size", help = "min size")
	p.add_option('--pyxit_max_size', default = 1.0, type = "float", dest = "pyxit_max_size", help = "max size")
	p.add_option('--cytomine_reviewed', type = 'string', default = "False", dest = "cytomine_reviewed", help = "Get reviewed annotations only")

	p.add_option('--cytomine_dump_annotations', type = 'string', default = "0", dest = "cytomine_dump_annotations",
				 help = "Dump training annotations or not")
	p.add_option('--build_model', type = "string", default = "0", dest = "build_model",
				 help = "Turn on (1) or off (0) model building")
	p.add_option('--predict', type = "string", default = "0", dest = "predict",
				 help = "Turn on (1) or off (0) prediction on test images")
	p.add_option('--cytomine_annotation_projects', type = "string", dest = "cytomine_annotation_projects",
				 help = "Projects from which annotations are extracted")
	p.add_option('--cytomine_predict_projects', type = "string", dest = "cytomine_predict_projects",
				 help = "Projects on which prediction is done")
	p.add_option('--verbose', type = "string", default = "0", dest = "verbose", help = "Turn on (1) or off (0) verbose mode")

	p.add_option('--keras_batch_size', type = "int", dest = "keras_batch_size", help = "Training batch size")
	p.add_option('--keras_n_epochs', type = "int", dest = "keras_n_epochs", help = "Number of epochs")
	p.add_option('--keras_shuffle', type = "string", dest = "keras_shuffle", help = "Turn on (1) or off (0) batch shuffle")
	p.add_option('--keras_validation_split', type = "float", dest = "keras_validation_split", help = "Batch validation split")
	options, arguments = p.parse_args( args = argv)


	parameters = {}
	parameters['keras_batch_size'] = options.keras_batch_size
	parameters['keras_n_epochs'] = options.keras_n_epochs
	parameters['keras_shuffle'] = options.keras_shuffle
	parameters['keras_validation_split'] = options.keras_validation_split
	parameters['cytomine_host'] = options.cytomine_host
	parameters['cytomine_public_key'] = options.cytomine_public_key
	parameters['cytomine_private_key'] = options.cytomine_private_key
	parameters['cytomine_base_path'] = options.cytomine_base_path
	parameters['cytomine_working_path'] = options.cytomine_working_path
	parameters['cytomine_base_path'] = options.cytomine_base_path
	parameters['cytomine_id_project'] = options.cytomine_id_project
	parameters['cytomine_id_software'] = options.cytomine_id_software
	parameters['cytomine_predict_terms'] = map(int, options.cytomine_predict_terms.split(','))
	parameters['model_id_job'] = 0
	if options.cytomine_roi_term:
		parameters['cytomine_roi_term'] = map(int,options.cytomine_roi_term.split(','))
	parameters['cytomine_reviewed_roi'] = str2bool(options.cytomine_reviewed_roi)
	parameters['cytomine_union'] = str2bool(options.cytomine_union)
	parameters['cytomine_postproc'] = str2bool(options.cytomine_postproc)
	parameters['cytomine_mask_internal_holes'] = str2bool(options.cytomine_mask_internal_holes)
	parameters['cytomine_count'] = str2bool(options.cytomine_count)
	if options.cytomine_min_size:
		parameters['cytomine_min_size'] = options.cytomine_min_size
	if options.cytomine_max_size:
		parameters['cytomine_max_size'] = options.cytomine_max_size
	parameters['cytomine_predict_step'] = options.cytomine_predict_step
	# parameters['pyxit_save_to'] = options.pyxit_save_to
	parameters['pyxit_post_classification'] = str2bool(options.pyxit_post_classification)
	parameters['pyxit_post_classification_save_to'] = options.pyxit_post_classification_save_to
	parameters['pyxit_save_to'] = options.pyxit_save_to
	parameters['pyxit_colorspace'] = options.pyxit_colorspace
	parameters['pyxit_nb_jobs'] = options.pyxit_nb_jobs
	parameters['pyxit_n_jobs'] = options.pyxit_nb_jobs
	parameters['cytomine_nb_jobs'] = options.pyxit_nb_jobs
	parameters['cytomine_id_image'] = options.cytomine_id_image
	parameters['cytomine_zoom_level'] = options.cytomine_zoom_level
	parameters['cytomine_tile_size'] = options.cytomine_tile_size
	parameters['cytomine_tile_min_stddev'] = options.cytomine_tile_min_stddev
	parameters['cytomine_tile_max_mean'] = options.cytomine_tile_max_mean
	parameters['cytomine_tile_overlap'] = options.cytomine_tile_overlap
	parameters['cytomine_union_min_length'] = options.cytomine_union_min_length
	parameters['cytomine_union_bufferoverlap'] = options.cytomine_union_bufferoverlap
	parameters['cytomine_union_area'] = options.cytomine_union_area
	parameters['cytomine_union_min_point_for_simplify'] = options.cytomine_union_min_point_for_simplify
	parameters['cytomine_union_min_point'] = options.cytomine_union_min_point
	parameters['cytomine_union_max_point'] = options.cytomine_union_max_point
	parameters['cytomine_union_nb_zones_width'] = options.cytomine_union_nb_zones_width
	parameters['cytomine_union_nb_zones_height'] = options.cytomine_union_nb_zones_height
	parameters['cytomine_startx'] = options.cytomine_startx
	parameters['cytomine_starty'] = options.cytomine_starty
	parameters['cytomine_endx'] = options.cytomine_endx
	parameters['cytomine_endy'] = options.cytomine_endy
	parameters['nb_jobs'] = options.nb_jobs
	parameters['pyxit_target_width'] = options.pyxit_target_width
	parameters['pyxit_target_height'] = options.pyxit_target_height
	parameters['dir_ls'] = ""
	parameters['pyxit_n_subwindows'] = 100
	parameters['pyxit_interpolation'] = 1
	parameters['pyxit_transpose'] = True
	parameters['pyxit_min_size'] = 0.1
	parameters['pyxit_max_size'] = 1
	parameters['pyxit_fixed_size'] = False
	parameters['forest_n_estimators'] = 10
	parameters['forest_max_features'] = 28
	parameters['forest_min_samples_split'] = 10
	parameters['forest_shared_mem'] = True
	parameters['svm'] = 0
	parameters['svm_c'] = 1.0
	parameters['cytomine_annotation_projects'] = map(int, options.cytomine_annotation_projects.split(','))
	parameters['cytomine_reviewed'] = str2bool(options.cytomine_reviewed)
	parameters['cytomine_dump_annotations'] = str2bool(options.cytomine_dump_annotations)
	parameters['build_model'] = str2bool(options.build_model)
	parameters['predict'] = str2bool(options.predict)
	parameters['dir_ls'] = os.path.join(parameters["cytomine_working_path"],
										str(parameters['cytomine_annotation_projects']).replace(',', '-').replace(
											'[', '').replace(']', '').replace(' ', ''), "zoom_level",
										str(parameters['cytomine_zoom_level']))

	pyxit_parameters = {}
	pyxit_parameters['pyxit_target_width'] = options.pyxit_target_width
	pyxit_parameters['pyxit_target_height'] = options.pyxit_target_height
	pyxit_parameters['pyxit_n_subwindows'] = options.pyxit_n_subwindows
	pyxit_parameters['pyxit_min_size'] = options.pyxit_min_size
	pyxit_parameters['pyxit_max_size'] = options.pyxit_max_size
	pyxit_parameters['pyxit_colorspace'] = options.pyxit_colorspace
	pyxit_parameters['pyxit_interpolation'] = options.pyxit_interpolation
	pyxit_parameters['pyxit_transpose'] = str2bool(options.pyxit_transpose)
	pyxit_parameters['pyxit_fixed_size'] = str2bool(options.pyxit_fixed_size)
	pyxit_parameters['pyxit_n_jobs'] = options.pyxit_nb_jobs
	pyxit_target_width = parameters['pyxit_target_width']
	pyxit_target_height = parameters['pyxit_target_height']
	zoom = parameters['cytomine_zoom_level']
	predictionstep= parameters['cytomine_predict_step']

	if options.verbose :
		print(parameters)

	# Create Cytomine connection
	conn = cytomine.Cytomine(parameters["cytomine_host"],
							 parameters["cytomine_public_key"],
							 parameters["cytomine_private_key"],
							 base_path = parameters['cytomine_base_path'],
							 working_path = parameters['cytomine_working_path'],
							 verbose = str2bool(options.verbose))


	# Create a new userjob if connected as human user
	print("Create Job and UserJob...")
	id_software = parameters['cytomine_id_software']
	current_user = conn.get_current_user()
	run_by_user_job = False
	if not current_user.algo :
		print("adduserJob...")
		user_job = conn.add_user_job(parameters['cytomine_id_software'], parameters['cytomine_id_project'])
		print ("set_credentials...")
		conn.set_credentials(str(user_job.publicKey), str(user_job.privateKey))
		print("done")
	else :
		user_job = current_user
		print("Already running as userjob")
		run_by_user_job = True
	job = conn.get_job(user_job.job)

	job = conn.update_job_status(job, status_comment = "Publish software parameters values")
	if not run_by_user_job :
		job_parameters_values = conn.add_job_parameters(user_job.job,
														conn.get_software(parameters['cytomine_id_software']),
														parameters)
	job = conn.update_job_status(job, status = job.RUNNING, progress = 0, status_comment = "Loading data...")

	# Dump annotations
	if parameters['cytomine_dump_annotations'] :
		# Get annotation descriptions (JSON) from project(s)
		annotations = None
		for prj in parameters['cytomine_annotation_projects'] :
			if parameters["cytomine_reviewed"] :
				annotations_prj = conn.get_annotations(id_project = prj, reviewed_only = parameters["cytomine_reviewed"])
			else :
				annotations_prj = conn.get_annotations(id_project = prj)
			if not annotations :
				annotations = annotations_prj
			else :
				annotations.data().extend(annotations_prj.data())

			if prj == 21907448 or prj == 155194683 :
				annotations_prj = conn.get_annotations(id_project = prj, id_term = 91376951)
				annotations.data().extend(annotations_prj.data())
			print("Nb annotations so far... = %d" % len(annotations.data()))
		print("Total annotations projects %s = %d" % (parameters['cytomine_annotation_projects'], len(annotations.data())))

		# Set output dir parameters
		if not os.path.exists(parameters['dir_ls']) :
			print("Creating annotation directory: %s" % parameters['dir_ls'])
			os.makedirs(parameters['dir_ls'])

		# Dump annotation images locally
		print("Dump training annotation images in %s...", parameters['dir_ls'])
		conn.dump_annotations(annotations = annotations, get_image_url_func = Annotation.get_annotation_alpha_crop_url,
							  dest_path = parameters['dir_ls'],
							  desired_zoom = parameters['cytomine_zoom_level'],
							  excluded_terms = parameters['cytomine_roi_term'])

		# Put positive terms under the same term and same for negative terms
		term_directories = os.listdir(parameters['dir_ls'])

		pos_path = os.path.join(parameters['dir_ls'], "1")
		if not os.path.exists(pos_path) :
			print("Creating positive annotation directory: %s" % pos_path)
			os.makedirs(pos_path)

		neg_path = os.path.join(parameters['dir_ls'], "0")
		if not os.path.exists(neg_path) :
			print("Creating negative annotation directory: %s" % neg_path)
			os.makedirs(neg_path)

		for dir in term_directories :
			dir_abs = os.path.join(parameters['dir_ls'], dir)
			print("Working on ", dir_abs)
			# Move files
			if int(dir) in parameters['cytomine_predict_terms'] :
				for image_file in os.listdir(dir_abs) :
					os.rename(os.path.join(dir_abs, image_file), os.path.join(pos_path, image_file))
			else:
				for image_file in os.listdir(dir_abs) :
					os.rename(os.path.join(dir_abs, image_file), os.path.join(neg_path, image_file))
			# Remove empty directory
			if int(dir) != 0 and int(dir) != 1:
				os.rmdir(dir_abs)

	model_name = "nsubw{}_winsize{}x{}_batchsize{}_epochs{}_shuffle{}_valsplit{}.h5"\
		.format(pyxit_parameters['pyxit_n_subwindows'],
				pyxit_parameters['pyxit_target_width'],
				pyxit_parameters['pyxit_target_height'],
				parameters['keras_batch_size'],
				parameters['keras_n_epochs'],
				parameters['keras_shuffle'],
				parameters['keras_validation_split'])
	if parameters['build_model'] :
		pyxit = PyxitClassifier(None,
								n_subwindows=pyxit_parameters['pyxit_n_subwindows'],
								min_size=pyxit_parameters['pyxit_min_size'],
								max_size=pyxit_parameters['pyxit_max_size'],
								target_width=pyxit_parameters['pyxit_target_width'],
								target_height=pyxit_parameters['pyxit_target_height'],
								n_jobs=pyxit_parameters['pyxit_n_jobs'],
								interpolation=pyxit_parameters['pyxit_interpolation'],
								transpose=pyxit_parameters['pyxit_transpose'],
								colorspace=pyxit_parameters['pyxit_colorspace'],
								fixed_size=pyxit_parameters['pyxit_fixed_size'],
								random_state=None,
								verbose=1,
								get_output = _get_output_from_mask,
								parallel_leaf_transform=False)
		# Build filenames and classes
		X, y = build_from_dir(parameters['dir_ls'])

		classes = np.unique(y)
		n_classes = len(classes)
		y_original = y
		y = np.searchsorted(classes, y)
		n_images = len(y)
		print("Number of images : ", n_images)

		# Extract subwindows
		_X, _y = pyxit.extract_subwindows(X, y)
		n_subw = len(_y)
		print("Number of subwindows : ", n_subw)

		# Reshape data structure
		_X = np.reshape(_X, (n_subw, pyxit_parameters['pyxit_target_width'], pyxit_parameters['pyxit_target_height'], 3))
		_y = np.reshape(_y, (n_subw, pyxit_parameters['pyxit_target_width'], pyxit_parameters['pyxit_target_height']))


		# Train FCN
		model_weights_filename = "weights_" + model_name + ".h5"
		mean, std = train(_X, _y,
						  model_weights_filename,
						  imgs_width = pyxit_parameters['pyxit_target_width'],
						  imgs_height = pyxit_parameters['pyxit_target_height'],
						  batch_size = parameters['keras_batch_size'],
						  epochs = parameters['keras_n_epochs'],
						  shuffle = parameters['keras_shuffle'],
						  validation_split = parameters['keras_validation_split']
						  )

		# Save mean and std used to normalize training data
		mean_std_save_filename = "meanstd_" + model_name + ".txt"
		mean_std_save_file = open(mean_std_save_filename, 'w')
		mean_std_save_file.write(str(mean)+'\n')
		mean_std_save_file.write(str(std)+'\n')




	if parameters['predict'] :
		# Retrieve mean and std used to normalize training data
		mean_std_save_filename = "meanstd_" + model_name + ".txt"
		mean_std_save_file = open(mean_std_save_filename, 'r')
		training_sample_mean = float(mean_std_save_file.readline())
		training_sample_std = float(mean_std_save_file.readline())

		# predict(imgs_test, model_weights_filename, mean, std)

		# Retrieve images to predict
		with open("image_folders_record.csv", 'r') as f :
			image_folders = f.readlines()


		print ("Nb images: %d" % len(image_folders))
		progress = 0
		progress_delta = 100 / len(image_folders)
		i_image = 0

		# Go through all images
		for image_name in image_folders :
			id_project = int(image_name.split('project-')[1].split('/crop')[0])

			id_image = int(image_name.split('candidates-')[1].split('-')[0])

			progress_msg = "Analyzing image %s (%d / %d )..." % (id_image, i_image, len(image_folders))
			job = conn.update_job_status(job, status = job.RUNNING, progress = progress, status_comment = progress_msg)

			# print "image id: %d width: %d height: %d resolution: %f magnification: %d filename: %s" %(image.id,image.width,image.height,image.resolution,image.magnification,image.filename)

			# Create local object to access the remote whole slide
			print("Creating connector to Slide Image from Cytomine server")
			image_instance = conn.get_image_instance(id_image, True)
			whole_slide = WholeSlide(image_instance)
			print("Whole slide: %d x %d pixels" % (whole_slide.width, whole_slide.height))


			# endx and endy allow to stop image analysis at a given x,y position  (for debugging)
			if not parameters['cytomine_endx'] and not parameters['cytomine_endy'] :
				print ("End is not defined.")
				endx = whole_slide.width
				endy = whole_slide.height
			else :
				endx = parameters['cytomine_endx']
				endy = parameters['cytomine_endy']

			# Initialize variables and tools for roi
			nx = parameters['cytomine_tile_size']
			ny = parameters['cytomine_tile_size']

			x, y = np.meshgrid(np.arange(nx), np.arange(ny))
			x, y = x.flatten(), y.flatten()
			points = np.vstack((x, y)).T
			local_tile_component = ([(0, 0), (0, ny), (nx, ny), (nx, 0), (0, 0)], [])

			# We can apply the segmentation model either in the whole slide (including background area), or only within multiple ROIs (of a given term)
			# For example ROI could be generated first using a thresholding step to detect the tissue
			# Here we build a polygon union containing all roi_annotations locations (user or reviewed annotations) to later match tile with roi masks
			if parameters['cytomine_roi_term'] or parameters['cytomine_reviewed_roi'] :
				if parameters['cytomine_reviewed_roi'] :
					# retrieve reviewed annotations for roi
					roi_annotations = conn.get_annotations(id_image = id_image,
														   id_term = str(parameters['cytomine_roi_term']).replace('[','').replace(']', '').replace(' ', ''),
														   id_project = id_project,
														   reviewed_only = True)
				else :
					# retrieve annotations with roi term
					roi_annotations = conn.get_annotations(id_image = id_image,
														   id_term = str(parameters['cytomine_roi_term']).replace('[','').replace(']', '').replace(' ', ''),
														   id_project = id_project)

				time.sleep(1)
				roi_annotations_locations = []
				for simplified_roi_annotation in roi_annotations.data() :
					roi_annotation = conn.get_annotation(simplified_roi_annotation.id)
					# roi_area_um += roi_annotation.area
					assert shapely.wkt.loads(
						roi_annotation.location).is_valid, "one roi_annotation.location is not valid"
					roi_annotations_locations.append(shapely.wkt.loads(roi_annotation.location))
					roi_annotations_union = roi_annotations_locations[0]
				for annot in roi_annotations_locations[1 :] :
					roi_annotations_union = roi_annotations_union.union(annot)
			else : # no ROI used
				# We build a rectangular roi_mask corresponding to the whole image filled with ones
				print ("We will process all tiles (no roi provided)")
				roi_mask = np.ones((ny, nx), dtype = np.bool)

			# Initiate the reader object which browse the whole slide image with tiles of size tile_size
			print ("Initiating the Slide reader")
			reader = CytomineReader(conn,
									whole_slide,
									window_position = Bounds(parameters['cytomine_startx'],
															 parameters['cytomine_starty'],
															 parameters['cytomine_tile_size'],
															 parameters['cytomine_tile_size']),
									zoom = zoom,
									overlap = parameters['cytomine_tile_overlap'] + 1)
			# opencv object image corresponding to a tile
			# cv_image = cv.CreateImageHeader((reader.window_position.width, reader.window_position.height), cv.IPL_DEPTH_8U, 1)
			wsi = 0
			geometries = []

			print ("Starting browsing the image using tiles")
			# posx,posy,poswidth,posheight = reader.window_position.x, reader.window_position.y, reader.window_position.width,reader.window_position.height
			iteration = 0
			while True :

				# print "***********************************************************"
				# print "********************** ITERATION {} ***********************".format(iteration)
				# print "***********************************************************"
				# iteration = iteration + 1

				# Get rasterized roi mask to match with this tile (if no ROI used, the roi_mask was built before and
				# corresponds to the whole image).
				if parameters['cytomine_roi_term'] :
					roi_mask = rasterize_tile_roi_union(nx, ny, points, local_tile_component, roi_annotations_union,
														whole_slide, reader)

				if np.count_nonzero(roi_mask) :
					print("nonzero roi_mask")
					posx, posy, poswidth, posheight = reader.window_position.x, reader.window_position.y, reader.window_position.width, reader.window_position.height
					print("Tile zoom: %d posx: %d posy: %d poswidth: %d posheight: %d" % (
					zoom, posx, posy, poswidth, posheight))
					image_folder_name = "%s/prediction/slides/project-%d/tiles" % (parameters['cytomine_working_path'], id_project)
					if not os.path.exists(image_folder_name) :
						print("Creating annotation directory: %s" % image_folder_name)
						os.makedirs(image_folder_name)

					image_filename = os.path.join(image_folder_name, "%d-zoom_%d-tile_%d_x%d_y%d_w%d_h%d.png" % (
					id_image, zoom, wsi, posx, posy, poswidth, posheight))
					save_image = True
					print("Image file :", image_filename)
					if not os.path.isfile(image_filename) :
						print("IMAGE FILE READ FROM WHOLESLIDE")
						quit()
						save_image = False
						read = False
						while (not read) :
							print("TIME : %s" % strftime("%Y-%m-%d %H:%M:%S", localtime()))
							try :
								reader.read(async = False)
								read = True
							except socket.error :
								print ("except socket.error...")
								print("TIME : %s" % strftime("%Y-%m-%d %H:%M:%S", localtime()))
								print (socket.error())
								continue
							except socket.timeout :
								print ("except socket.timeout...")
								print("TIME : %s" % strftime("%Y-%m-%d %H:%M:%S", localtime()))
								print (socket.timeout)
								continue
							except :
								print ("except...")
								print ("TIME : %s" % strftime("%Y-%m-%d %H:%M:%S", localtime()))
								continue
						image = reader.data
						image.save(image_filename, "PNG")
						print ("END READ...")
					else :
						print ("IMAGE FILE READ FROM FILE")
						image = Image.open(image_filename)

					# Get statistics about the current tile
					print ("Computing tile statistics")
					tilemean = ImageStat.Stat(image).mean
					print ("Tile mean pixel values: %d %d %d" % (tilemean[0], tilemean[1], tilemean[2]))
					tilevar = ImageStat.Stat(image).var
					print ("Tile variance pixel values: %d %d %d" % (tilevar[0], tilevar[1], tilevar[2]))
					tilestddev = ImageStat.Stat(image).stddev
					print ("Tile stddev pixel values: %d %d %d" % (tilestddev[0], tilestddev[1], tilestddev[2]))
					extrema = ImageStat.Stat(image).extrema
					print (extrema)
					print ("extrema: min R:%d G:%d B:%d" % (extrema[0][0], extrema[1][0], extrema[2][0]))

					# Criteria to determine if tile is empty, specific to this application
					mindev = parameters['cytomine_tile_min_stddev']
					maxmean = parameters['cytomine_tile_max_mean']
					if (((tilestddev[0] < mindev) and (tilestddev[1] < mindev) and (tilestddev[2] < mindev)) or (
							(tilemean[0] > maxmean) and (tilemean[1] > maxmean) and (tilemean[2] > maxmean))) :
						print ("Tile empty (filtered by min stddev or max mean)")

					else :
						# This tile is not empty, we process it
						# Add current tile annotation on server just for progress visualization purpose, not working
						# current_tile = box(pow(2, zoom)*posx,
						#                   whole_slide.height-pow(2, zoom)*posy-pow(2, zoom)*parameters['cytomine_tile_size'],
						#                   pow(2, zoom)*posx+pow(2, zoom)*parameters['cytomine_tile_size'],
						#                   whole_slide.height-pow(2, zoom)*posy)
						# current_tile_annotation = conn.add_annotation(current_tile.wkt, id_image)

						print ("TIME : %s" % strftime("%Y-%m-%d %H:%M:%S", localtime()))
						print ("Tile file: %s" % image_filename)
						print ("Extraction of subwindows in tile %d" % wsi)
						width, height = image.size
						# nb_iter is the number of subwindows we extract in the tile, if predictionstep is 1 we extract all existing subwindows
						nb_iter = ((height - 2 * pyxit_target_height) * (width - 2 * pyxit_target_width)) / (
						predictionstep * predictionstep)

						# pbar = ProgressBar(maxval=nb_iter).start()
						print ("%d subwindows to extract" % nb_iter)
						half_width = math.floor(pyxit_target_width / 2)
						half_height = math.floor(pyxit_target_width / 2)
						# Coordinates of extracted subwindows
						y_roi = range(pyxit_target_height / 2, height - pyxit_target_height / 2, predictionstep)
						x_roi = range(pyxit_target_width / 2, width - pyxit_target_width / 2, predictionstep)

						n_jobs = parameters['nb_jobs']
						n_jobs, _, starts = _partition_images(n_jobs, len(y_roi))

						# Parallel extraction of subwindows in the current tile
						all_data = Parallel(n_jobs = n_jobs)(
							delayed(_parallel_crop_boxes)(
								y_roi[starts[i] :starts[i + 1]],
								x_roi,
								image_filename,
								half_width,
								half_height,
								parameters['pyxit_colorspace'])
							for i in xrange(n_jobs))

						# Reduce
						boxes = np.vstack(boxe for boxe, _ in all_data)
						_X = np.vstack([X for _, X in all_data])

						print ("Before reshape ", _X.shape)
						n_subw = len(_X)
						print("n_subw : ", n_subw)
						_X = np.reshape(_X, (n_subw, pyxit_parameters['pyxit_target_width'], pyxit_parameters['pyxit_target_height'], 3))
						print("After reshape ", _X.shape)
						# Predict subwindow masks
						print("Prediction of subwindows for tile %d " % wsi)
						model_weights_filename = "weights_" + model_name + ".h5"
						_Y = predict(_X, model_weights_filename,
									 imgs_width = pyxit_parameters['pyxit_target_width'],
						  			 imgs_height = pyxit_parameters['pyxit_target_height'],
									 mean = training_sample_mean, std = training_sample_std)
						print("After predict ", _Y.shape)
						print(_Y[0, 0, 0])
						_Y = np.reshape(_Y, (n_subw, pyxit_parameters['pyxit_target_width'], pyxit_parameters['pyxit_target_height']))
						# I want _Y.shape = (n_subw, 32, 32)
						print (len(y_roi))
						print (len(x_roi))
						# I want len(y_roi) * len(x_roi) = n_subw = 1440

						tile_mask = np.zeros((width, height))

						print("_Y.shape : ", _Y.shape)
						print("tile_mask.shape : ", tile_mask.shape)
						for x, y, mask in zip(x_roi, y_roi, _Y):
							min_x = int(x - half_width)
							min_y = int(y - half_height)
							max_x = int(x + half_width)
							max_y = int(y + half_height)
							tile_mask[min_x:max_x, min_y:max_y] += mask

						# Divide by number of overlaps on a pixel
						tile_mask = tile_mask * predictionstep / (pyxit_target_height * pyxit_target_width)
						print("Tile mask : ")
						print(tile_mask)
						# WARNING IF votes_class, change to tile_mask
						# Delete predictions at borders
						print ("Delete borders")
						for i in xrange(0, width) :
							for j in xrange(0, parameters['cytomine_tile_overlap'] / 2) :
								tile_mask[i, j] = 0
							for j in xrange(height - parameters['cytomine_tile_overlap'] / 2, height) :
								tile_mask[i, j] = 0

						for j in xrange(0, height) :
							for i in xrange(0, parameters['cytomine_tile_overlap'] / 2) :
								tile_mask[i, j] = 0
							for i in xrange(width - parameters['cytomine_tile_overlap'] / 2, width) :
								tile_mask[i, j] = 0

						print ("Tile mask shape : ", tile_mask.shape)
						print
						votes = tile_mask * 255


						# pbar.finish()


						# only predict in roi region based on roi mask
						votes[np.logical_not(roi_mask)] = 0

						# process mask
						votes = process_mask(votes)
						print("Tile mask shape after process : ", tile_mask.shape)
						print

						# current time
						print ("TIME : %s" % strftime("%Y-%m-%d %H:%M:%S", localtime()))

						# # Save of confidence map locally
						# print ("Creating output tile file locally")
						# output = Image.fromarray(np.uint8(votes))
						# output_folder = "%s/prediction/output/project-%d/tiles/" %(parameters["cytomine_working_path"], parameters["cytomine_id_project"])
						# if not os.path.exists(output_folder):
						# 	os.makedirs(output_folder)
						# outputfilename = os.path.join(output_folder, "%d-zoom_%d-tile_%d_xxOUTPUT-%dx%d.png" % (id_image, zoom, wsi, pyxit_target_width, pyxit_target_height))
						# output.save(outputfilename, "PNG")
						# print ("Tile OUTPUT file: %s" % outputfilename)
						#
						# # Convert and transfer annotations of current tile
						# print ("Find components")
						# # current time
						# print ("TIME : %s" % strftime("%Y-%m-%d %H:%M:%S", localtime()))

						if parameters['cytomine_mask_internal_holes'] :
							# opencv cv2 new object finder with internal holes:
							votes = votes.astype(np.uint8)
							components = ObjectFinder(np.uint8(votes)).find_components()
							components = whole_slide.convert_to_real_coordinates(whole_slide, components,
																				 reader.window_position, reader.zoom)
							geometries.extend(Utils().get_geometries(components))
						else :
							# opencv old object finder without all internal contours:
							cv.SetData(cv_image, output.tobytes())
							components = ObjectFinder_(cv_image).find_components()
							components = whole_slide.convert_to_real_coordinates_(whole_slide, components,
																				  reader.window_position, reader.zoom)
							geometries.extend(Utils_().get_geometries(components))

						print ("TIME : %s" % strftime("%Y-%m-%d %H:%M:%S", localtime()))

						print ("Uploading annotations...")
						print ("Number of geometries: %d" % len(geometries))
						print ("TIME : %s" % strftime("%Y-%m-%d %H:%M:%S", localtime()))
						start = time.time()
						# print geometries
						print ("------------------------------------------------------------------")
						for geometry in geometries :
							print ("Uploading geometry %s" % geometry)
							startsingle = time.time()
							uploaded = False
							while (not uploaded) :
								try :
									annotation = conn.add_annotation(geometry, id_image)
									uploaded = True
								except socket.timeout, socket.error :
									print("socket timeout/error add_annotation")
									time.sleep(1)
									continue
							endsingle = time.time()
							print ("Elapsed time ADD SINGLE ANNOTATION: %d" % (endsingle - startsingle))

							print (annotation)
							if annotation :
								startsingle = time.time()
								termed = False
								while (not termed) :
									try :
										conn.add_annotation_term(annotation.id, parameters['cytomine_predict_term'],
																 parameters['cytomine_predict_term'], 1.0,
																 annotation_term_model = models.AlgoAnnotationTerm)
										termed = True
									except socket.timeout, socket.error :
										print ("socket timeout/error add_annotation_term")
										time.sleep(1)
										continue
								endsingle = time.time()
								print ("Elapsed time ADD SINGLE ANNOTATION TERM: %d" % (endsingle - startsingle))
						print ("------------------------------------------------------------------")
						# current time
						end = time.time()
						print ("Elapsed time ADD ALL ANNOTATIONS: %d" % (end - start))
						print ("TIME : %s" % strftime("%Y-%m-%d %H:%M:%S", localtime()))
						geometries = []

					# conn.delete_annotation(current_tile_annotation.id)

				else :
					print ("This tile (%05d) is not included in any ROI, so we skip processing" % wsi)

				wsi += 1
				# if wsi<1: reader.next()
				# else: break
				if (not reader.next()) or ((reader.window_position.x > endx) and (reader.window_position.y > endy)) :
					break
				# end of browsing the whole slide





			# Segmentation model was applied on individual tiles. We need to merge geometries generated from each
			# tile.
			# We use a groovy/JTS script that downloads annotation geometries and perform union locally to relieve
			# the Cytomine server
			if parameters['cytomine_union'] :
				print ("In union")
				print ("TIME : %s" % strftime("%Y-%m-%d %H:%M:%S", localtime()))
				print ("Union of polygons for job %d and image %d, term: %d" % (job.userJob, id_image, parameters['cytomine_predict_term']))
				start = time.time()
				host = parameters["cytomine_host"].replace("http://", "")
				unioncommand = "groovy -cp \"../../lib/jars/*\" ../../lib/union4.groovy http://%s %s %s %d %d %d %d %d %d %d %d %d %d" % (
					host,
					user_job.publicKey, user_job.privateKey,
					id_image, job.userJob,
					parameters['cytomine_predict_term'], # union_term,
				parameters['cytomine_union_min_length'], # union_minlength,
				parameters['cytomine_union_bufferoverlap'], # union_bufferoverlap,
				parameters['cytomine_union_min_point_for_simplify'], # union_minPointForSimplify,
				parameters['cytomine_union_min_point'], # union_minPoint,
				parameters['cytomine_union_max_point'], # union_maxPoint,
				parameters['cytomine_union_nb_zones_width'], # union_nbzonesWidth,
				parameters['cytomine_union_nb_zones_height']) # union_nbzonesHeight)
				old_path = os.getcwd()
				os.chdir(current_path)
				print (unioncommand)
				os.system(unioncommand)
				# old version was using a cytomine core webservice for union
				# conn.union_polygons(job.userJob, id_image, union_term, union_minlength, union_area, buffer_length=union_bufferoverlap)
				end = time.time()
				print ("Elapsed time UNION: %d s" % (end - start))
				print ("TIME : %s" % strftime("%Y-%m-%d %H:%M:%S", localtime()))
				os.chdir(old_path)

			# Postprocessing to remove small/large annotations according to min/max area
			if parameters['cytomine_postproc'] :
				print("In post-processing")
				print("POST-PROCESSING BEFORE UNION...")
				print("TIME : %s" % strftime("%Y-%m-%d %H:%M:%S", localtime()))
				retrieved = False
				while (not retrieved) :
					try :
						annotations = conn.get_annotations(id_user = job.userJob, id_image = id_image,
														   id_project = parameters['cytomine_id_project'],
														   showGIS = True)
						retrieved = True
					except socket.timeout, socket.error :
						print("socket timeout/error get_annotations")
						time.sleep(1)
						continue

				# remove/edit useless annotations
				print("TIME : %s" % strftime("%Y-%m-%d %H:%M:%S", localtime()))
				start = time.time()
				for annotation in annotations.data() :
					if annotation.area == 0 :
						conn.delete_annotation(annotation.id)
					else :
						if annotation.area < parameters['cytomine_min_size'] :
							conn.delete_annotation(annotation.id)
						elif annotation.area > parameters['cytomine_max_size'] :
							conn.delete_annotation(annotation.id)
						else :
							print("OK KEEP ANNOTATION %d" % annotation.id)
						# if parameters['cytomine_simplify']:
						#   print "ANNOTATION SIMPLIFICATION"
						#  new_annotation = conn.add_annotation(annotation.location, annotation.image, minPoint=100, maxPoint=500)
						# if new_annotation:
						#   conn.add_annotation_term(new_annotation.id, predict_term, predict_term,
						# 1.0, annotation_term_model = models.AlgoAnnotationTerm)
						#   conn.delete_annotation(annotation.id) #delete old annotation
						# predict_term = parameters['cytomine_predict_term']

				print("TIME : %s" % strftime("%Y-%m-%d %H:%M:%S", localtime()))
				end = time.time()
				print("Elapsed time POST-PROCESS ALL ANNOTATIONS: %d" % (end - start))

			# Perform classification of detected geometries using a classification model (pkl)
			if parameters['pyxit_post_classification'] :
				print ("POSTCLASSIFICATION OF all candidates")
				print ("Create POSTCLASSIFICATION Job and UserJob...")
				conn2 = cytomine.Cytomine(parameters["cytomine_host"],
										  parameters["cytomine_public_key"],
										  parameters["cytomine_private_key"],
										  base_path = parameters['cytomine_base_path'],
										  working_path = parameters['cytomine_working_path'],
										  verbose = True)
				id_software = parameters['cytomine_id_software']
				# create a new userjob related to the classification model
				pc_user_job = conn2.add_user_job(id_software, image_instance.project)
				conn2.set_credentials(str(pc_user_job.publicKey), str(pc_user_job.privateKey))
				pc_job = conn2.get_job(pc_user_job.job)
				pc_job = conn2.update_job_status(pc_job, status_comment = "Create software parameters values...")
				pc_job = conn2.update_job_status(pc_job, status = pc_job.RUNNING, progress = 0,
												 status_comment = "Loading data...")
				# Retrieve locally annotations from Cytomine core produced by the segmentation job as candidates
				candidate_annotations = conn2.get_annotations(id_user = job.userJob, id_image = id_image,
															  id_term = parameters['cytomine_predict_term'],
															  showWKT = True, showMeta = True)
				nb_candidate_annotations = len(candidate_annotations.data())
				folder_name = "%s/slides/project-%d/tiles/crops-candidates-%d-%d/zoom-%d/" % (
				parameters["cytomine_working_path"], parameters["cytomine_id_project"], id_image, job.userJob, 0)
				if not os.path.exists(folder_name) :
					os.makedirs(folder_name)
				annotation_mapping = {}
				for i, annotation in enumerate(candidate_annotations.data()) :
					url = annotation.get_annotation_alpha_crop_url(parameters['cytomine_predict_term'],
																   desired_zoom = 0)
					filename = folder_name + str(annotation.id) + ".png"  # str(i)
					annotation_mapping[annotation.id] = filename
					conn2.fetch_url_into_file(url, filename, False, True)
					np_image = cv2.imread(filename, -1)
					if np_image is not None :
						alpha = np.array(np_image[:, :, 3])
						image = np.array(np_image[:, :, 0 :3])
					# image[alpha == 0] = (255,255,255)  #to replace surrounding by white
					cv2.imwrite(filename, image)
				print ("Building attributes from ", os.path.dirname(os.path.dirname(folder_name)))
				# Extract subwindows from all candidates
				X, y = build_from_dir(os.path.dirname(os.path.dirname(folder_name)))
				post_fp = open(parameters['pyxit_post_classification_save_to'], "r")
				classes = pickle.load(post_fp)
				pyxit = pickle.load(post_fp)
				print (pyxit)
				time.sleep(3)
				# pyxit parameters are in the model file
				y_proba = pyxit.predict_proba(X)
				y_predict = classes.take(np.argmax(y_proba, axis = 1), axis = 0)
				y_rate = np.max(y_proba, axis = 1)
				# We classify each candidate annotation and keep only those predicted as cytomine_predict_term
				for k, annotation in enumerate(candidate_annotations.data()) :
					filename = annotation_mapping[annotation.id]
					j = np.where(X == filename)[0][0]
					if int(y_predict[j]) == parameters['cytomine_predict_term'] :
						print ("POSTCLASSIFICATION Annotation KEPT id: %d class: %d proba: %d" % (
						annotation.id, int(y_predict[j]), y_rate[j]))
					else :
						print ("POSTCLASSIFICATION Annotation REJECTED id: %d class: %d proba: %d" % (
						annotation.id, int(y_predict[j]), y_rate[j]))
					new_annotation = conn2.addAnnotation(annotation.location, id_image)
					conn2.addAnnotationTerm(new_annotation.id, int(y_predict[j]), int(y_predict[j]), y_rate[j],
											annotation_term_model = models.AlgoAnnotationTerm)

				print ("POSTCLASSIFICATION END.")
				print ("TIME : %s" % strftime("%Y-%m-%d %H:%M:%S", localtime()))
				pc_job = conn.update_job_status(pc_job, status = pc_job.TERMINATED, progress = 100,
												status_comment = "Finish Job..")
				# ...


			# Perform stats (counting) in roi area
			if parameters['cytomine_count'] and parameters['cytomine_roi_term'] :
				if parameters['pyxit_post_classification'] :
					id_job = pc_job.userJob
				else :
					id_job = job.userJob
				print ("COUNTING...")
				# Count number of annotations in roi area
				# Get Rois
				roi_annotations = conn.get_annotations(id_image = id_image,
													   id_term = str(parameters['cytomine_roi_term']).replace('[',
																											  '').replace(
														   ']', '').replace(' ', ''),
													   id_project = parameters['cytomine_id_project'])
				# Count included annotations (term = predict_term) in each ROI
				for roi_annotation in roi_annotations.data() :
					included_annotations = conn.included_annotations(id_image = id_image, id_user = id_job,
																	 id_terms = parameters['cytomine_predict_term'],
																	 id_annotation_roi = roi_annotation.id)
					print ("STATSImageID %d name %s: Number of annotations (term: %d) included in ROI %d: %d" % (
					id_image, image_instance.originalFilename, parameters['cytomine_predict_term'], roi_annotation.id,
					len(included_annotations.data())))
					roi_annot_descr = conn.get_annotation(roi_annotation.id)
					print ("STATSImageID %d ROI area: %d" % (id_image, roi_annot_descr.area))

			print ("END image %d." % i)
			break
			progress += progress_delta
			i += 1
		# break : image loop
		# job = conn.update_job_status(job, status = job.TERMINATED, progress = 100, status_comment =  "Finish Job..")
		sys.exit()


if __name__ == "__main__":
	import sys
	main(sys.argv[1:])
