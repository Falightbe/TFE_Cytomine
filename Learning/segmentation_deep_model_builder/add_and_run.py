from __future__ import print_function

from skimage.transform import resize
from keras.models import Model as Model_keras
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
try:
	import Image, ImageStat
except:
	from PIL import Image, ImageStat

import os, optparse
from pyxit.estimator import PyxitClassifier, _get_output_from_mask

import shapely
from shapely.geometry.polygon import Polygon
import shapely.wkt
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)
from matplotlib.path import Path
import scipy.ndimage

import cytomine
from pyxit.data import build_from_dir
from pyxit.estimator import _get_image_data
from cytomine.models import Annotation
from cytomine import cytomine
n_channels = 3

K.set_image_data_format('channels_last')  # TF dimension ordering in this code


smooth = 1.


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
	inputs = Input((imgs_width, imgs_height, n_channels))
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
	pool5 = MaxPooling2D(pool_size = (2, 2))(conv5)

	conv6 = Conv2D(1024, (3, 3), activation = 'relu', padding = 'same')(pool5)
	conv6 = Conv2D(1024, (3, 3), activation = 'relu', padding = 'same')(conv6)

	up7 = concatenate([Conv2DTranspose(512, (2, 2), strides = (2, 2), padding = 'same')(conv6), conv5], axis = 3)
	conv7 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(up7)
	conv7 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(conv7)

	up8 = concatenate([Conv2DTranspose(256, (2, 2), strides = (2, 2), padding = 'same')(conv7), conv4], axis = 3)
	conv8 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(up8)
	conv8 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(conv8)

	up9 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv8), conv3], axis=3)
	conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(up9)
	conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv9)

	up10 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv9), conv2], axis=3)
	conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(up10)
	conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv10)

	up11 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv10), conv1], axis=3)
	conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(up11)
	conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv11)

	conv12 = Conv2D(1, (1, 1), activation='sigmoid')(conv11)

	model = Model_keras(inputs=[inputs], outputs=[conv12])

	model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=['accuracy', dice_coef])

	return model


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
	imgs_train = imgs_train.astype('float32')
	# mean = np.mean(imgs_train)  # mean for data centering
	# std = np.std(imgs_train)  # std for data normalization

	# imgs_train -= mean
	# imgs_train /= std
	imgs_mask_train = imgs_mask_train.astype('float32')

	# Creating and compiling model
	model = get_unet(imgs_width, imgs_height)
	model_checkpoint = ModelCheckpoint(model_weights_filename, monitor='val_loss', save_best_only=True)

	# Fitting model
	print('Fitting model...')
	n_samples = len(imgs_train)
	print(imgs_train.shape)
	print(imgs_mask_train.shape)
	imgs_mask_train = np.reshape(imgs_mask_train, (n_samples, imgs_width, imgs_height, 1))

	model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=shuffle,
			  validation_split=validation_split,
			  callbacks=[model_checkpoint])

	return mean, std

def load_model(model_weights_filename, imgs_width, imgs_height):
	model = get_unet(imgs_width, imgs_height)
	model_checkpoint = ModelCheckpoint(model_weights_filename, monitor = 'val_loss', save_best_only = True)
	model.load_weights(model_weights_filename)
	return model


def predict(imgs_test, model, mean, std):
	imgs_test = imgs_test.astype('float32')
	imgs_test -= mean
	imgs_test /= std

	# Predicting masks on test data
	print('Predicting masks on test data...')
	imgs_mask_test = model.predict(imgs_test, verbose=1)
	return imgs_mask_test


def stats_dumped_annotations(positive_dir, negative_dir):
	positive_annotation_height_list = []
	positive_annotation_width_list = []
	negative_annotation_height_list = []
	negative_annotation_width_list = []
	n_cannot_open = 0

	print("In stats_dumped_annotations")
	stats = open("annotation_stats.csv", 'w')

	for image_file in os.listdir(positive_dir) :
		try:
			im = Image.open(os.path.join(positive_dir, image_file))
		except:
			n_cannot_open += 1
			continue
		annot_width, annot_height = im.size
		if annot_width < 64 or annot_height < 64 :
			print("Small positive image (%d, %d) : %s" % (annot_height, annot_width, image_file))
		positive_annotation_width_list.append(annot_width)
		positive_annotation_height_list.append(annot_height)
		im.close()

	for image_file in os.listdir(negative_dir) :
		try:
			im = Image.open(os.path.join(negative_dir, image_file))
		except:
			n_cannot_open += 1
			continue
		annot_width, annot_height = im.size
		if annot_width < 64 or annot_height < 64 :
			print("Small negative image (%d, %d) : %s" %(annot_height, annot_width, image_file))
		negative_annotation_width_list.append(annot_width)
		negative_annotation_height_list.append(annot_height)
		im.close()

	d_positive = {'Height' : positive_annotation_height_list, 'Width' : positive_annotation_width_list}
	d_negative = {'Height' : negative_annotation_height_list, 'Width' : negative_annotation_width_list}
	positive_df = pd.DataFrame(data = d_positive)
	negative_df = pd.DataFrame(data = d_negative)

	print("Dumped annotations statistics : ")
	print("Cannot open %d images" % n_cannot_open)
	print("\nPositive annotations : ")
	print(str(positive_df.describe()))
	print("\nNegative annotations : ")
	print(str(negative_df.describe()))


	# plt.scatter(positive_annotation_height_list, positive_annotation_width_list, color = 'r', label = 'Positive annotations')
	# plt.scatter(negative_annotation_height_list, negative_annotation_width_list, color = 'b', label = 'Negative annotations')
	# plt.xlabel('Height')
	# plt.ylabel('Width')
	# ax.set_yscale('log')
	# ax.set_xscale('log')

	# Plot
	# plt.figure()
	# ax = plt.gca()
	# ax.scatter(positive_annotation_height_list, positive_annotation_width_list, c = 'r', s = 1, alpha = 0.6, label = 'Positive annotations')
	# ax.scatter(negative_annotation_height_list, negative_annotation_width_list, c = 'b', s = 1, alpha = 0.6, label = 'Negative annotations')
	# ax.set_yscale('log')
	# ax.set_xscale('log')
	# legend = plt.legend(loc = 'lower right', framealpha = 0.1)
	# plt.show()


def main(argv):
	current_path = os.getcwd() + '/' + os.path.dirname(__file__)

	# Define command line options
	p = optparse.OptionParser(description='Cytomine Segmentation prediction', prog='Cytomine segmentation prediction', version='0.1')

	p.add_option('--cytomine_host', type="string", default = 'beta.cytomine.be', dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
	p.add_option('--cytomine_public_key', type="string", default = '', dest="cytomine_public_key", help="Cytomine public key")
	p.add_option('--cytomine_private_key',type="string", default = '', dest="cytomine_private_key", help="Cytomine private key")
	p.add_option('--cytomine_base_path', type="string", default = '/api/', dest="cytomine_base_path", help="Cytomine base path")
	p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")
	p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path", help="The working directory (eg: /tmp)")
	p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")

	p.add_option('-i', '--cytomine_id_image', type='int', dest='cytomine_id_image', help="image id from cytomine", metavar='IMAGE')
	p.add_option('-z', '--cytomine_zoom_level', type='int', dest='cytomine_zoom_level', help="working zoom level")
	p.add_option('-j', '--nb_jobs', type='int', dest='nb_jobs', help="number of parallel jobs")
	p.add_option('--cytomine_predict_terms', type='str', dest='cytomine_predict_terms', help="term id of all positive terms. The first term is the output predicted annotation term")
	p.add_option('--cytomine_excluded_terms', type='string', dest='cytomine_excluded_terms', help="term id of excluded terms)")

	p.add_option('--pyxit_target_width', type='int', dest='pyxit_target_width', help="pyxit subwindows width")
	p.add_option('--pyxit_target_height', type='int', dest='pyxit_target_height', help="pyxit subwindows height")
	p.add_option('--pyxit_colorspace', type='int', dest='pyxit_colorspace', help="pyxit colorspace encoding")
	p.add_option('--pyxit_nb_jobs', type='int', dest='pyxit_nb_jobs', help="pyxit number of jobs for trees")
	p.add_option('--pyxit_fixed_size', type = 'string', default = "0", dest = "pyxit_fixed_size", help = "extract fixed size subwindows")
	p.add_option('--pyxit_transpose', type = 'string', default = "0", dest = "pyxit_transpose", help = "transpose subwindows")
	p.add_option('--pyxit_n_subwindows', type='int', default="10", dest="pyxit_n_subwindows", help="number of subwindows")
	p.add_option('--pyxit_interpolation', default = 2, type = "int", dest = "pyxit_interpolation", help = "interpolation method 1,2,3,4")
	p.add_option('--pyxit_min_size', default = 0.5, type = "float", dest = "pyxit_min_size", help = "min size")
	p.add_option('--pyxit_max_size', default = 1.0, type = "float", dest = "pyxit_max_size", help = "max size")
	p.add_option('--cytomine_reviewed', type = 'string', default = "False", dest = "cytomine_reviewed", help = "Get reviewed annotations only")

	p.add_option('--cytomine_dump_annotations', type = 'string', default = "0", dest = "cytomine_dump_annotations", help = "Dump training annotations or not")
	p.add_option('--cytomine_dump_annotation_stats', type = 'string', default = "0", dest = "cytomine_dump_annotation_stats", help = "Calculate stats on dumped annotations or not")
	p.add_option('--build_model', type = "string", default = "0", dest = "build_model", help = "Turn on (1) or off (0) model building")
	p.add_option('--cytomine_annotation_projects', type = "string", dest = "cytomine_annotation_projects", help = "Projects from which annotations are extracted")
	p.add_option('--verbose', type = "string", default = "0", dest = "verbose", help = "Turn on (1) or off (0) verbose mode")

	p.add_option('--keras_save_to', type = 'string', default = "", dest = 'keras_save_to', help = "keras model weight file")
	p.add_option('--keras_batch_size', type = "int", dest = "keras_batch_size", help = "Training batch size")
	p.add_option('--keras_n_epochs', type = "int", dest = "keras_n_epochs", help = "Number of epochs")
	p.add_option('--keras_shuffle', type = "string", dest = "keras_shuffle", help = "Turn on (1) or off (0) batch shuffle")
	p.add_option('--keras_validation_split', type = "float", dest = "keras_validation_split", help = "Batch validation split")
	options, arguments = p.parse_args( args = argv)


	parameters = {}
	parameters['keras_save_to'] = options.keras_save_to
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
	parameters['cytomine_predicted_annotation_term'] = parameters['cytomine_predict_terms'][0]
	parameters['cytomine_excluded_terms'] = map(int,options.cytomine_excluded_terms.split(','))

	parameters['pyxit_colorspace'] = options.pyxit_colorspace
	parameters['pyxit_nb_jobs'] = options.pyxit_nb_jobs
	parameters['pyxit_n_jobs'] = options.pyxit_nb_jobs
	parameters['cytomine_nb_jobs'] = options.pyxit_nb_jobs
	parameters['cytomine_id_image'] = options.cytomine_id_image
	parameters['cytomine_zoom_level'] = options.cytomine_zoom_level
	parameters['nb_jobs'] = options.nb_jobs
	parameters['pyxit_target_width'] = options.pyxit_target_width
	parameters['pyxit_target_height'] = options.pyxit_target_height
	parameters['pyxit_n_subwindows'] = options.pyxit_n_subwindows
	parameters['pyxit_interpolation'] = options.pyxit_interpolation
	parameters['pyxit_transpose'] = str2bool(options.pyxit_transpose)
	parameters['pyxit_min_size'] = options.pyxit_min_size
	parameters['pyxit_max_size'] = options.pyxit_max_size
	parameters['pyxit_fixed_size'] = str2bool(options.pyxit_fixed_size)
	parameters['cytomine_annotation_projects'] = map(int, options.cytomine_annotation_projects.split(','))
	parameters['cytomine_reviewed'] = str2bool(options.cytomine_reviewed)
	parameters['cytomine_dump_annotation_stats'] = str2bool(options.cytomine_dump_annotation_stats)
	parameters['cytomine_dump_annotations'] = str2bool(options.cytomine_dump_annotations)
	parameters['build_model'] = str2bool(options.build_model)
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

	if options.verbose :
		print(parameters)

	# Create Cytomine connection
	conn = cytomine.Cytomine(parameters["cytomine_host"],
							 parameters["cytomine_public_key"],
							 parameters["cytomine_private_key"],
							 base_path = parameters['cytomine_base_path'],
							 working_path = parameters['cytomine_working_path'],
							 verbose = str2bool(options.verbose))

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
							  excluded_terms = parameters['cytomine_excluded_terms'])


		# Put positive terms under the same term and same for negative terms
		term_directories = os.listdir(parameters['dir_ls'])
		pos_image_path = os.path.join(parameters['dir_ls'], "image", "1")
		pos_mask_path = os.path.join(parameters['dir_ls'], "mask", "1")
		neg_image_path = os.path.join(parameters['dir_ls'], "image", "0")
		neg_mask_path = os.path.join(parameters['dir_ls'], "mask", "0")
		if not os.path.exists(pos_image_path) :
			print("Creating positive annotation directory: %s" % pos_image_path)
			os.makedirs(pos_image_path)
		if not os.path.exists(neg_image_path) :
			print("Creating negative annotation directory: %s" % neg_image_path)
			os.makedirs(neg_image_path)

		for dir in term_directories :
			dir_abs = os.path.join(parameters['dir_ls'], dir)

			# Move files
			if int(dir) in parameters['cytomine_predict_terms'] :
				for image_file in os.listdir(dir_abs) :
					os.rename(os.path.join(dir_abs, image_file), os.path.join(pos_image_path, image_file))
					image = Image.open(os.path.join(pos_image_path, image_file))
					alpha = image.getdata(band = 3)
					mask = Image.frombytes("1", image.size, alpha)
					mask.save(os.path.join(pos_mask_path, image_file),"PNG")

			else:
				for image_file in os.listdir(dir_abs) :
					os.rename(os.path.join(dir_abs, image_file), os.path.join(neg_image_path, image_file))
					image = Image.open(os.path.join(neg_image_path, image_file))
					alpha = image.getdata(band = 3)
					mask = Image.frombytes("1", image.size, alpha)
					mask.save(os.path.join(neg_mask_path, image_file),"PNG")

			# Remove empty directory
			if int(dir) != 0 and int(dir) != 1:
				os.rmdir(dir_abs)

	if parameters['cytomine_dump_annotation_stats'] :
		pos_path = os.path.join(parameters['dir_ls'], "1")
		neg_path = os.path.join(parameters['dir_ls'], "0")
		stats_dumped_annotations(pos_path, neg_path)

	# if parameters['build_model'] :
	# 	# Model name
	# 	model_name = "nsubw{}_winsize{}x{}_minsize{}_maxsize{}_batchsize{}_epochs{}_shuffle{}_valsplit{}_colorspace{}"\
	# 		.format(parameters['pyxit_n_subwindows'],
	# 				parameters['pyxit_target_width'],
	# 				parameters['pyxit_target_height'],
	# 				parameters['pyxit_min_size'],
	# 				parameters['pyxit_max_size'],
	# 				parameters['keras_batch_size'],
	# 				parameters['keras_n_epochs'],
	# 				parameters['keras_shuffle'],
	# 				parameters['keras_validation_split'],
	# 				pyxit_parameters['pyxit_colorspace']).replace(".", "")
	# 	print("Model_name :", model_name)
	#
	# 	pyxit = PyxitClassifier(None,
	# 							n_subwindows = pyxit_parameters['pyxit_n_subwindows'],
	# 							min_size = pyxit_parameters['pyxit_min_size'],
	# 							max_size = pyxit_parameters['pyxit_max_size'],
	# 							target_width = pyxit_parameters['pyxit_target_width'],
	# 							target_height = pyxit_parameters['pyxit_target_height'],
	# 							n_jobs = pyxit_parameters['pyxit_n_jobs'],
	# 							interpolation = pyxit_parameters['pyxit_interpolation'],
	# 							transpose = pyxit_parameters['pyxit_transpose'],
	# 							colorspace = pyxit_parameters['pyxit_colorspace'],
	# 							fixed_size = pyxit_parameters['pyxit_fixed_size'],
	# 							random_state = None,
	# 							verbose = 1,
	# 							get_output = _get_output_from_mask,
	# 							parallel_leaf_transform = False)
	#
	# 	# Build filenames and classes
	# 	X, y = build_from_dir(parameters['dir_ls'])
	#
	# 	classes = np.unique(y)
	# 	n_classes = len(classes)
	# 	y_original = y
	# 	y = np.searchsorted(classes, y)
	# 	n_images = len(y)
	# 	print("Number of images : ", n_images)
	#
	# 	# Extract subwindows
	# 	_X, _y = pyxit.extract_subwindows(X, y)
	# 	n_subw = len(_y)
	# 	print("Number of subwindows : ", n_subw)
	#
	# 	# Reshape data structure
	# 	_X = np.reshape(_X, (
	# 	n_subw, pyxit_parameters['pyxit_target_width'], pyxit_parameters['pyxit_target_height'], n_channels))
	# 	_y = np.reshape(_y, (n_subw, pyxit_parameters['pyxit_target_width'], pyxit_parameters['pyxit_target_height']))
	#
	# 	# Train FCN
	# 	if not os.path.exists(parameters['keras_save_to']) :
	# 		os.makedirs(parameters['keras_save_to'])
	#
	# 	model_weights_file_path = os.path.join(parameters['keras_save_to'], "weights_" + model_name + ".h5")
	#
	# 	mean, std = train(_X, _y,
	# 					  model_weights_file_path,
	# 					  imgs_width = pyxit_parameters['pyxit_target_width'],
	# 					  imgs_height = pyxit_parameters['pyxit_target_height'],
	# 					  batch_size = parameters['keras_batch_size'],
	# 					  epochs = parameters['keras_n_epochs'],
	# 					  shuffle = parameters['keras_shuffle'],
	# 					  validation_split = parameters['keras_validation_split'])
	#
	# 	# Save mean and std used to normalize training data
	# 	mean_std_save_file_path = os.path.join(parameters['keras_save_to'], "meanstd_" + model_name + ".txt")
	# 	mean_std_save_file = open(mean_std_save_file_path, 'w')
	# 	mean_std_save_file.write(str(mean) + '\n')
	# 	mean_std_save_file.write(str(std) + '\n')


	if parameters['build_model'] :
		# Model name
		model_name = "nsubw{}_winsize{}x{}_minsize{}_maxsize{}_batchsize{}_epochs{}_shuffle{}_valsplit{}_colorspace{}_zoom{}_until4x4_IDG"\
			.format(parameters['pyxit_n_subwindows'],
					parameters['pyxit_target_width'],
					parameters['pyxit_target_height'],
					parameters['pyxit_min_size'],
					parameters['pyxit_max_size'],
					parameters['keras_batch_size'],
					parameters['keras_n_epochs'],
					parameters['keras_shuffle'],
					parameters['keras_validation_split'],
					pyxit_parameters['pyxit_colorspace'],
					parameters['cytomine_zoom_level']).replace(".", "")
		print("Model_name :", model_name)

		pyxit = PyxitClassifier(None,
								n_subwindows = 1,
								min_size = 1,
								max_size = 1,
								target_width = pyxit_parameters['pyxit_target_width'],
								target_height = pyxit_parameters['pyxit_target_height'],
								n_jobs = pyxit_parameters['pyxit_n_jobs'],
								interpolation = pyxit_parameters['pyxit_interpolation'],
								transpose = pyxit_parameters['pyxit_transpose'],
								colorspace = pyxit_parameters['pyxit_colorspace'],
								fixed_size = pyxit_parameters['pyxit_fixed_size'],
								random_state = None,
								verbose = 1,
								get_output = _get_output_from_mask,
								parallel_leaf_transform = False)
		# pyxit = PyxitClassifier(None,
		# 						n_subwindows=pyxit_parameters['pyxit_n_subwindows'],
		# 						min_size=pyxit_parameters['pyxit_min_size'],
		# 						max_size=pyxit_parameters['pyxit_max_size'],
		# 						target_width=pyxit_parameters['pyxit_target_width'],
		# 						target_height=pyxit_parameters['pyxit_target_height'],
		# 						n_jobs=pyxit_parameters['pyxit_n_jobs'],
		# 						interpolation=pyxit_parameters['pyxit_interpolation'],
		# 						transpose=pyxit_parameters['pyxit_transpose'],
		# 						colorspace=pyxit_parameters['pyxit_colorspace'],
		# 						fixed_size=pyxit_parameters['pyxit_fixed_size'],
		# 						random_state=None,
		# 						verbose=1,
		# 						get_output = _get_output_from_mask,
		# 						parallel_leaf_transform=False)

		# Build filenames and classes
		X, y = build_from_dir(parameters['dir_ls'])

		classes = np.unique(y)
		n_classes = len(classes)
		y_original = y
		y = np.searchsorted(classes, y)
		n_images = len(y)
		print("Number of images : ", n_images)
		print("Start extraction of subwindows...")

		# Extract subwindows
		_X, _y = pyxit.extract_subwindows(X, y)
		print("Over")
		n_subw = len(_y)
		print("Number of subwindows : ", n_subw)

		# Reshape data structure
		_X = np.reshape(_X, (n_subw, pyxit_parameters['pyxit_target_width'], pyxit_parameters['pyxit_target_height'], n_channels))
		_y = np.reshape(_y, (n_subw, pyxit_parameters['pyxit_target_width'], pyxit_parameters['pyxit_target_height'], 1))
		print(type(_X))
		print(type(_y))

		# ImageDataGenerator :  two instances with the same arguments
		data_gen_args = dict(rotation_range = 180.,
							 width_shift_range = 0.1,
							 height_shift_range = 0.1,
							 zoom_range = 0.2,
							 rescale = 1 / 255,
							 horizontal_flip = True,
							 vertical_flip = True)
		# featurewise_center = True,
		#  featurewise_std_normalization = True)

		image_datagen = ImageDataGenerator(**data_gen_args)
		mask_datagen = ImageDataGenerator(**data_gen_args)

		# Provide the same seed and keyword arguments to the fit and flow methods
		seed = 1
		# print("Fit image data generator (image)...")
		# image_datagen.fit(_X[0:10], augment = True, seed = seed)
		# print("Fit image data generator (mask)...")
		# mask_datagen.fit(_y[0:10], augment = True, seed = seed)
		# labels = np.ones((n_subw, 1)
		print(type(_X))
		print(type(_y))
		# print(type(labels))

		print('Flow on images...')
		# image_generator = image_datagen.flow(_X, labels, seed = seed, shuffle = False)
		image_generator = image_datagen.flow_from_directory(
			os.path.join(parameters['dir_ls'], "image"),
			class_mode = None,
			target_size = (128, 128),
			seed = seed)
		print('Flow on masks...')
		# mask_generator = mask_datagen.flow(_y, labels, seed = seed, shuffle = False)
		mask_generator = mask_datagen.flow_from_directory(
			os.path.join(parameters['dir_ls'], "mask"),
			class_mode = None,
			target_size = (128, 128),
			seed = seed)

		# combine generators into one which yields image and masks
		train_generator = zip(image_generator, mask_generator)

		# Creating and compiling model
		if not os.path.exists(parameters['keras_save_to']) :
			os.makedirs(parameters['keras_save_to'])

		model_weights_filename = os.path.join(parameters['keras_save_to'], "weights_" + model_name + ".h5")
		print('Fitting model...')
		model = get_unet()
		model_checkpoint = ModelCheckpoint(model_weights_filename, monitor = 'val_loss', save_best_only = True)

		# Train FCN
		model.fit_generator(train_generator, steps_per_epoch = 100, epochs = 50, callbacks = [model_checkpoint],
							verbose = 1)



if __name__ == "__main__":
	import sys
	main(sys.argv[1:])
