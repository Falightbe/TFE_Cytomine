
from cytomine_sldc import CytomineSlide
from cytomine_sldc import CytomineTileBuilder

from sldc import PolygonClassifier, Segmenter, WorkflowBuilder
import pickle

import os, optparse
import socket

from sklearn.externals.joblib import Parallel, delayed
from shapely.geometry import Point
import numpy as np
import scipy.ndimage
import math

from pyxit.estimator import _get_image_data, _partition_images
from cytomine.models import ImageInstanceCollection
from cytomine import Cytomine
from cytomine_utilities import CytomineJob
from progressbar import *
try:
	import Image, ImageStat
except:
	from PIL import Image, ImageStat


# For parallel extraction of subwindows in current tile
def _parallel_crop_boxes (y_roi, x_roi, image, half_width, half_height, pyxit_colorspace):
	_X = []
	boxes = np.empty((len(x_roi)*len(y_roi), 4),dtype=np.int)
	i = 0

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
	# TODO the question is :
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

# Parameter values are now set through command-line
parameters = {
	'cytomine_host': None,
	'cytomine_public_key' : None,
	'cytomine_private_key' : None,
	'cytomine_base_path' : None,
	'cytomine_working_path' : None,
	'cytomine_id_software' : 0,
	'cytomine_id_project' : 0,
	'cytomine_id_image' : None,
	'cytomine_zoom_level' : 0,
	'cytomine_tile_size' : 512,
	'cytomine_predict_term' : 0,
	'pyxit_predict_step' : 1,
	'pyxit_save_to' : None,
	'pyxit_colorspace' : 2,
	'nb_jobs' : 20,
	'pyxit_min_size' : 0.1,
	'pyxit_max_size' : 1,
	'forest_shared_mem' : True,
	'pyxit_n_jobs' : 10, #threads
	'pyxit_target_width' : 24,  #fixed size in segmentation mode
	'pyxit_target_height' : 24, #fixed size in segmentation mode
	'pyxit_interpolation' : 1, #interpolation used if subwindows are resized
	'pyxit_transpose' : 1, #do we apply rotation/mirroring to subwindows (to enrich training set)
	'pyxit_colorspace' : 2, # which colorspace do we use ?
	'pyxit_fixed_size' : True, #fixed size in segmentation mode
	'pyxit_save_to' : '',
	#classifier parameters
	'forest_n_estimators' : 10, #number of trees
	'forest_max_features' : 28, #number of attributes considered at each node
	'forest_min_samples_split' : 1, #nmin
	'svm' : 0, #no svm in segmentation mode
	'svm_c': 1.0
}


class TumorSegmenter(Segmenter):
	"""This class implements the tumor segmentation procedure"""
	def __init__(self, path_to_pyxit, parameters):
		"""Constructor"""
		fp = open(path_to_pyxit, "r")
		self._classifier = pickle.load(fp)
		self._parameters = parameters

	def segment(self, image):
		########################################
		# Implement the tile segmentation here.
		# The image parameters is the numpy image
		# representing the tile
		#
		# ...
		# ...
		#
		# The mask should be a numpy array with
		# background pixels set to 0 and foreground
		# ones set to 255.
		########################################

		# Parameters
		# TODO image is a ndarray
		print image.shape

		height, width, _ = image.shape
		pyxit_target_width = self._parameters['pyxit_target_width']
		pyxit_target_height = self._parameters['pyxit_target_height']
		n_jobs = self._parameters['nb_jobs']
		prediction_step = parameters['cytomine_predict_step'] # X,Y displacement between two successive subwindows
		nb_iter = ((height - 2 * pyxit_target_height) * (width - 2 * pyxit_target_width)) / (prediction_step * prediction_step) # number of subwindows we extract in the tile
		half_width = math.floor(pyxit_target_width / 2)
		half_height = math.floor(pyxit_target_width / 2)

		# Coordinates of extracted subwindows
		y_roi = range(pyxit_target_height / 2, height - pyxit_target_height / 2, prediction_step)
		x_roi = range(pyxit_target_width / 2, width - pyxit_target_width / 2, prediction_step)

		n_jobs, _, starts = _partition_images(n_jobs, len(y_roi))

		# Parallel extraction of subwindows in the current tile
		print "Parallel extraction of subwindows in the tile"
		all_data = Parallel(n_jobs = n_jobs)(
			delayed(_parallel_crop_boxes)(
				y_roi[starts[i] :starts[i + 1]],
				x_roi,
				image,
				half_width,
				half_height,
				parameters['pyxit_colorspace'])
			for i in xrange(n_jobs))

		# Reduce
		boxes = np.vstack(boxe for boxe, _ in all_data)
		_X = np.vstack([X for _, X in all_data])

		# Classify tile subwindow pixels
		_Y = self._classifier.predict(_X)

		# Parallel construction of confidence map in tile
		print "Parallel construction of confidence map in tile"
		pixels = range(pyxit_target_width * pyxit_target_height)
		n_jobs, _, starts = _partition_images(n_jobs, len(pixels))

		all_votes_class = Parallel(n_jobs = n_jobs)(
			delayed(_parallel_confidence_map)(
				pixels[starts[i] :starts[i + 1]],
				_Y[starts[i] :starts[i + 1]],
				starts[i],
				boxes,
				width,
				height,
				self._classifier.n_classes_[0],
				pyxit_target_width,
				pyxit_target_height)
			for i in xrange(n_jobs))

		votes_class = all_votes_class[0]

		for v in all_votes_class[1 :] :
			votes_class += v

		# Delete predictions at borders
		print "Delete borders"
		for i in xrange(0, width) :
			for j in xrange(0, pyxit_target_height / 2) :
				votes_class[i, j, :] = [1, 0]
			for j in xrange(height - pyxit_target_height / 2, height) :
				votes_class[i, j, :] = [1, 0]

		for j in xrange(0, height) :
			for i in xrange(0, pyxit_target_width / 2) :
				votes_class[i, j, :] = [1, 0]
			for i in xrange(width - pyxit_target_width / 2, width) :
				votes_class[i, j, :] = [1, 0]

		votes = np.argmax(votes_class, axis = 2) * 255

		# Predict in roi region based on roi mask TODO
		# votes[np.logical_not(roi_mask)] = 0

		# Process mask
		mask = process_mask(votes)
		return mask


class TumorClassifier(PolygonClassifier):
	"""A classifier that always predict the class 1 and the probability 1.0 for a polygon"""
	def __init__(self, path_to_pyxit):
		"""Tumor classifier's constructor"""
		# Fetch classifier
		fp = open(path_to_pyxit, "r")
		self._classifier = pickle.load(fp)

	def predict(self, image, polygon):
		# Crop the image to fit the polygon
		original = np.array(image)
		crop = np.array(image)

		for i, pixel in np.ndenumerate(original) :
			y, x, z = i

			point = Point(y, x)

			if not polygon.contains(point):
				crop[(y, x, 2)] = 0

		y_proba = self._classifier.predict(crop)
		y_predict = self._classifier.classes_.take(np.argmax(y_proba, axis=1), axis=0)
		return y_predict, y_proba


class TumorJob(CytomineJob):
	"""This class implements the initialization of the workflow and the processing"""
	def __init__(self, cytomine, software_id, project_id, parameters):
		"""Tumor job's constructor"""
		super(TumorJob, self).__init__(cytomine, software_id, project_id, parameters = parameters)

		builder = WorkflowBuilder()

		# Add the workflow components and set the workflow parameters
		builder.set_n_jobs(parameters["nb_jobs"])
		# builder.set_parallel_dc()
		builder.set_segmenter(TumorSegmenter(parameters["pyxit_segmenter_save_to"], parameters))
		builder.set_tile_size(parameters["cytomine_tile_size"], parameters["cytomine_tile_size"])
		# builder.set_overlap() # If not called, an overlap of 5 is provided by default.
		# builder.set_distance_tolerance() # If not called, a thickness of 7 is provided by default.
		builder.set_tile_builder(CytomineTileBuilder(cytomine, parameters["cytomine_working_path"]))
		builder.add_catchall_classifier(TumorClassifier(parameters["pyxit_classifier_save_to"]))

		self._workflow = builder.get()

	def process(self, slide):
		"""Process one slide using the workflow"""
		results = self._workflow.process(slide)

		for polygon, dispatch, cls, proba in results:
			# Do something with the detected polygons (upload, save on disk,... ?)
			# ...
			# TODO : line 684 -> 812
			uploaded = False
			while not uploaded:
				try :
					annotation = self.__cytomine.add_annotation(polygon, id_image)
					uploaded = True
				except socket.timeout, socket.error :
					print "socket timeout/error add_annotation"
					time.sleep(1)
					continue
			endsingle = time.time()
			print "Elapsed time ADD SINGLE ANNOTATION: %d" % (endsingle - startsingle)

			if annotation :
				startsingle = time.time()
				termed = False
				while not termed :
					try :
						self.__cytomine.add_annotation_term(annotation.id, parameters['cytomine_predict_term'], parameters['cytomine_predict_term'], 1.0, annotation_term_model = models.AlgoAnnotationTerm)
						termed = True
					except socket.timeout, socket.error :
						print "socket timeout/error add_annotation_term"
						time.sleep(1)
						continue
				endsingle = time.time()
				print "Elapsed time ADD SINGLE ANNOTATION TERM: %d" % (endsingle - startsingle)

			pass


def main(argv):
	"""
	Parameters
	----------
	argv: list
	Script's parameters list (without executable name)
	"""
	# Parse the script parameters
	current_path = os.getcwd() + '/' + os.path.dirname(__file__)
	# Define command line options
	print "Main function"
	p = optparse.OptionParser(description='Cytomine Segmentation prediction', prog='Cytomine segmentation prediction',
							  version='0.1')
	p.add_option('--cytomine_host', type="string", default='beta.cytomine.be', dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
	p.add_option('--cytomine_public_key', type="string", default='', dest="cytomine_public_key", help="Cytomine public key")
	p.add_option('--cytomine_private_key', type="string", default='', dest="cytomine_private_key", help="Cytomine private key")
	p.add_option('--cytomine_base_path', type="string", default='/api/', dest="cytomine_base_path", help="Cytomine base path")
	p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")
	p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path", help="The working directory (eg: /tmp)")
	p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")

	p.add_option('-i', '--cytomine_id_image', type='int', dest='cytomine_id_image', help="image id from cytomine", metavar='IMAGE')

	p.add_option('-t', '--cytomine_tile_size', type='int', dest='cytomine_tile_size', help="sliding tile size")
	p.add_option('-j', '--pyxit_nb_jobs', type='int', dest='pyxit_nb_jobs', help="number of parallel jobs")
	p.add_option('--cytomine_predict_term', type='int', dest='cytomine_predict_term', help="term id of predicted term (binary mode)")
	p.add_option('--cytomine_predict_step', type='int', dest='cytomine_predict_step', help="pyxit step between successive subwindows")
	p.add_option('--pyxit_segmenter_save_to', type='string', dest='pyxit_segmenter_save_to', help="pyxit segmentation model file")  # future: get it from server db
	p.add_option('--pyxit_classifier_save_to', type='string', dest='classifier_save_to', help="pyxit post classification model file")  # future: get it from server db
	p.add_option('--pyxit_colorspace', type='int', dest='pyxit_colorspace', help="pyxit colorspace encoding")  # future: get it from server db
	p.add_option('--pyxit_target_width', type = 'int', dest = 'pyxit_target_width', help = "pyxit subwindows width")
	p.add_option('--pyxit_target_height', type = 'int', dest = 'pyxit_target_height', help = "pyxit subwindows height")
	p.add_option('--verbose', type='string', default="0", dest="verbose", help="Turn on (1) or off (0) verbose mode")

	options, arguments = p.parse_args(args=argv)

	# TODO : parameters initialisation
	parameters['cytomine_host'] = options.cytomine_host
	parameters['cytomine_public_key'] = options.cytomine_public_key
	parameters['cytomine_private_key'] = options.cytomine_private_key
	parameters['cytomine_base_path'] = options.cytomine_base_path
	parameters['cytomine_working_path'] = options.cytomine_working_path
	parameters['cytomine_base_path'] = options.cytomine_base_path

	parameters['cytomine_id_project'] = options.cytomine_id_project
	parameters['cytomine_id_software'] = options.cytomine_id_software
	parameters['cytomine_predict_term'] = options.cytomine_predict_term
	parameters['cytomine_predict_step'] = options.cytomine_predict_step
	parameters['pyxit_segmenter_save_to'] = options.pyxit_segmenter_save_to
	parameters['pyxit_classifier_save_to'] = options.classifier_save_to
	parameters['pyxit_colorspace'] = options.pyxit_colorspace
	parameters['pyxit_nb_jobs'] = options.pyxit_nb_jobs
	parameters['cytomine_nb_jobs'] = options.pyxit_nb_jobs
	parameters['cytomine_id_image'] = options.cytomine_id_image
	parameters['nb_jobs'] = options.pyxit_nb_jobs
	parameters['pyxit_target_width'] = options.pyxit_target_width
	parameters['pyxit_target_height'] = options.pyxit_target_height
	parameters['pyxit_n_subwindows'] = 100
	parameters['dir_ls'] = os.path.join(parameters["cytomine_working_path"], "prediction")

	print parameters

	client = Cytomine(parameters["cytomine_host"],
					parameters["cytomine_public_key"],
					parameters["cytomine_private_key"],
					base_path = parameters['cytomine_base_path'],
					working_path = parameters['cytomine_working_path'],
					verbose= True)

	# Fetch image collection
	image_instances = ImageInstanceCollection()
	image_instances.project = parameters['cytomine_id_project']
	image_instances = client.fetch(image_instances)
	images = image_instances.data()

	# Launch the processing
	with TumorJob(client, parameters['cytomine_id_software'], parameters['cytomine_id_project'], parameters) as job:
		for image in images :
			slide = CytomineSlide(client, image.id)
			job.process(slide)


if __name__ == "__main__":
	import sys
	main(sys.argv[1:])