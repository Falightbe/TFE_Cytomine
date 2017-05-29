# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2016. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */


__author__ = "Marée Raphael <raphael.maree@ulg.ac.be>"
__contributors__ = ["Stévens Benjamin <b.stevens@ulg.ac.be>"]
__copyright__ = "Copyright 2010-2016 University of Liège, Belgium, http://www.cytomine.be/"

import cytomine
import os, optparse
from cytomine.models import *
import cPickle as pickle
import numpy as np
import time
from datetime import datetime
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
import math

from pyxit import pyxitstandalone
from cytomine.utils import parameters_values_to_argv




def str2bool(v) :
	return v.lower() in ("yes", "true", "t", "1")


def main(argv) :
	# Parameter values are set through command-line, see test-train.sh
	parameters = {
		'cytomine_host' : None,
		'cytomine_public_key' : None,
		'cytomine_private_key' : None,
		'cytomine_base_path' : None,
		'cytomine_working_path' : '/home/maree/tmp/cytomine/annotations/',
		'cytomine_id_software' : 1,
		'cytomine_id_project' : 1,
		'cytomine_zoom_level' : 1,
		'cytomine_dump_type' : 1,
		'cytomine_annotation_projects' : [1], # id of projets from which we dump annotations for learning
		'cytomine_predict_terms' : [1], #
		'cytomine_excluded_terms' : [2, 3], # exclude these term ids
		'cytomine_reviewed' : True
	}

	# Parameter values are set through command-line, see test-train.sh
	pyxit_parameters = {
		'dir_ls' : "/",
		# 'dir_ts' : "/",
		'forest_shared_mem' : False,
		# processing
		'pyxit_n_jobs' : 10,
		# subwindows extraction
		'pyxit_n_subwindows' : 100,
		'pyxit_min_size' : 0.1,
		'pyxit_max_size' : 1.0,
		'pyxit_target_width' : 24, # 24x24
		'pyxit_target_height' : 24,
		'pyxit_interpolation' : 1,
		'pyxit_transpose' : 1, # do we apply rotation/mirroring to subwindows (to enrich training set)
		'pyxit_colorspace' : 2, # which colorspace do we use ?
		'pyxit_fixed_size' : False, # do we extracted fixed sizes or random sizes (false)
		# classifier parameters
		'forest_n_estimators' : 10, # number of trees
		'forest_max_features' : 28, # number of attributes considered at each node
		'forest_min_samples_split' : 1, # nmin
		'svm' : 0,
		'svm_c' : 1.0,
	}
	# Define command line options
	p = optparse.OptionParser(description = 'Pyxit/Cytomine Classification Model Builder',
							  prog = 'PyXit Classification Model Builder (PYthon piXiT)')

	# Cytomine arguments
	p.add_option("--cytomine_host", type = "string", default = '', dest = "cytomine_host",
				 help = "The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
	p.add_option('--cytomine_public_key', type = "string", default = '', dest = "cytomine_public_key",
				 help = "Cytomine public key")
	p.add_option('--cytomine_private_key', type = "string", default = '', dest = "cytomine_private_key",
				 help = "Cytomine private key")
	p.add_option('--cytomine_base_path', type = "string", default = '/api/', dest = "cytomine_base_path",
				 help = "Cytomine base path")
	p.add_option('--cytomine_id_software', type = "int", dest = "cytomine_id_software",
				 help = "The Cytomine software identifier")
	p.add_option('--cytomine_working_path', default = "/tmp/", type = "string", dest = "cytomine_working_path",
				 help = "The working directory (eg: /tmp)")
	p.add_option('--cytomine_id_project', type = "int", dest = "cytomine_id_project",
				 help = "The Cytomine project identifier")
	p.add_option('-z', '--cytomine_zoom_level', type = 'int', dest = 'cytomine_zoom_level', help = "working zoom level")
	p.add_option('--cytomine_dump_type', type = 'int', dest = 'cytomine_dump_type',
				 help = "annotation type (1=crop, 2=alphamask)")
	p.add_option('--cytomine_annotation_projects', type = "string", dest = "cytomine_annotation_projects",
				 help = "Projects from which annotations are extracted")
	p.add_option('--cytomine_predict_terms', type = 'string', default = '0', dest = 'cytomine_predict_terms',
				 help = "term ids of predicted terms (=positive class in binary mode)")
	p.add_option('--cytomine_excluded_terms', type = 'string', default = '0', dest = 'cytomine_excluded_terms',
				 help = "term ids of excluded terms")
	p.add_option('--cytomine_reviewed', type = 'string', default = "False", dest = "cytomine_reviewed",
				 help = "Get reviewed annotations only")
	p.add_option('--path_to_project_info', type = "string", default = None, dest = "path_to_project_info",
				 help = "Path to file containing the project information")

	p.add_option('--cytomine_test_projects', type = "string", dest = "cytomine_test_projects",
				 help = "Projects used to test the classification models")
	p.add_option('--cytomine_n_test_images_per_project', type = "int", dest = "cytomine_n_test_images_per_project",
				 help = "Number of images to test on per project")
	p.add_option('--build_models', type = "string", default = "0", dest = "build_models",
				 help = "Turn on (1) or off (0) model building")
	p.add_option('--test_models', type = "string", default = "0", dest = "test_models",
				 help = "Turn on (1) or off (0) model testing")
	p.add_option('--cytomine_dump', type = "string", default = "0", dest = "cytomine_dump",
				 help = "Turn on (1) or off (0) model annotation dump")
	p.add_option('--dump_test_annotations', type = "string", default = "0", dest = "dump_test_annotations",
				 help = "Turn on (1) or off (0) test annotation dump")
	p.add_option('--plot_results', type = "string", default = "0", dest = "plot_results",
				 help = "Turn on (1) or off (0) result plot")
	# Pyxit arguments
	p.add_option('--pyxit_target_width', type = 'int', dest = 'pyxit_target_width', help = "pyxit subwindows width")
	p.add_option('--pyxit_target_height', type = 'int', dest = 'pyxit_target_height', help = "pyxit subwindows height")
	p.add_option('--pyxit_save_to', type = 'string', dest = 'pyxit_save_to',
				 help = "pyxit model directory") # future: get it from server db
	p.add_option('--pyxit_colorspace', type = 'int', dest = 'pyxit_colorspace',
				 help = "pyxit colorspace encoding") # future: get it from server db
	p.add_option('--pyxit_n_jobs', type = 'int', dest = 'pyxit_n_jobs',
				 help = "pyxit number of jobs for trees") # future: get it from server db
	p.add_option('--pyxit_n_subwindows', default = 10, type = "int", dest = "pyxit_n_subwindows",
				 help = "number of subwindows")
	p.add_option('--pyxit_min_size', default = 0.5, type = "float", dest = "pyxit_min_size", help = "min size")
	p.add_option('--pyxit_max_size', default = 1.0, type = "float", dest = "pyxit_max_size", help = "max size")
	p.add_option('--pyxit_interpolation', default = 2, type = "int", dest = "pyxit_interpolation",
				 help = "interpolation method 1,2,3,4")
	p.add_option('--pyxit_transpose', type = "string", default = "False", dest = "pyxit_transpose",
				 help = "transpose subwindows")
	p.add_option('--pyxit_fixed_size', type = "string", default = "False", dest = "pyxit_fixed_size",
				 help = "extract fixed size subwindows")
	p.add_option('--cv_k_folds', type = "int", dest = "cv_k_folds",
				 help = "The number of folds")
	p.add_option('--forest_n_estimators', default = 10, type = "int", dest = "forest_n_estimators",
				 help = "number of base estimators (T)")
	p.add_option('--forest_max_features', default = 1, type = "int", dest = "forest_max_features",
				 help = "max features at test node (k)")
	p.add_option('--forest_min_samples_split', default = 1, type = "int", dest = "forest_min_samples_split",
				 help = "minimum node sample size (nmin)")
	p.add_option('--forest_shared_mem', default = False, action = "store_true", dest = "forest_shared_mem",
				 help = "shared mem")
	p.add_option('--svm', default = 0, dest = "svm",
				 help = "final svm classifier: 0=nosvm, 1=libsvm, 2=liblinear, 3=lr-l1, 4=lr-l2", type = "int")
	p.add_option('--svm_c', default = 1.0, type = "float", dest = "svm_c", help = "svm C")
	# p.add_option('--verbose', action="store_true", default=True, dest="verbose", help="Turn on verbose mode")
	p.add_option('--verbose', type = "string", default = "0", dest = "verbose",
				 help = "Turn on (1) or off (0) verbose mode")


	options, arguments = p.parse_args(args = argv)

	parameters['cytomine_host'] = options.cytomine_host
	parameters['cytomine_public_key'] = options.cytomine_public_key
	parameters['cytomine_private_key'] = options.cytomine_private_key
	parameters['cytomine_base_path'] = options.cytomine_base_path
	parameters['cytomine_working_path'] = options.cytomine_working_path
	parameters['cytomine_base_path'] = options.cytomine_base_path
	parameters['cytomine_id_project'] = options.cytomine_id_project
	parameters['cytomine_id_software'] = options.cytomine_id_software
	parameters['cytomine_annotation_projects'] = map(int, options.cytomine_annotation_projects.split(','))
	parameters['cytomine_test_projects'] = map(int, options.cytomine_test_projects.split(','))
	parameters['cytomine_predict_terms'] = map(int, options.cytomine_predict_terms.split(','))
	parameters['cytomine_excluded_terms'] = map(int, options.cytomine_excluded_terms.split(','))
	parameters['cytomine_zoom_level'] = options.cytomine_zoom_level
	parameters['cytomine_dump_type'] = options.cytomine_dump_type
	parameters['cytomine_dump'] = str2bool(options.cytomine_dump)
	parameters['test_models'] = str2bool(options.test_models)
	parameters['build_models'] = str2bool(options.build_models)
	parameters['dump_test_annotations'] = str2bool(options.dump_test_annotations)
	parameters['plot_results'] = str2bool(options.plot_results)
	parameters['cytomine_reviewed'] = str2bool(options.cytomine_reviewed)
	parameters['path_to_project_info'] = options.path_to_project_info
	parameters['cytomine_n_test_images_per_project'] = options.cytomine_n_test_images_per_project
	pyxit_parameters['pyxit_target_width'] = options.pyxit_target_width
	pyxit_parameters['pyxit_target_height'] = options.pyxit_target_height
	pyxit_parameters['pyxit_n_subwindows'] = options.pyxit_n_subwindows
	pyxit_parameters['pyxit_min_size'] = options.pyxit_min_size
	pyxit_parameters['pyxit_max_size'] = options.pyxit_max_size
	pyxit_parameters['pyxit_colorspace'] = options.pyxit_colorspace
	pyxit_parameters['pyxit_interpolation'] = options.pyxit_interpolation
	pyxit_parameters['pyxit_transpose'] = str2bool(options.pyxit_transpose)
	pyxit_parameters['pyxit_fixed_size'] = str2bool(options.pyxit_fixed_size)
	pyxit_parameters['cv_k_folds'] = options.cv_k_folds
	pyxit_parameters['forest_n_estimators'] = options.forest_n_estimators
	pyxit_parameters['forest_max_features'] = options.forest_max_features
	pyxit_parameters['forest_min_samples_split'] = options.forest_min_samples_split
	pyxit_parameters['forest_shared_mem'] = options.forest_min_samples_split
	pyxit_parameters['svm'] = options.svm
	pyxit_parameters['svm_c'] = options.svm_c
	pyxit_parameters['pyxit_save_to'] = options.pyxit_save_to
	pyxit_parameters['pyxit_n_jobs'] = options.pyxit_n_jobs
	pyxit_parameters['dir_ls'] = os.path.join(parameters["cytomine_working_path"],
											  str(parameters['cytomine_annotation_projects']).replace(',', '-').replace(
												  '[', '').replace(']', '').replace(' ', ''), "zoom_level",
											  str(parameters['cytomine_zoom_level']), "dump_type",
											  str(parameters['cytomine_dump_type']))

	# Check for errors in the options
	if options.verbose :
		print "[pyxit.main] Options = ", options

	# Create JOB/USER/JOB
	conn = cytomine.Cytomine(parameters["cytomine_host"],
							 parameters["cytomine_public_key"],
							 parameters["cytomine_private_key"],
							 base_path = parameters['cytomine_base_path'],
							 working_path = parameters['cytomine_working_path'],
							 verbose = str2bool(options.verbose))

	# Image dump type (for classification use 1)
	if (parameters['cytomine_dump_type'] == 1) :
		annotation_get_func = Annotation.get_annotation_crop_url
	elif (parameters['cytomine_dump_type'] == 2) :
		annotation_get_func = Annotation.get_annotation_alpha_crop_url
	else :
		print "default annotation type crop"
		annotation_get_func = Annotation.get_annotation_crop_url

	if parameters['cytomine_dump']:
		# Get annotation descriptions (JSON) from project(s)
		annotations = None
		for prj in parameters['cytomine_annotation_projects'] :
			if parameters["cytomine_reviewed"] :
				print "Retrieving reviewed annotations..."
				annotations_prj = conn.get_annotations(id_project = prj, reviewed_only = parameters["cytomine_reviewed"])
				print "Reviewed annotations: %d" % len(annotations_prj.data())
			else : # We go here
				print "Retrieving (unreviewed) annotations..."
				annotations_prj = conn.get_annotations(id_project = prj)
				print "(Unreviewed) annotations: %d" % len(annotations_prj.data())
			if not annotations :
				annotations = annotations_prj
			else :
				annotations.data().extend(annotations_prj.data())

			if prj == 21907448 or prj == 155194683 :
				annotations_prj = conn.get_annotations(id_project = prj, id_term = 91376951)
				print "Notadeno annotations: %d" % len(annotations_prj.data())
				annotations.data().extend(annotations_prj.data())
			print "Nb annotations so far... = %d" % len(annotations.data())
		print "Total annotations projects %s = %d" % (parameters['cytomine_annotation_projects'], len(annotations.data()))


		# Set output dir parameters
		if not os.path.exists(pyxit_parameters['dir_ls']) :
			print "Creating annotation directory: %s" % pyxit_parameters['dir_ls']
			os.makedirs(pyxit_parameters['dir_ls'])


		# Dump annotation images locally
		conn.dump_annotations(annotations = annotations, get_image_url_func = annotation_get_func,
		 									dest_path = pyxit_parameters['dir_ls'],
		 									desired_zoom = parameters['cytomine_zoom_level'],
		 									excluded_terms = parameters['cytomine_excluded_terms'])

		quit()
		# Put positive terms under the same term and same for negative terms
		term_directories = os.listdir(pyxit_parameters['dir_ls'])

		pos_path = os.path.join(pyxit_parameters['dir_ls'], "1")
		if not os.path.exists(pos_path) :
			print "Creating positive annotation directory: %s" % pos_path
			os.makedirs(pos_path)

		neg_path = os.path.join(pyxit_parameters['dir_ls'], "0")
		if not os.path.exists(neg_path) :
			print "Creating negative annotation directory: %s" % neg_path
			os.makedirs(neg_path)

		print parameters['cytomine_predict_terms']

		for dir in term_directories:
			dir_abs = os.path.join(pyxit_parameters['dir_ls'], dir)
			# Move files
			if int(dir) in parameters['cytomine_predict_terms'] :
				for image_file in os.listdir(dir_abs) :
					os.rename(os.path.join(dir_abs, image_file), os.path.join(pos_path, image_file))
			else :
				for image_file in os.listdir(dir_abs) :
					os.rename(os.path.join(dir_abs, image_file), os.path.join(neg_path, image_file))
			# Remove empty directory
			os.rmdir(dir_abs)


		print pyxit_parameters['dir_ls']
		os.path.dirname(pyxit_parameters['dir_ls'])

		# Put negative terms undre the same term

		print "END dump annotations"

	# Build models
	if parameters['build_models']:

		# Create directory
		if pyxit_parameters['cv_k_folds'] is None:
			d = os.path.dirname(pyxit_parameters['pyxit_save_to'])
			if not os.path.exists(d) :
				os.makedirs(d)

		# Build models with varying parameters
		min_size = 0.1
		for max_size in np.arange(0.15, 0.8, 0.05) :
			# Model without svm
			build_model(pyxit_parameters, parameters, min_size, max_size, n_trees = 100, svm = 0, min_samples_split = 10)
			quit()
			# Model with svm
			build_model(pyxit_parameters, parameters, min_size, max_size, n_trees = 100, svm = 1, min_samples_split = 100)

	# Dump test annotations
	if parameters['dump_test_annotations'] :
		# Dump test project images
		image_folder_records = open("image_folders_record.csv", 'w')

		annotations = None
		for project_id in parameters['cytomine_test_projects']:
			# Get userjob_id to test for each image
			if parameters['path_to_project_info'] is not None:
				path = os.path.join(parameters["path_to_project_info"], str(project_id))
				df = pd.read_csv(os.path.join(path, "image_info.csv"), sep = ';')

				images= df.loc[df['Term ID'] == 20202][np.isfinite(df['Job user ID'])].as_matrix(['Image ID', 'Job user ID'])
			else:
				images = np.array([parameters["cytomine_id_image"], parameters['cytomine_id_userjob']])

			# Select random images from project
			n_images = len(images)
			n_test_images_per_project = parameters['cytomine_n_test_images_per_project']
			if n_images < parameters['cytomine_n_test_images_per_project'] :
				n_test_images_per_project = n_images
			if n_images == 0 and n_test_images_per_project == 0 :
				continue
			random_idx = np.random.choice(n_images, n_test_images_per_project, replace = False)

			# For each randomly selected image,
			for idx in random_idx:
				i_id = int(images[idx][0])
				u_id = int(images[idx][1])

				candidate_annotations = conn.get_annotations(id_user = u_id,
														 id_image = i_id,
														 id_project = project_id)
				print "Number of annotations to predict: %d" % len(candidate_annotations.data())

				folder_name = os.path.join(parameters["cytomine_working_path"],
										   "annotations",
										   "project-{}".format(project_id),
										   "crops-candidates-{}-{}".format(i_id, u_id),
										   "zoom-{}".format(parameters['cytomine_zoom_level']))

				if not os.path.exists(folder_name):
					os.makedirs(folder_name)

				print "Dumping annotation cropped images to classify to %s" %folder_name
				# Dump annotation images locally
				conn.dump_annotations(annotations = candidate_annotations,
									  get_image_url_func = annotation_get_func,
									  dest_path = folder_name,
									  desired_zoom = parameters['cytomine_zoom_level'])

				# Record image in file
				image_folder_records.write(folder_name + "\n")

		image_folder_records.close()

	# Test each model with each test image
	if parameters['test_models'] :

		# Open folder with where the annotation images are
		with open("image_folders_record.csv", 'r') as f :
			image_folders = f.readlines()



		model_folder = pyxit_parameters['pyxit_save_to']
		print "Open model folder :"
		print model_folder

		for model_name in sorted(os.listdir(model_folder)) :
			print "Open model name :"
			print model_name



			# Create model test result folder
			test_results_folder = os.path.join(parameters['cytomine_working_path'], "test_results_colorspace1", model_name)
			if not os.path.exists(test_results_folder) :
				os.makedirs(test_results_folder)

			for model in sorted(os.listdir(os.path.join(model_folder, model_name))) :
				# Record model test results
				test_result_file_name = model.strip('.pkl') + '.csv'
				test_result_file = open(os.path.join(test_results_folder, test_result_file_name), 'w')
				test_result_file.write("Project ID;"
										"Image ID;"
										"Userjob ID;"
										"Annotation ID;"
										"Reviewed;"
										"Prediction;"
										"Positive prediction probability;"
										"Negative prediction probability\n")

				# Load Classifier model
				print "Open model :"
				print model
				model_path = os.path.join(model_folder, model_name, model)
				classifier = open(model_path, "r")
				classes = pickle.load(classifier)
				pyxit = pickle.load(classifier)
				print "Model: %s" % pyxit

				# Retrieve model information
				annotation_projects, rest = model.split("_svm")
				tmp, rest = rest.split("_minsize")
				svm = int(tmp)
				tmp, rest = rest.split("%_maxsize")
				min_size = int(tmp)/100
				tmp, rest = rest.split("%_ntrees")
				max_size = int(tmp) / 100
				tmp, rest = rest.split(".pkl")
				n_trees = int(tmp)

				for image_folder_name in image_folders:
					image_folder_name = image_folder_name.strip("\n")

					# Get image annotation information
					zoom = os.path.basename(image_folder_name).strip("zoom-")
					rest = os.path.dirname(image_folder_name)
					candidate_name = os.path.basename(rest).strip("crops-candidates-")
					image_id, userjob_id = candidate_name.split("-")
					image_id = int(image_id)
					userjob_id = int(userjob_id)
					rest = os.path.dirname(rest)
					project_id = int(os.path.basename(rest).strip("project-"))

					# Get reviewed annotations from image
					terms = str(parameters['cytomine_predict_terms']).strip('[').strip(']').replace(' ', '')
					annotations = conn.get_annotations(id_project = project_id, id_image = image_id,
													id_term = terms,
													showWKT = True, showMeta = True, showGIS = True,
													reviewed_only = True)

					# Get parent id of reviewed annotations (Annotations whose ID is a key in the table is reviewed)
					review_parent_id_table = {}
					for annot in annotations.data() :
						# Get parent id and put (parent_id, reviewed_id) in a table
						review_parent_id_table[annot.parentIdent] = annot.id

					print "Building subwindows from ", image_folder_name
					# Extract subwindows from all candidates annotations
					X, y = pyxitstandalone.build_from_dir(image_folder_name)

					# Apply pyxit classifier model to X (parameters are already reused from model pkl file)
					y_proba = pyxit.predict_proba(X) * 100

					for j, annotation_file_name in enumerate(X):
						print annotation_file_name
						annotation_id = int(os.path.basename(annotation_file_name).strip('.png').split('_')[1])
						y_predict = classes.take(np.argmax(y_proba[j], 0), axis = 0)
						y_rate = np.max(y_proba[j])

						# Check if annotation has been reviewed
						reviewed = False
						if review_parent_id_table.has_key(annotation_id):
							reviewed = True

						test_result_file.write("{};{};{};{};{};{};{};{}\n".format(project_id,
												image_id,
												userjob_id,
												annotation_id,
												reviewed,
												y_predict,
												y_proba[j, 1],
												y_proba[j, 0]))


				test_result_file.close()
				

	# Plot results
	if parameters['plot_results'] :

		test_results_folder = os.path.join(parameters['cytomine_working_path'], "test_results_colorspace0")
		stats_file = open(os.path.join(parameters['cytomine_working_path'], 'stats.txt'), 'w') # file containing test stats


		for sub_folder in sorted(os.listdir(test_results_folder)) :
			for results_file in sorted(os.listdir(os.path.join(test_results_folder, sub_folder))):
				# Read results file
				print os.path.join(test_results_folder, sub_folder, results_file)

				# Read csv
				df = pd.read_csv(os.path.join(test_results_folder, sub_folder, results_file), sep = ';')
				df.insert(0, "Index", df.index)
				n_data = df.shape[0]
				model = os.path.join(sub_folder, results_file.strip('.csv') + '.plk')
				stats_file.write('\n\nModel name : {}\n'.format(model))

				# Find unique project ids
				plt.figure(figsize = [14, 7])
				unique_project_ids = df['Project ID'].unique()
				n_projects = len(unique_project_ids)
				# Get annotation prediciton probability
				pred_positive = df.loc[df['Reviewed'] == True].as_matrix(["Index", "Positive prediction probability"])
				pred_negative = df.loc[df['Reviewed'] == False].as_matrix(["Index", "Positive prediction probability"])
				precision_array = np.zeros(n_projects)
				recall_array = np.zeros(n_projects)
				tp_array = np.zeros(n_projects)
				fn_array = np.zeros(n_projects)
				fp_array = np.zeros(n_projects)
				tn_array = np.zeros(n_projects)

				for i, id in enumerate(unique_project_ids):
					project_data = df.loc[df['Project ID'] == id]
					idx_id = int(project_data['Index'].head(1))
					plt.axvline(x = idx_id, color = 'b', linestyle = 'dashed', linewidth = 1, alpha = 0.3)

					# Write stats about model tests
					project_name = conn.get_project(id).name
					print project_name

					tp_array[i] = project_data.loc[project_data['Reviewed'] == True].loc[project_data['Prediction'] == 1].shape[0]
					fn_array[i] = project_data.loc[project_data['Reviewed'] == True].loc[project_data['Prediction'] == 0].shape[0]
					fp_array[i] = project_data.loc[project_data['Reviewed'] == False].loc[project_data['Prediction'] == 1].shape[0]
					tn_array[i] = project_data.loc[project_data['Reviewed'] == False].loc[project_data['Prediction'] == 0].shape[0]
					precision_array[i] = float(tp_array[i])/(float(tp_array[i]) + float(fp_array[i]))
					recall_array[i] = float(tp_array[i])/(float(tp_array[i]) + float(fn_array[i]))

					stats_file.write('\n\t{} ({})\n'.format(project_name, id))
					stats_file.write('\tNumber annotations : {}\n'.format(project_data.shape[0]))
					stats_file.write('\tTP : {} \tFN : {}\t| {}\n'.format(tp_array[i], fn_array[i], (tp_array[i] + fn_array[i])))
					stats_file.write('\tFP : {} \tTN : {}\t| {}\n'.format(fp_array[i], tn_array[i], (fp_array[i] + tn_array[i])))
					stats_file.write('\t     -----------------------\n')
					stats_file.write('\t     {}\t\t{}\n'.format((tp_array[i] + fp_array[i]), (fn_array[i] + tn_array[i])))
					stats_file.write('\tPrecision : \t{}\n'.format(precision_array[i]))
					stats_file.write('\tRecall : \t{}\n'.format(recall_array[i]))
				# Plot : For each different model, scatter plot of all annotations of all projects

				plt.scatter(pred_negative[:,0], pred_negative[:,1], c = 'r', label = "Actual negative", alpha = 0.6)
				plt.scatter(pred_positive[:,0], pred_positive[:,1], c = 'g', label = "Actual positive", alpha = 0.7)
				plt.title('Classification probability of annotations')
				plt.xlabel('Annotations')
				plt.ylabel('Probability (%)')
				plt.ylim([-5, 105])
				plt.xlim([-100, n_data+100])
				legend = plt.legend(loc = 'lower right', framealpha = 0.1)

				# Save plot figure
				fig_name = model.strip('.pkl') + '.png'
				fig_path = os.path.join(parameters['cytomine_working_path'], "plots_colorspace0", fig_name)
				if not os.path.exists(os.path.dirname(fig_path)) :
					os.makedirs(os.path.dirname(fig_path))
				print fig_path
				plt.savefig(fig_path)
				plt.clf()
				plt.close()

				# Plot precision recall map for project : Each point is a project
				plt.figure()
				plt.scatter(recall_array * 100, precision_array * 100, c = 'b')
				plt.title('Precision-Recall map')
				plt.xlabel('Recall')
				plt.ylabel('Precision')
				plt.xlim([-5, 105])
				plt.ylim([-5, 105])
				plt.savefig(os.path.join(parameters['cytomine_working_path'], "plots_colorspace0", os.path.dirname(fig_name), "PR_" + os.path.basename(fig_name)))
				plt.clf()
				plt.close()

		# For each different model, scatter plot of all annotations of all projects
		# For each different model, plot three points for each project :
		#  - best threshold (equal distance),
		#  - lowest probability for positive prediction,
		#  - highest probability for negative prediction

def build_model(pyxit, parameters, min_size, max_size, n_trees, svm, min_samples_split):

	pyxit_parameters = pyxit.copy()

	# Create model name
	min_size_str = str(int(min_size * 100)) + "%"
	max_size_str = str(int(max_size * 100)) + "%"
	model_name = '_'.join(str(e) for e in parameters['cytomine_annotation_projects']) + "_svm" + str(
		svm) + "_minsize" + min_size_str + "_maxsize" + max_size_str + "_ntrees" + str(n_trees) + ".pkl"
	if svm == 0 :
		folder = "nosvm_minss" + str(min_samples_split)
	elif svm == 1 :
		folder = "svm"
	else :
		folder = "else"
	model_path = os.path.join(folder, model_name)
	print "************ Build model : ", model_path, " ************"

	if pyxit_parameters['cv_k_folds'] is None:
		# Create model path to save it
		del pyxit_parameters['cv_k_folds']
		pyxit_parameters['pyxit_save_to'] = os.path.join(pyxit_parameters['pyxit_save_to'], model_path)
		d = os.path.dirname(pyxit_parameters['pyxit_save_to'])
		if not os.path.exists(d) :
			os.makedirs(d)

		print "Creating model : {}".format(pyxit_parameters['pyxit_save_to'])
	else:
		del pyxit_parameters['pyxit_save_to']

	# Adjust parameters
	pyxit_parameters["pyxit_min_size"] = min_size
	pyxit_parameters["pyxit_max_size"] = max_size
	pyxit_parameters["forest_n_estimators"] = n_trees
	pyxit_parameters["svm"] = svm
	pyxit_parameters["forest_min_samples_split"] = min_samples_split

	# Produce pyxit_parameters for training
	argv = []
	for key in pyxit_parameters :
		value = pyxit_parameters[key]
		if type(value) is bool or value == 'True' :
			if bool(value) :
				argv.append("--%s" % key)
		elif not value == 'False' :
			argv.append("--%s" % key)
			argv.append("%s" % value)

	# Build and save model
	pyxitstandalone.main(argv)
	print "End model : {}\n\n".format(model_path)


if __name__ == "__main__" :
	import sys

	main(sys.argv[1 :])
