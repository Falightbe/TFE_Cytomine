import os
import time
from collections import Counter

import numpy as np
from cytomine import Cytomine
from cytomine.models import *
from shapely.geometry import MultiPolygon
from shapely.ops import cascaded_union

from polygon_manip import *

try :
	import Image, ImageStat
except :
	from PIL import Image, ImageStat

TERM_ID_ADE = 20202
TERM_ID_POU = 5735


class Project_Analyser(object) :
	"""Build data related to Cytomine project and segmentation in local directory"""

	def __init__(self, host, public_key, private_key, base_path, working_path, project_id, modes, directory, roi_term,
				 positive_term, roi_max_size = None, roi_zoom = None, positive_user_job_id = None):
		# Public attributes
		self.start_time = time.time()
		self.n_images = None
		self.modes = modes
		self.project_id = project_id

		# Private attributes
		self.__term_id_positive = positive_term
		self.__term_id_roi = roi_term
		self.__host = host
		self.__public_key = public_key
		self.__private_key = private_key
		self.__conn = Cytomine(host, public_key, private_key, base_path = base_path, working_path = working_path,
							   verbose = False)
		self.__positive_userjob_id = positive_user_job_id
		self.project_name = self.get_name()

		if positive_user_job_id is None :
			self.path = os.path.join(directory, str(self.project_id))
		else :
			self.path = os.path.join(directory, str(self.project_id), str(self.__positive_userjob_id))
		if not os.path.exists(self.path) :
			os.makedirs(self.path)

		self.__filename_txt = "{}/log.txt".format(self.path, self.project_name)
		self.__txt = open(self.__filename_txt, 'w')

		if 1 in modes:
			self.__image_info_filename_csv = "{}/image_info.csv".format(self.path, self.project_name)
			self.__image_info_csv = open(self.__image_info_filename_csv, 'w')
		if 2 in modes:
			self.__area_filename_csv = "{}/area.csv".format(self.path, self.project_name)
			self.__area_csv = open(self.__area_filename_csv, 'w')
		if 3 in modes:
			self.__color_filename_csv = "{}/color.csv".format(self.path, self.project_name)
			self.__color_csv = open(self.__color_filename_csv, 'w')
			self.__roi_max_size = roi_max_size
			self.__roi_zoom = roi_zoom

		self.__images = None

	def get_name(self):
		project = self.__conn.get_project(self.project_id)
		return project.name

	def get_images(self):
		image_instances = ImageInstanceCollection()
		image_instances.project = self.project_id
		image_instances = self.__conn.fetch(image_instances)
		self.__images = image_instances.data()
		self.n_images = len(self.__images)

	def annotations(self, reviewed, image_id, term_id, user_id = None):
		"""Fetches data about annotations 
		Args:
			reviewed : reviewed only or not
			image_id : image ID
			term_id : term ID
			user_id : user ID
		Returns:
			union_polygones : geometric object corresponding to the union of all annotations
			n_annotations : number of annotations
			list_user_id : list of user IDs related to each annotation
			n_polygones	: number of polygons all annotations included
		"""
		# Fetch annotations
		annotations = self.__conn.get_annotations(id_project = self.project_id, id_user = user_id, id_image = image_id,
												  id_term = term_id, showWKT = True, showMeta = True, showGIS = True,
												  reviewed_only = reviewed)

		if len(annotations.data()) == 0 :
			self.__txt.write(
				"Image {}, Term {}, Reviewed only {} : NO ANNOTATIONS FETCHED\n".format(image_id, term_id, reviewed))
		list_user_id = []

		list_polygones = []
		n_polygones = 0

		if 1 in self.modes :
			print "Build list of polygons corresponding to annotations..."
		if 2 in self.modes :
			print "Writing annotation information..."

		for a in annotations.data() :
			list_user_id.append(a.user)

			# Build list of polygons corresponding to annotations
			if 1 in self.modes :
				p = str2polygons(a.location)
				invalid = False
				if isinstance(p, MultiPolygon) :
					for i in p :
						n_polygones += 1
						if i.is_valid :
							list_polygones.append(i)
						else :
							list_polygones.append(i.buffer(0))
							invalid = True
				else :
					n_polygones += 1
					if p.is_valid :
						list_polygones.append(p)
					else :
						list_polygones.append(p.buffer(0))
						invalid = True
				if invalid :
					self.__txt.write(
						"Image {}, Term {}, Reviewed only {}, Annotation {} : Invalid polygon\n".format(image_id,
																										term_id,
																										reviewed, a.id))

			# Writing annotation information
			if 2 in self.modes :
				self.__area_csv.write(
					"{};{};{};{};{};{}\n".format(self.project_id, image_id, a.id, a.term[0], reviewed, a.area))

		# Get image from URL and retrieve HSV data
		if 3 in self.modes and term_id == self.__term_id_roi and reviewed :
			annotation_ids = [a.id for a in annotations.data()]


			print "Dump ROI annotations : Annotation IDs {} ".format(annotation_ids)
			image_dir = os.path.join(self.path, "roi")
			self.__conn.dump_annotations(annotations = annotations,
										 get_image_url_func = Annotation.get_annotation_crop_url,
										 dest_path = image_dir,
										 desired_zoom = self.__roi_zoom)

			for ann_id in annotation_ids :
				try:
					im = Image.open(os.path.join(image_dir, str(term_id), "{}_{}.png".format(image_id, ann_id)))
				except:
					continue
				im = im.convert('HSV')
				pixels = im.getdata()
				pixels = np.array(pixels)
				pixel_mean = np.mean(pixels, axis = 0)
				pixel_std = np.std(pixels, axis = 0)
				self.__color_csv.write(
					"{};{};{};{};{};{};{};{};{};{};{}\n".format(self.project_id, image_id, ann_id, term_id, reviewed,
																pixel_mean[0], pixel_mean[1], pixel_mean[2],
																pixel_std[0], pixel_std[1], pixel_std[2]))

		# Union of the polygons
		union_polygones = cascaded_union(list_polygones)

		n_annotations = len(annotations.data())

		return union_polygones, n_annotations, list_user_id, n_polygones

	def find_user_job_id(self, list_user_id, image_id) :
		"""Find most probable user in user list
		Args:
			list_user_id : list of users
			image_id : image ID
		Returns:
			user_id : user ID
			job_id : job ID corresponding to user job
		"""

		count = Counter(list_user_id);
		most_common = count.most_common()
		self.__txt.write(
			"Image {} : Most common user ids from review annotations are : {}\n".format(image_id, most_common))

		if len(most_common) != 0 :
			user_id = most_common[0][0]
		else :
			self.__txt.write("Image {} : Review annotations don't share a common user\n".format(image_id))
			return None, None

		# Check the job id

		user = self.__conn.get_user(id_user = user_id)
		if user.algo :
			return user_id, user.job
		else :
			self.__txt.write("Image {}, User {} : User is not an algorithm\n".format(image_id, user_id))
			return user_id, None

	def image_analysis(self, image_id, term_id) :
		"""Fetch data from image annotation with a certain term
		Args:
			image_id : image ID
			term_id : term ID
		Returns:
			_
		"""
		# Get reviewed annotations
		review, n_annotations_review, list_user_id_review, n_polygones_review = self.annotations(True, image_id,
																								 term_id)

		# Write in image_info csv file
		if 1 in self.modes :
			self.__image_info_csv.write("%d;" % self.project_id)
			self.__image_info_csv.write("%d;" % image_id)
			self.__image_info_csv.write("%d;" % term_id)

		# Find most frequent user id that is a job
		if self.__positive_userjob_id is not None and term_id == self.__term_id_positive:
			list_user_id_review = [self.__positive_userjob_id]
		user_id, job_id = self.find_user_job_id(list_user_id_review, image_id)
		if job_id is None : # don't know if job_id is None OR user_id is None
			if 1 in self.modes :
				self.__image_info_csv.write("\n")
			return

		# Get predicted annotations
		predict, n_annotations_predict, list_user_id_predict, n_polygones_predict = self.annotations(False, image_id,
																									 term_id,
																									 user_id = user_id)
		if not (n_annotations_predict) :
			self.__txt.write(
				'Image {}, User {}, Job {} : No predicted annotations found\n'.format(image_id, user_id, job_id))
			self.__image_info_csv.write("\n")
			return
		print "Annotation data :"
		print "Number of reviewed annotations : {}".format(n_annotations_review)
		print "Number of predicted annotations : {}".format(n_annotations_predict)
		print "User ID : {}, Job ID : {}".format(user_id, job_id)

		if 1 in self.modes :
			# Area calculations
			print "Area calculations..."
			not_in_job = polygone_intersection_area(review, predict, 1)
			print "."
			not_in_review = polygone_intersection_area(predict, review, 1)
			print "."
			inter = polygone_intersection_area(review, predict, 0)
			print "."

			# Annotation existence verification
			if predict.area == 0 or review.area == 0 :
				self.__image_info_csv.write("\n")
				return

			# Write user_id
			self.__image_info_csv.write("%d;" % user_id)

			# Get job description
			job_desc = self.__conn.get_job(job_id)
			if job_desc is not None :
				print "JOB PARAMETERS : \n{}".format(job_desc.jobParameters)

				self.__image_info_csv.write('{}; '.format(job_desc.jobParameters))
			else :
				self.__txt.write(
					'Image {}, User {}, Job {} : Job not found, cannot retrieve parameters\n'.format(image_id, user_id,
																									 job_id))
				self.__image_info_csv.write("[];")

			# Write number of annotations
			self.__image_info_csv.write("%d;" % n_annotations_review)
			self.__image_info_csv.write("%d;" % n_annotations_predict)
			self.__image_info_csv.write("%d;" % n_polygones_review)
			self.__image_info_csv.write("%d;" % n_polygones_predict)
			self.__image_info_csv.write("%d;" % (n_polygones_review - n_polygones_predict))
			# Write areas
			self.__image_info_csv.write("%f;" % (review.area))
			self.__image_info_csv.write("%f;" % (predict.area))
			self.__image_info_csv.write("%f;" % (100 * (review.area - predict.area) / review.area))
			# Write confusion matrix
			self.__image_info_csv.write("%f;" % (inter)) # TP
			self.__image_info_csv.write("%f;" % (not_in_job)) # FN
			self.__image_info_csv.write("%f;" % (not_in_review)) # FP
			# Write Recall	
			if review.area != 0 :
				self.__image_info_csv.write("%f;" % (100 * inter / review.area))
			else :
				self.__image_info_csv.write("0;")
			# Write Precision
			if predict.area != 0 :
				self.__image_info_csv.write("%f;" % (100 * inter / predict.area))
			else :
				self.__image_info_csv.write("0;")
			# End of entry
			self.__image_info_csv.write("\n")

	def launch(self) :
		"""Launch the project analysis"""
		self.get_images()
		self.__txt.write('Project name : {} \n'.format(self.project_name))
		self.__txt.write('Project ID : {} \n'.format(self.project_id))
		self.__txt.write('Number of images : {}\n\n'.format(self.n_images))
		self.__txt.write('Logs : \n')

		# Write in data files column names
		if 1 in self.modes :
			self.__image_info_csv.write(
				"Project ID;Image ID;Term ID;Job user ID;Job parameters;Number of Reviewed Annotations;"
				"Number of Predicted Annotations;Number of Reviewed Polygons;Number of Predicted Polygons;"
				"Number of reviewed polygons missed by prediction;Reviewed Area;Predicted Area;Area "
				"Percentage of review missed by prediction;TP;FN;FP;Recall;Precision;\n")
		if 2 in self.modes :
			self.__area_csv.write("Project ID;Image ID;Annotation ID;Term ID;Reviewed;Area\n")

		if 3 in self.modes :
			self.__color_csv.write(
				"Project ID;Image ID;Annotation ID;Term ID;Reviewed;Mean H;Mean S;Mean V;Standard deviation H;Standard deviation S;Standard deviation V\n")

		for i in self.__images :
			image_id = i.id
			print "\nAnalyse image : {}...".format(image_id)
			self.image_analysis(image_id, self.__term_id_positive)
			self.image_analysis(image_id, self.__term_id_roi)

		seconds = time.time() - self.start_time
		self.__txt.write("\n\nIt took %s" % time.strftime('%H:%M:%S', time.gmtime(seconds)))
		self.__txt.close()
		if 1 in self.modes :
			self.__image_info_csv.close()
		if 2 in self.modes :
			self.__area_csv.close()
		if 3 in self.modes :
			self.__color_csv.close()
