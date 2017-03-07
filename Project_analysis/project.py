import sys
import time
import math
from cytomine import Cytomine
from cytomine.models import *
from shapely.geometry import Polygon, Point, MultiPolygon, box
from shapely.wkt import loads
from collections import Counter
from shapely.ops import cascaded_union, unary_union
import os 
from polygon_manip import *

TERM_ID_ADE = 20202
TERM_ID_POU = 5735



class Project_Analyser(object):
	"""Build data related to Cytomine project and segmentation in local directory"""
	def __init__(self, host, public_key, private_key, base_path, working_path, project_id, modes, directory, review_term, job_term):
		# Public attributes
		self.start_time = time.time()
		self.n_images = None
		self.modes = modes
		self.project_id = project_id

		# Private attributes
		self.__term_id_ade = job_term
		self.__term_id_pou = review_term
		self.__host = host
		self.__public_key = public_key
		self.__private_key = private_key
		self.__conn = Cytomine(host, public_key, private_key, base_path = base_path, working_path = working_path, verbose= False);
		self.project_name = self.get_name()
		path = "{}/{}".format(directory, self.project_id)
		if not os.path.exists(path):
			os.makedirs(path)

		self.__filename_txt = "{}/log.txt".format(path, self.project_name)
		self.__txt = open(self.__filename_txt, 'w')

		if 1 in modes:
			self.__jr_filename_csv = "{}/jr.csv".format(path, self.project_name)
			self.__jr_csv = open(self.__jr_filename_csv, 'w')
		if 2 in modes:
			self.__area_filename_csv = "{}/area.csv".format(path, self.project_name)
			self.__area_csv = open(self.__area_filename_csv, 'w')
		if 3 in modes:
			self.__color_filename_csv = "{}/color.csv".format(path, self.project_name)
			self.__color_csv = open(self.__color_filename_csv, 'w')

		self.__images = None
		
		
	def get_name(self):
		project = self.__conn.get_project(self.project_id)
		return project.name

	def get_images(self):
		image_instances = ImageInstanceCollection();
		image_instances.project = self.project_id;
		image_instances = self.__conn.fetch(image_instances);
		self.__images = image_instances.data();
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
		annotations = self.__conn.get_annotations(id_project = self.project_id, id_user = user_id, id_image = image_id, id_term = term_id, showWKT=True, showMeta=True, showGIS=True, reviewed_only = reviewed)
		
		if len(annotations.data()) == 0 :
			self.__txt.write("Image {}, Term {}, Reviewed only {} : NO ANNOTATIONS FETCHED\n".format(image_id, term_id, reviewed))
		list_user_id = [];
	        
		list_polygones = [];
		n_polygones = 0

		if 1 in self.modes:
			print "Build list of polygons corresponding to annotations..."
		if 2 in self.modes:
			print "Writing annotation information..."

		for a in annotations.data():
			list_user_id.append(a.user);

			# Build list of polygons corresponding to annotations
			if 1 in self.modes:
				p = str2polygons(a.location)
				invalid = False
				if isinstance(p, MultiPolygon):
					for i in p:
						n_polygones = n_polygones + 1
						if i.is_valid:
							list_polygones.append(i)
						else:
							list_polygones.append(i.buffer(0))
							invalid = True
				else:
					n_polygones = n_polygones + 1
					if p.is_valid:
						list_polygones.append(p)
					else:
						list_polygones.append(p.buffer(0))
						invalid = True
				if invalid:
					self.__txt.write("Image {}, Term {}, Reviewed only {}, Annotation {} : Invalid polygon\n".format(image_id, term_id, reviewed, a.id))

			# Writing annotation information
			if 2 in self.modes:
				self.__area_csv.write("{};{};{};{};{};{}\n".format(self.project_id, image_id, a.id, a.term[0], reviewed, a.area))
			
			# Get image from URL and retrieve HSV data
			if 3 in self.modes and term_id == 5735:
				print dir(a)
				print a.get_annotation_alpha_crop_url(max_size = 256)
				quit()

		
		# Union of the polygons
		union_polygones = cascaded_union(list_polygones)

		n_annotations = len(annotations.data());

		return union_polygones, n_annotations, list_user_id, n_polygones	

	
	def find_user_job_id(self, list_user_id, image_id):
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
		self.__txt.write("Image {} : Most common user ids from review annotations are : {}\n".format(image_id, most_common))

		if len(most_common) != 0:
			user_id = most_common[0][0]
		else:
			self.__txt.write("Image {} : Review annotations don't share a common user\n".format(image_id))
			return None, None

		# Check the job id
		user = self.__conn.get_user(id_user = user_id)
		if user.algo:
			return user_id, user.job
		else:
			self.__txt.write("Image {}, User {} : User is not an algorithm\n".format(image_id, user_id))
			return user_id, None

	def image_analysis(self, image_id, term_id):
		"""Fetch data from image annotation with a certain term
		Args:
			image_id : image ID
			term_id : term ID
		Returns:
			_
		"""
		# Get reviewed annotations
		review, n_annotations_review, list_user_id_review, n_polygones_review = self.annotations(True, image_id, term_id);

		# Write in JR csv file
		if 1 in self.modes:
			self.__jr_csv.write("%d; " %self.project_id)
			self.__jr_csv.write("%d; " %image_id)
			self.__jr_csv.write("%d; " %term_id)

		# Find most frequent user id that is a job
		user_id, job_id = self.find_user_job_id(list_user_id_review, image_id)
		if job_id is None: # don't know if job_id is None OR user_id is None
			if 1 in self.modes:
				self.__jr_csv.write("\n")
			return

		# Get job annotations
		job, n_annotations_job, list_user_id_job, n_polygones_job  = self.annotations(False, image_id, term_id, user_id = user_id);
	
		print "Annotation data :"
		print "Number of reviewed annotations : {}".format(n_annotations_review)
		print "Number of reviewed annotations : {}".format(n_annotations_job)
		print "User ID : {}, Job ID : {}".format(user_id, job_id)
		quit()
		if 1 in self.modes:
			# Area calculations
			print "Area calculations..."
			not_in_job = polygone_intersection_area(review, job, 1)
			print "."
			not_in_review = polygone_intersection_area(job, review, 1)
			print "."
			inter = polygone_intersection_area(review, job, 0)
			print "."

			# Annotation existence verification
			if job.area == 0 or review.area == 0:
				self.__jr_csv.write("\n")
				return

			# Write user_id
			self.__jr_csv.write("%d; " %user_id)

			# Get job description
			job_desc = self.__conn.get_job(job_id);
			if job_desc is not None:
				self.__jr_csv.write('{}; '.format(job_desc.jobParameters))
			else:
				self.__txt.write('Image {}, User {}, Job {} : Job not found, cannot retrieve parameters\n'.format(image_id, user_id, job_id))
				self.__jr_csv.write("[];")

			# Write number of annotations
			self.__jr_csv.write("%d; " %n_annotations_review)
			self.__jr_csv.write("%d; " %n_annotations_job)
			self.__jr_csv.write("%d; " %n_polygones_review)
			self.__jr_csv.write("%d; " %n_polygones_job)
			self.__jr_csv.write("%d; " %(n_polygones_review - n_polygones_job))
			# Write areas
			self.__jr_csv.write("%f; " %(review.area))
			self.__jr_csv.write("%f; " %(job.area))
			self.__jr_csv.write("%f; " %(100*(review.area-job.area)/review.area))
			# Write confusion matrix
			self.__jr_csv.write("%f; " %(inter)) # TP
			self.__jr_csv.write("%f; " %(not_in_job)) # FN
			self.__jr_csv.write("%f; " %(not_in_review)) # FP
			# Write Recall	
			if review.area != 0:
				self.__jr_csv.write("%f; " %(100*inter/review.area))
			else :
				self.__jr_csv.write("0;")
			# Write Precision
			if job.area != 0:
				self.__jr_csv.write("%f; " %(100*inter/job.area))
			else :
				self.__jr_csv.write("0;")
			# End of entry
			self.__jr_csv.write("\n")
		

	def launch(self):
		"""Launch the project analysis"""
		self.get_images()
		self.__txt.write('Project name : {} \n'.format(self.project_name))
		self.__txt.write('Project ID : {} \n'.format(self.project_id))
		self.__txt.write('Number of images : {}\n\n'.format(self.n_images))
		self.__txt.write('Logs : \n')

		# Write in data files column names
		if 1 in self.modes:
			self.__jr_csv.write("Project ID;Image ID;Term ID;Job user ID;Job parameters;Number of Review Annotations;Number of Job Annotations;Number of Review Polygons;Number of Job Polygons;Number of review polygons missed by job;Review Area;Job Area;Area Percentage of review missed by job;TP;FN;FP;Recall;Precision;\n")
		if 2 in self.modes:
			self.__area_csv.write("Project ID;Image ID;Annotation ID;Term ID;Reviewed;Area\n")

		for i in self.__images:
			image_id = i.id
			print "\nAnalyse image : {}...".format(image_id)
			self.image_analysis(image_id, self.__term_id_ade)
			self.image_analysis(image_id, self.__term_id_pou)

		seconds = time.time() - self.start_time
		self.__txt.write("\n\nIt took %s" %time.strftime('%H:%M:%S', time.gmtime(seconds)))
		self.__txt.close()
		if 1 in self.modes:
			self.__jr_csv.close()
		if 2 in self.modes:
			self.__area_csv.close()
		if 3 in self.modes:
			self.__color_csv.close()
