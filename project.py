import sys
import time
from cytomine import Cytomine
from cytomine.models import *
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.wkt import loads
from collections import Counter
from shapely.ops import cascaded_union, unary_union
import os 
def str2polygones (string):

	polygones = loads(string)

	return polygones

class Cytomine_Project(object):


	def __init__(self, host, public_key, private_key, project_id):
		# Public attributes
		self.start_time = time.time()
		self.n_images = None
		self.project_id = project_id
	
		
		
		# Private attributes
		self.__column_names = "Project ID;Image ID;Term ID;Job user ID;Job parameters;Number of Review Annotations;Number of Job Annotations;Number of Review Polygons;Number of Job Polygons;Number of review polygons missed by job;Review Area;Job Area;Area Percentage of review missed by job;TP;FN;FP;Recall;Precision;\n"
		self.__host = host
		self.__public_key = public_key
		self.__private_key = private_key
		date = time.strftime("%Y%m%d_%H%M%S");
		self.__conn = Cytomine(host, public_key, private_key, base_path = '/api/', working_path = '/tmp/', verbose= True);
		self.project_name = self.get_name()
		directory = "Project_analysis/{}_{}".format(self.project_name, date)
		if not os.path.exists(directory):
    			os.makedirs(directory)
		self.__filename_csv = "{}/data.csv".format(directory)
		self.__filename_txt = "{}/log.txt".format(directory)
		self.__csv = open(self.__filename_csv, 'w')
		self.__txt = open(self.__filename_txt, 'w')
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
		
		
	def get_annotations(self, reviewed, image_id, term_id, user_id = None):
		# Fetch annotations
		annotations = self.__conn.get_annotations(
                                   id_project = self.project_id, 
                                   id_user = user_id,
                                   id_image = image_id, 
                                   id_term = term_id, 
                                   showWKT=True, 
                                   showMeta=True,
                                   showGIS=True,
                                   reviewed_only = reviewed)
                
                if len(annotations.data()) == 0 :
                	self.__txt.write("Image {}, Term {}, Reviewed {} : NO ANNOTATIONS\n".format(image_id, term_id, reviewed))
		list_user_id = [];
		list_polygones = [];
		
		# Build list of polygons corresponding to annotations
		print "Build list of polygons corresponding to annotations..."
		n_polygones = 0
		for a in annotations.data():

			p = str2polygones(a.location);
			invalid = False
			if isinstance(p, MultiPolygon):
				for i in p:
					n_polygones = n_polygones + 1
					if i.is_valid:
						list_polygones.append(i);
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
				self.__txt.write("Invalid polygon on image {}, annotation {}\n".format(image_id, a.id))
			
						
			list_user_id.append(a.user);
			

		# Union of the polygons
		union_polygones = cascaded_union(list_polygones)
		
		n_annotations = len(list_user_id);
		
		return union_polygones, n_annotations, list_user_id, n_polygones	
		
		
	def job_reivew_image_analysis(self, image_id, term_id):
		# Get reviewed annotations
		review, n_annotations_review, list_user_id_review, n_polygones_review = self.get_annotations(True, image_id, term_id);
		
		
		# Find most frequent user id that is a job
		count = Counter(list_user_id_review);
		if len(count.most_common()) != 0:
			user_id = count.most_common()[0][0]
		else:
			user_id = 0
			
#		print "\nCount : "
#		print count
#		print "\nCount.most_common() : "
#		print count.most_common()
#		print "\nCount.most_common()[0][0] : "
#		print count.most_common()[0][0]
#		
#		quit()

	
		# Get job annotations
		job, n_annotations_job, list_user_id_job, n_polygones_job  = self.get_annotations(False, image_id, term_id, user_id = user_id);
		
		
		# Job and review analysis
		self.__csv.write("%d; " %self.project_id)
		self.__csv.write("%d; " %image_id)
		self.__csv.write("%d; " %term_id)
		
		# Annotation existence verification
		if job.area == 0 or review.area == 0:
			 self.__csv.write("\n")
			 return
		
		# Write user_id
		self.__csv.write("%d; " %user_id)
		
		# Get job description
		job_desc = self.__conn.get_job(user_id - 1);
		if job_desc is not None:
			self.__csv.write('{}; '.format(job_desc.jobParameters))
		else:
			self.__txt.write('Job not found : {}\n'.format(user_id - 1))
			self.__csv.write("[];")

		# Write number of annotations
		self.__csv.write("%d; " %n_annotations_review)
		self.__csv.write("%d; " %n_annotations_job)
		self.__csv.write("%d; " %n_polygones_review)
		self.__csv.write("%d; " %n_polygones_job)
		self.__csv.write("%d; " %(n_polygones_review - n_polygones_job))
		
		print "Area calculations..."
		not_in_job = review.difference(job);
		not_in_review = job.difference(review);
		inter = review.intersection(job);
#		not_in_job = review;
#		not_in_review = job;
#		inter = review;


	
		# Write areas
		self.__csv.write("%f; " %(review.area))
		self.__csv.write("%f; " %(job.area))
		self.__csv.write("%f; " %(100*(review.area-job.area)/review.area))
	
		# Write confusion matrix
		self.__csv.write("%f; " %(inter.area)) # TP
		self.__csv.write("%f; " %(not_in_job.area)) # FN
		self.__csv.write("%f; " %(not_in_review.area)) # FP
		
		# Write Recall	
		if review.area != 0:
			self.__csv.write("%f; " %(100*inter.area/review.area))
		else :
			self.__csv.write("0;")
	
		# Write Precision
		if job.area != 0:
			self.__csv.write("%f; " %(100*inter.area/job.area))

		else :
			self.__csv.write("0;")
			
		# End of entry
		self.__csv.write("\n")
		
		
		
		
	def job_review_analysis(self):
		self.get_images()
		self.__csv.write(self.__column_names)
		self.__txt.write('Project name : {} \n'.format(self.project_name))
		self.__txt.write('Project ID : {} \n'.format(self.project_id))
		self.__txt.write('Number of images : {}\n\n'.format(self.n_images))

		
		self.__txt.write('Logs : \n')
		for i in self.__images:
			
			self.job_reivew_image_analysis(i.id, 20202)
			self.job_reivew_image_analysis(i.id, 5735)
			
	
		seconds = time.time() - self.start_time
		
		self.__txt.write("\n\nIt took %s" %time.strftime('%H:%M:%S', time.gmtime(seconds)))
		self.__txt.close()
		self.__csv.close()
			
		
def main():

	project_id = int(sys.argv[1])
	host="http://beta.cytomine.be"
	public_key="e56aa80d-9593-4636-acd7-14ad4e1d333b"
	private_key="3afc3636-cbfc-4381-abc7-9e82526e6240"

	
	prj = Cytomine_Project(host, public_key, private_key, project_id)

	prj.job_review_analysis()




if __name__ == "__main__":main() 
