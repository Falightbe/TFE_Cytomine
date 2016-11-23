import sys
import time
from cytomine import Cytomine
from cytomine.models import *
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.wkt import loads
from collections import Counter
from shapely.ops import cascaded_union, unary_union

def str2polygones (string):

	polygones = loads(string)

	return polygones

class Cytomine_Project(object):


	def __init__(self, host, public_key, private_key, project_id, project_name):
		# Public attributes
		self.start_time = time.time()
		self.project_name = project_name
		self.n_images = None
		
		
		# Private attributes
		self.__column_names = "Project ID;Image ID;Term ID;Job user ID;Job parameters;Number of Review Annotations;Number of Job Annotations;Review Area;Job Area;Excess Area detected by job;TP;FN;FP;Recall;Precision;\n"
		self.__host = host
		self.__public_key = public_key
		self.__private_key = private_key
		self.__project_id = project_id
		date = time.strftime("%Y%m%d_%H%M%S");
		self.__filename_csv = "Project_analysis/{}_{}.csv".format(project_name, date)
		self.__filename_txt = "Project_analysis/{}_{}.txt".format(project_name, date)
		self.__conn = Cytomine(host, public_key, private_key, base_path = '/api/', working_path = '/tmp/', verbose= True);
		self.__csv = open(self.__filename_csv, 'w')
		self.__txt = open(self.__filename_txt, 'w')
		self.__images = None
		
		
		
	
	def get_images(self):
		image_instances = ImageInstanceCollection();
		image_instances.project = self.__project_id;
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
                                   reviewed_only = reviewed);
		list_user_id = [];
		list_polygones = [];
		
		# Build list of polygons corresponding to annotations
		for a in annotations.data():
			p = str2polygones(a.location);
			if isinstance(p, MultiPolygon):
				for i in p:
					list_polygones.append(i);
			else:
				list_polygones.append(p);
			list_user_id.append(a.user);
			
			
		# Union of the polygons
		union_polygones = cascaded_union([u if u.is_valid else u.buffer(0) for u in list_polygones])
		
		n_annotations = len(list_user_id);
		
		return union_polygones, n_annotations, list_user_id	
		
		
	def job_reivew_image_analysis(self, image_id, term_id):
		# Get reviewed annotations
		review, n_annotations_review, list_user_id_review = getAnnotation(image_id, True, term_id);
		
		
		# Find most frequent user id that is a job
		user_id = 0
		count = Counter(list_user_id_review);
		for c in count.most_common():
			user_id = c[0]


	
		# Get job annotations
		job, n_annotations_job, list_user_id_job  = getAnnotation(image_id, False, term_id, user_id = user_id);
		
		# Job and review analysis
		self.__csv.write("%d; " %id_project)
		self.__csv.write("%d; " %id_image)
		self.__csv.write("%d; " %id_term)
		
		# Write user_id
		f_tab.write("%d; " %user_id)
		
		# Get job description
		job_desc = conn.get_job(user_id - 1);
		if job_desc is not None:
			f.write('Job parameters : {}\n'.format(job_desc.jobParameters))
			f_tab.write('{}; '.format(job_desc.jobParameters))
		else:
			f.write("Job parameters : []\n")
			f_tab.write("[]; ")
			
		# Write number of annotations
		f_tab.write("%d; " %n_annotations_review)
		f_tab.write("%d; " %n_annotations_job)
		not_in_job = review.difference(job);
		not_in_review = job.difference(review);
		inter = review.intersection(job);

	
		# Write areas
		f_tab.write("%f; " %(review.area))
		f_tab.write("%f; " %(job.area))
		f_tab.write("%f; " %(job.area - review.area))
	
		# Write confusion matrix
		f_tab.write("%f; " %(inter.area)) # TP
		f_tab.write("%f; " %(not_in_job.area)) # FN
		f_tab.write("%f; " %(not_in_review.area)) # FP

		# Write Recall	
		if review.area != 0:
			f_tab.write("%f; " %(100*inter.area/review.area))
		else :
			f_tab.write("0;")
	
		# Write Precision
		if job.area != 0:
			f_tab.write("%f; " %(100*inter.area/job.area))

		else :
			f_tab.write("0;")
			
		# End of entry
		f_tab.write("\n")
		
		
		
		
	def job_review_analysis(self):
		self.get_images()
		self.__csv.write(self.__column_names)
		self.__txt.write('Project {} : {} images'.format(self.__project_id, self.n_images))

		for i in self.__images:
			
			self.compute_data(i.id, 20202)
			self.compute_data(i.id, 5735)
		
		self.__txt.write("\n\n\nIt took %s" %(time.time() - self.start_time))
		self.__txt.close()
		self.__csv.close()
			
		
def main():
	project_id=150079801 # ULG-LBTD-AGIC5
	project_name = "ULG-LBTD-AGIC5"
	host="http://beta.cytomine.be"
	public_key="e56aa80d-9593-4636-acd7-14ad4e1d333b"
	private_key="3afc3636-cbfc-4381-abc7-9e82526e6240"
	
	
	prj = Cytomine_Project(host, public_key, private_key, project_id, project_name)
	prj.job_review_analysis()




if __name__ == "__main__":main() 
