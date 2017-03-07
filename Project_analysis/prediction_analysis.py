import sys
import time
import math
from cytomine import Cytomine
from cytomine.models import *
from shapely.geometry import Polygon, Point, MultiPolygon, box
from collections import Counter
from shapely.ops import cascaded_union, unary_union
import os, optparse

from project import Project_Analyser
from project_statistics import *


def main(argv):
	#Define command line options
	p = optparse.OptionParser()

	# Statistical analysis options
	p.add_option('--directory', type="string", dest="directory", help='The statistical data local directory')
	p.add_option('--modes', type="string", dest="modes", default = '', help='The mode code')
	p.add_option('--review_term', type='int', dest='review_term', help="term id of review term (ROI, lung)")
	p.add_option('--job_term', type='int', dest='job_term', help="term id of job term (tumor)")

	# Cytomine options
	p.add_option('--cytomine_host', type="string", default = 'beta.cytomine.be', dest="host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
	p.add_option('--cytomine_public_key', type="string", default = '', dest="public_key", help="Cytomine public key")
	p.add_option('--cytomine_private_key',type="string", default = '', dest="private_key", help="Cytomine private key")
	p.add_option('--cytomine_base_path', type="string", default = '/api/', dest="base_path", help="Cytomine base path")
	p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="working_path", help="The working directory (eg: /tmp)")  
	p.add_option('--project_ids', type="string", dest="project_ids", help="The Cytomine project identifier")	

	options, arguments = p.parse_args(args = argv)

	projects = map(int,options.project_ids.split(','))
	modes = map(int,options.modes.split(','))
	print "Analyse projects :"
	print projects
	for p_id in projects: 
		# Build data in local directory
		prj = Project_Analyser(options.host, options.public_key, options.private_key, options.base_path, options.working_path, p_id, modes, options.directory, options.review_term, options.job_term)
		
		prj.launch()
		# Compute statiscal analysis on data
		if 1 in modes:
			basic_statistics(prj.project_name, "{}/{}".format(options.directory, p_id))

		if 2 in modes:
			blob_size_statistics(prj.project_name, "{}/{}".format(options.directory, p_id))
	
    
if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
