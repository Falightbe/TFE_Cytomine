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
	p.add_option('--image_file', type = "string", dest = "image_file", default = '', help = 'The file containing image userjob information')
	p.add_option('--roi_term', type='int', dest='roi_term', help="term id of roi term (ROI, lung)")
	p.add_option('--positive_term', type='int', dest='positive_term', help="term id of the term of positive prediction (tumor)")

	# Cytomine options
	p.add_option('--cytomine_host', type="string", default = 'beta.cytomine.be', dest="host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
	p.add_option('--cytomine_public_key', type="string", default = '', dest="public_key", help="Cytomine public key")
	p.add_option('--cytomine_private_key',type="string", default = '', dest="private_key", help="Cytomine private key")
	p.add_option('--cytomine_base_path', type="string", default = '/api/', dest="base_path", help="Cytomine base path")
	p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="working_path", help="The working directory (eg: /tmp)")  
	p.add_option('--project_ids', type="string", dest="project_ids", help="The Cytomine project identifier")
	p.add_option('--userjob_id', default=None, type="int", dest="userjob_id", help="The Cytomine user job id of a specific prediction job to analyse")

	options, arguments = p.parse_args(args = argv)

	projects = map(int,options.project_ids.split(','))
	modes = map(int,options.modes.split(','))
	print "Analyse projects :"
	print projects
	for p_id in projects: 
		# Build data in local directory
		prj = Project_Analyser(host = options.host, public_key = options.public_key, private_key = options.private_key,
							   base_path = options.base_path, working_path = options.working_path,
							   image_file = options.image_file,
							   modes = modes,
							   directory = options.directory, roi_term = options.roi_term,
							   positive_term = options.positive_term, roi_zoom = 5)
		
		prj.launch()
		# Compute statiscal analysis on data
		stat_directory = prj.path
		if 1 in modes:
			basic_statistics(prj.project_name, stat_directory, {options.positive_term : "Adenocarcinome", options.roi_term : "Poumon"}, 1, options.roi_term, options.positive_term)

		if 2 in modes:
			blob_size_statistics(prj.project_name, stat_directory)

		if 3 in modes:
			color_statistics(prj.project_name, stat_directory)


if __name__ == "__main__":
	import sys
	main(sys.argv[1:])
