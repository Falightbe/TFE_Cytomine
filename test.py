from shapely.geometry import Polygon, Point

import sys
import time
from cytomine import Cytomine
from cytomine.models import *
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.wkt import loads
from collections import Counter
from shapely.ops import cascaded_union


job = Polygon([(0, 0), (1, 1), (1, 0)])
review = Polygon([(0, 0), (-1, -2), (-1, -1)])
not_in_job = review.difference(job);
not_in_review = job.difference(review);
union = job.union(review)

print "Area job : %f" %job.area

print "Area not in job : %f" %not_in_job.area

print "Area review : %f" %review.area

print "Area not in review : %f" %not_in_review.area
print "Area union : %f" %union.area


multi = MultiPolygon([job, review])
print multi
new = multi[0]
print new
print multi[1]
print type(multi)
if isinstance(multi, MultiPolygon):
	print "It is"
	
job = Polygon([(0, 0), (1, 1), (1, 0), (-1, 0), (0, -1)])
if job.is_valid:
	print "It is valid"
	
else:
	print "It is not valid"
	

print job.buffer(0.0)

