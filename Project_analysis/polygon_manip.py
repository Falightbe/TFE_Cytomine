import math
from shapely.wkt import loads
from shapely.geometry import box

"""Polygon manipulations"""

def str2polygons (string):
	"""Returns geometric object from string.
	Args:
		string
	Returns:
		shapely geometry
	"""
	return loads(string)
	
def polygone_intersection_area(p1, p2, mode):
	"""....
	Args:
		p1 : geometric object
		p2 : geometric object
		mode : = 0 if intersection between p1 and p2
		       = 1 if p1 \ p2
	Returns:
		...
	"""
	# Find polygone boundaries
	minx =	int(math.floor(min(p1.bounds[0], p2.bounds[0])))
	miny =	int(min(p1.bounds[1], p2.bounds[1]))
	maxx =	int(math.ceil(max(p1.bounds[2], p2.bounds[2])))
	maxy =	int(max(p1.bounds[3], p2.bounds[3]))
	x = range(minx, maxx, 1000)
	x.append(maxx)
	total_area = 0
	
	for i in range(len(x) - 1):
		grid_cell = box(x[i], miny, x[i+1], maxy)
		p1_cell = p1.intersection(grid_cell)
		p2_cell = p2.intersection(grid_cell)
		if mode == 0:
			inter = p1_cell.intersection(p2_cell)
		elif mode == 1:
			inter = p1_cell.difference(p2_cell)
		else:
			inter = p1_cell
			
		total_area = total_area + inter.area
	return total_area

