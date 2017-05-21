
try:
	import Image, ImageStat
except:
	from PIL import Image, ImageStat

import shapely.wkt
import numpy as np
from matplotlib.path import Path
import scipy.ndimage
from pyxit.data import build_from_dir
import shapely

#For parallel extraction of subwindows in current tile
def _parallel_crop_boxes (y_roi, x_roi, image_filename, half_width, half_height, pyxit_colorspace):
	try:
		import Image
	except:
		from PIL import Image

	_X = []
	boxes = np.empty((len(x_roi)*len(y_roi), 4),dtype=np.int)
	i = 0
	image = Image.open(image_filename)
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

#For parallel construction of confidence map in current tile
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


#To convert a polyogn into a list of components
def polygon_2_component(polygon):
	exterior = list(polygon.exterior.coords)
	interiors = []
	for interior in polygon.interiors:
		interiors.append(list(interior.coords))
	return (exterior, interiors)


#To convert a union of roi polygons into a rasterized mask
def rasterize_tile_roi_union(nx, ny, points, local_tile_component, roi_annotations_union, whole_slide, reader):
	tile_component = whole_slide.convert_to_real_coordinates(whole_slide, [local_tile_component], reader.window_position, reader.zoom)[0]
	tile_polygon = shapely.geometry.Polygon(tile_component[0], tile_component[1])
	tile_roi_union = tile_polygon.intersection(roi_annotations_union)

	tile_roi_union_components = []
	if (tile_roi_union.geom_type == "Polygon"):
		tile_roi_union_components.append(polygon_2_component(tile_roi_union))
	if (tile_roi_union.geom_type == "MultiPolygon"):
		for geom in tile_roi_union.geoms:
			tile_roi_union_components.append(polygon_2_component(geom))

	local_tile_roi_union_components = whole_slide.convert_to_local_coordinates(whole_slide, tile_roi_union_components, reader.window_position, reader.zoom)
	local_tile_roi_union_polygons = [shapely.geometry.Polygon(component[0], component[1]) for component in local_tile_roi_union_components]

	local_tile_roi_union_raster = np.zeros((ny, nx), dtype=np.bool)
	for polygon in local_tile_roi_union_polygons:
		vertices = np.concatenate([np.asarray(polygon.exterior)] + [np.asarray(r) for r in polygon.interiors])
		#grid = points_inside_poly(points, vertices) #deprecated > matplotlib 1.2
		path = Path(vertices)
		grid = path.contains_points(points)
		grid = grid.reshape((ny,nx))
		local_tile_roi_union_raster |= grid

	return local_tile_roi_union_raster


#To remove unvalid polygon patterns
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
	#TODO the question is :
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