
import numpy as np
import os
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import os.path
try:
	import Image
except:
	from PIL import Image
directory = "tmp/20207620-21903232-669418/zoom_level/2/"

map_classes = None
X = []
y = []
i = 0
j = 0
for c in os.listdir(directory):
	for _file in os.listdir(os.path.join(directory, c)):
		j = j+1
		print "Open..."
		image = Image.open(open(os.path.join(directory, c, _file), 'rb'))\
		# print image.mode
		print image.info
		image = image.convert('HSV')
		Polygon([(0, 100), (100, 100), (100, 0)])
		print "Image size : {}".format(image.size)
		pixels = np.array(image)
		print pixels[0, 0]
		im_copy = np.array(image)

		region = Polygon([(20, 20), (450, 40), (5, 55)])

		for index, pixel in np.ndenumerate(pixels) :
			# Unpack the index.
			# print "Pixel : {}".format(pixel)
			# print "Index : {}".format(index)

			row, col, channel = index
			# We only need to look at spatial pixel data for one of the four channels.
			# if channel != 0 :
			# 	print "Channel not zero"
			# 	print "Pixel : {}".format(pixel)
			# 	print "Index : {}".format(index)
			# 	print
			# 	continue
			point = Point(row, col)
			if not region.contains(point) :
				im_copy[(row, col, 2)] = 0

		cut_image = Image.fromarray(im_copy, 'HSV')
		cut_image.convert('RGB').save('output.png')
		image.convert('RGB').save('input.png')
		quit()
print "There are %d invalid images among %d" %(i, j)