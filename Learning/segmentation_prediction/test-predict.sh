#!/bin/bash

#
# * Copyright (c) 2009-2015. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */


#__author__          = "Marée Raphael <raphael.maree@ulg.ac.be>"
#__copyright__       = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"



#Example to run a segmentation model predictor based on the algorithm published by Dumont et al., 2009.

#0. Edit the add_software.py file to add the software to Cytomine Core (once) and project (once)

#1. Edit following XXX and 0 values with your cytomine identifiers
host="http://beta.cytomine.be"
software=816476
public_key="e56aa80d-9593-4636-acd7-14ad4e1d333b"
private_key="3afc3636-cbfc-4381-abc7-9e82526e6240"
id_project=155192336
working_path="/home/falight/TFE_Cytomine/Learning/tmp"
predict_terms=20202 #id of term to be added to annotations produced by the segmentation model

model_file=test3$zoom.pkl
zoom=2 
windowsize=24
colorspace=2
njobs=10
tile_size=512 
tile_min_stddev=5
tile_max_mean=250
startx=12500
starty=200
roi_term=5735
predict_step=4
min_size=4000 
max_size=100000000
union_minlength=10 
union_bufferoverlap=5 
union_area=5000
union_min_point_for_simplify=1000  
union_min_point=500 
union_max_point=1000
union_nb_zones_width=5 
union_nb_zones_height=5 
cytomine_internal_holes=1
cytomine_count=0
cytomine_reviewed_roi=1
post_classification=false
verbose=1
for image in 150090531  #cytomine ids of images (within project) to work on
do
    python image_prediction_wholeslide.py --cytomine_host $host --cytomine_public_key $public_key --cytomine_private_key $private_key --cytomine_base_path /api/ --cytomine_id_software $software --cytomine_working_path $working_path --cytomine_id_project $id_project -i $image -z $zoom -t $tile_size --cytomine_tile_min_stddev $tile_min_stddev --cytomine_tile_max_mean $tile_max_mean --startx $startx --starty $starty -j $njobs --cytomine_predict_term $predict_terms --cytomine_roi_term $roi_term --pyxit_target_width $windowsize --pyxit_target_height $windowsize --pyxit_colorspace $colorspace --pyxit_nb_jobs $njobs --pyxit_save_to $working_path/models/$model_file --cytomine_predict_step $predict_step --cytomine_union 1 --cytomine_postproc 0 --cytomine_min_size $min_size --cytomine_union_min_length $union_minlength --cytomine_union_bufferoverlap $union_bufferoverlap --cytomine_union_area $union_area --cytomine_union_min_point_for_simplify $union_min_point_for_simplify  --cytomine_union_min_point $union_min_point --cytomine_union_max_point $union_max_point --cytomine_union_nb_zones_width $union_nb_zones_width --cytomine_union_nb_zones_height $union_nb_zones_height --cytomine_mask_internal_holes $cytomine_internal_holes --cytomine_count $cytomine_count --cytomine_reviewed_roi $cytomine_reviewed_roi --pyxit_post_classification $post_classification --verbose $verbose #--pyxit_post_classification_save_to $working_path/models/post_classification_model.pkl --cytomine_max_size $max_size

done


##2. Edit pyxit parameter values to apply segmentation model (most parameter values are derived from the model file, in theory should keep the same values)
#zoom=2 #zoom level to apply model (0 = maximum resolution)
#windowsize=24 #size of fixed-size subwindows
#colorspace=2 #colorspace to encode pixel values
#njobs=10 #number of parallel threads


##3. Edit additional parameters related to how the model is applied in the whole slide or subregions, and how predictions are merged (union)
#tile_size=512 #size of tiles on which to apply segmentation model (results will then be merged accross whole slide using union procedure)
#tile_min_stddev=5 #we do not apply model on tiles which r and g and b stddev is inferior to tile_min_stddev
#tile_max_mean=250 #we do not apply model on tiles which r and g and b mean is superior to tile_max_mean (e.g. white background tiles)
#startx=12500 #x position where to start in the whole slide
#starty=200 #y position where to start in the whole slide
#roi_term=XXX #id of roi term (to apply predictor only in the roi (e.g. in a pre-segmented tissue section only) instead of in the whole slide)
#predict_step=4 #1=we apply the predictor to every pixel position, n=every n pixel
#min_size=4000 #discard detected regions that are smaller than min_size (to set this value you can draw a manual annotation of objects of interest on the web ui and look their area)
#max_size=100000000 #discard detected regions that are larger than max_size
##union parameters (to merge annotations over the whole slide by merging local annotations in each tile)
#union_minlength=10 # we consider merging polygons that have at least 20 pixels in common
#union_bufferoverlap=5 # for each polygon, we look in a surrounding region of 5 pixels for considering neighboor polygons
#union_area=5000
#union_min_point_for_simplify=1000  #if an annotation has more than x points after union, it will be simplified (default 10000)
#union_min_point=500 #minimum number of points for simplified annotation
#union_max_point=1000 #maximum number of points for simplified annotation
#union_nb_zones_width=5 #an image is divided into this number of horizontal grid cells to perform lookup
#union_nb_zones_height=5 #an image is divided into this number of vertical grid cells to perform lookup

#4. Run
#Notes: 
#- If you want to apply the segmentation model only in a region of interest (e.g. in the tissue section previously segmented) use the --cytomine_roi_term
#(use --cyotmine_reviewed_roi if the roi is part of the reviewed annotations), otherwise (application in the whole image) remove this parameter
#- It uses the segmentation model filename specified by --pyxit_save_to parameter
#- If you want to perform a post classification of segmented objects, uncomment pyxit_post_classification_save_to and specify the location of your classification model
#- If you want to perform some statistics (counts), use --cytomine_count
#- Union uses jts and cytomine groovy-java client in lib/ subdirectory (see code). You should have an updated Cytomine client jar.
#- Segmentation model file is located in $working_path/models/$model_file

