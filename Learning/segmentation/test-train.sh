# -*- coding: utf-8 -*-

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



#Example to run a segmentation model builder (based on Dumont et al., 2009)

#1. Edit add_software.py and add the software to your Cytomine project if not existing yet


#2. Edit following XXX and 0 values with your cytomine identifiers
cytomine_host="http://beta.cytomine.be"
cytomine_id_software=816476
cytomine_public_key="e56aa80d-9593-4636-acd7-14ad4e1d333b"
cytomine_private_key="3afc3636-cbfc-4381-abc7-9e82526e6240"
cytomine_id_project=150079801
working_path="/home/falight/TFE_Cytomine/Learning/tmp/segmentation" #e.g. /bigdata/tmp/cytomine/
cytomine_annotation_projects=20207620,21903232,669418 #AS6 et AGAR23 et AGAR15-POUMON
cytomine_predict_terms=20202,4746,2171300,2171395 #id of terms to be grouped into the positive class (e.g. tumor) separated by ,
cytomine_excluded_terms=5735 #id of terms that will not be used (neither positive nor negative class) separated by ,

#3. Edit pyxit parameter values to build segmentation model
model_file=20207620_21903232_669418.pkl
zoom=2
windowsize=24
colorspace=2 
njobs=10 
interpolation=1
nbt=10
k=28
nmin=10 
subw=100 
verbose=true
reviewed=false
fixed_size=false # jamais utilisé dans estimator 
transpose=true
tile_size=512

#4. Run
#Note: 
#-Annotations will be dumped into $working_path/$id_project/zoom_level/$zoom/...
#-Model will be created into $working_path/models/$model_file
#-If you want to use only reviewed annotations, uncomment --cytomine_reviewed

python add_and_run_job.py --cytomine_host $cytomine_host --cytomine_public_key $cytomine_public_key --cytomine_private_key $cytomine_private_key --cytomine_base_path /api/ --cytomine_id_software $cytomine_id_software --cytomine_working_path $working_path --cytomine_id_project $cytomine_id_project --cytomine_annotation_projects $cytomine_annotation_projects  -z $zoom --cytomine_predict_terms $cytomine_predict_terms --cytomine_excluded_terms $cytomine_excluded_terms --pyxit_target_width $windowsize --pyxit_target_height $windowsize --pyxit_colorspace $colorspace --pyxit_n_jobs $njobs --pyxit_save_to $working_path/models/$model_file --pyxit_interpolation $interpolation --forest_n_estimators $nbt --forest_max_features $k --forest_min_samples_split $nmin --pyxit_n_subwindows $subw --verbose $verbose --cytomine_reviewed $reviewed --pyxit_fixed_size $fixed_size --pyxit_transpose $transpose



#model_file=SegModel_p150079801_a82731537_z$zoom.pkl #filename of the segmentation model that will be created in $working_path/models/$model_file
#zoom=2 #zoom level to extract annotations (0 = maximum resolution)
#windowsize=24 #size of fixed-size subwindows
#colorspace=2 #colorspace to encode pixel values (see pyxitstandalone.py)
#njobs=10 #number of parallel threads
#interpolation=1 #interpolation (not used)
#nbt=10 #number of trees
#k=28 #number of tests evaluated in each internal tree node
#nmin=10 #minimum node sample size
#subw=100 #number of subwindows extracted by annotation crop
