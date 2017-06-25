#!/usr/bin/env bash


# Cytomine parameters
cytomine_host="http://beta.cytomine.be"
cytomine_public_key="e56aa80d-9593-4636-acd7-14ad4e1d333b"
cytomine_private_key="3afc3636-cbfc-4381-abc7-9e82526e6240"
#cytomine_working_path="/home/falight/TFE_Cytomine/Learning/tmp/segmentation_deep"
cytomine_working_path="/home/mass/ifilesets/ULG/s121985/nobackup/Learning/segmentation_deep_model_builder/"
cytomine_id_project=150079801
cytomine_predict_terms=20202,4746,2171300,2171395 #id of terms to be grouped into the positive class (e.g. tumor) separated by ,
cytomine_excluded_terms=5735 #id of terms that will not be used (neither positive nor negative class) separated by ,


#################################################### Model building ####################################################
# Pyxit parameters for subwindow extraction for training
cytomine_id_software_model_building=207421104 # Software Segmentation_Model_Predict can now be used in project  ULG-LBTD-E2B-NOV2013
window_size=32 #
pyxit_colorspace=2
pyxit_nb_jobs=10
verbose=true
cytomine_reviewed=false
cytomine_zoom_level=2
pyxit_transpose=true
pyxit_n_subwindows=10
pyxit_fixed_size=true
pyxit_interpolation=1
pyxit_min_size=0.5
pyxit_max_size=0.9
nb_jobs=10

# Keras model parameters
keras_save_to="/home/falight/TFE_Cytomine/Learning/segmentation_deep_model_builder/models"
#keras_save_to="/home/mass/ifilesets/ULG/s121985/nobackup/Learning/segmentation_deep_model_builder/models"
keras_batch_size=128
keras_n_epochs=30
keras_shuffle=true
keras_validation_split=0.2

# Run parameters
cytomine_dump_annotations=false
cytomine_dump_annotation_stats=false
build_model=true
cytomine_annotation_projects=20207620,21903232,669418,21907448,155194683  #AS6 et AGAR23 et AGAR15-POUMON + AGAR25 + AGIC7


###################################################### Prediction ######################################################
cytomine_id_software_prediction=207434390 # Software Segmentation_Model_Predict can now be used in project  ULG-LBTD-E2B-NOV2013
cytomine_roi_term=5735 #id of terms that will not be used (neither positive nor negative class) separated by ,
cytomine_reviewed_roi=true

# Tile and union parameters
cytomine_min_size=4000
cytomine_max_size=100000000
cytomine_zoom_level=2
cytomine_tile_size=512
cytomine_tile_min_stddev=5
cytomine_tile_max_mean=250
cytomine_tile_overlap=32 #
cytomine_union_min_length=10
cytomine_union_bufferoverlap=5
cytomine_union_area=5000
cytomine_union_min_point_for_simplify=1000
cytomine_union_min_point=500
cytomine_union_max_point=1000
cytomine_union_nb_zones_width=5
cytomine_union_nb_zones_height=5
nb_jobs=10
startx=12500
starty=200

# Pyxit parameters
cytomine_predict_step=8
pyxit_post_classification=false
pyxit_post_classification_save_to=""


# Run parameters
cytomine_union=true
cytomine_postproc=false
cytomine_count=false
cytomine_mask_internal_holes=true
cytomine_predict_projects=95267849,132961573,82731537,118492022,181136007,160950554,150079801,151547032,148113329,155192336,151746100,155194951,180770924


#################################################### Analysis test #####################################################
modes=1,2		#separated by ,
roi_term=5735
analyse_roi=0 # whether we analyse predictions for ROI or not
positive_term=20202
userjob_id=206831012
project_ids=155194683,155194951,180770924,20207620,7873585


#################################################### Launch python #####################################################
#python add_and_run.py \
python /home/mass/ifilesets/ULG/s121985/TFE_Cytomine/Learning/segmentation_deep_model_builder/add_and_run.py \
--cytomine_host $cytomine_host --cytomine_public_key $cytomine_public_key \
--cytomine_private_key $cytomine_private_key \
--cytomine_base_path /api/ \
--cytomine_id_software $cytomine_id_software_model_building \
--cytomine_working_path $cytomine_working_path \
--cytomine_id_project $cytomine_id_project \
--cytomine_annotation_projects $cytomine_annotation_projects \
--verbose $verbose -z $cytomine_zoom_level \
--cytomine_excluded_terms $cytomine_excluded_terms \
--cytomine_predict_terms $cytomine_predict_terms \
--pyxit_target_width $window_size \
--pyxit_target_height $window_size \
--pyxit_colorspace $pyxit_colorspace \
--pyxit_nb_jobs $pyxit_nb_jobs \
--pyxit_n_subwindows $pyxit_n_subwindows \
--pyxit_transpose $pyxit_transpose \
--pyxit_fixed_size $pyxit_fixed_size \
--pyxit_interpolation $pyxit_interpolation \
--cytomine_reviewed $cytomine_reviewed \
--cytomine_dump_annotations $cytomine_dump_annotations \
--cytomine_dump_annotation_stats $cytomine_dump_annotation_stats \
--build_model $build_model \
--keras_batch_size $keras_batch_size \
--keras_n_epochs $keras_n_epochs \
--keras_shuffle $keras_shuffle \
--keras_validation_split $keras_validation_split \
--keras_save_to $keras_save_to \
--nb_jobs $nb_jobs

#python add_and_run.py  \
python /home/mass/ifilesets/ULG/s121985/TFE_Cytomine/Learning/segmentation_deep_prediction/add_and_run.py \
--cytomine_host $cytomine_host \
--cytomine_public_key $cytomine_public_key \
--cytomine_private_key $cytomine_private_key \
--cytomine_base_path /api/  \
--cytomine_id_software $cytomine_id_software_prediction  \
--cytomine_working_path $cytomine_working_path  \
--cytomine_id_project $cytomine_id_project  \
--cytomine_tile_size $cytomine_tile_size  \
--cytomine_predict_step $cytomine_predict_step  \
--cytomine_tile_overlap $cytomine_tile_overlap  \
--cytomine_tile_min_stddev $cytomine_tile_min_stddev  \
--cytomine_tile_max_mean $cytomine_tile_max_mean  \
--cytomine_union $cytomine_union  \
--cytomine_postproc $cytomine_postproc \
--cytomine_min_size $cytomine_min_size  \
--cytomine_union_min_length $cytomine_union_min_length  \
--cytomine_union_bufferoverlap $cytomine_union_bufferoverlap  \
--cytomine_union_area $cytomine_union_area  \
--cytomine_union_min_point_for_simplify $cytomine_union_min_point_for_simplify   \
--cytomine_union_min_point $cytomine_union_min_point  \
--cytomine_union_max_point $cytomine_union_max_point  \
--cytomine_union_nb_zones_width $cytomine_union_nb_zones_width  \
--cytomine_union_nb_zones_height $cytomine_union_nb_zones_height  \
--cytomine_mask_internal_holes $cytomine_mask_internal_holes  \
--cytomine_count $cytomine_count  \
--pyxit_post_classification $pyxit_post_classification  \
--verbose $verbose  \
--cytomine_predict_projects $cytomine_predict_projects  \
-z $cytomine_zoom_level  \
--cytomine_roi_term $cytomine_roi_term  \
--cytomine_predict_terms $cytomine_predict_terms  \
--pyxit_target_width $window_size  \
--pyxit_target_height $window_size  \
--pyxit_colorspace $pyxit_colorspace  \
--pyxit_nb_jobs $pyxit_nb_jobs  \
--pyxit_n_subwindows $pyxit_n_subwindows  \
--pyxit_transpose $pyxit_transpose  \
--pyxit_fixed_size $pyxit_fixed_size  \
--pyxit_interpolation $pyxit_interpolation  \
--cytomine_reviewed $cytomine_reviewed \
--cytomine_reviewed_roi $cytomine_reviewed_roi  \
--keras_batch_size $keras_batch_size  \
--keras_n_epochs $keras_n_epochs  \
--keras_shuffle $keras_shuffle  \
--keras_validation_split $keras_validation_split  \
--keras_save_to $keras_save_to  \
--startx $startx  \
--starty $starty  \
--nb_jobs $nb_jobs

#python prediction_analysis.py  \
#--cytomine_host $cytomine_host  \
#--cytomine_public_key $cytomine_public_key  \
#--cytomine_private_key $cytomine_private_key  \
#--cytomine_base_path /api/  \
#--cytomine_working_path $working_path  \
#--project_ids $project_ids  \
#--roi_term $roi_term  \
#--positive_term $positive_term  \
#--modes $modes  \
#--directory $directory
