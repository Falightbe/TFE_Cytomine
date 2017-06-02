#!/usr/bin/env bash
# Pyxit Classification Model evaluation using algorithms in Maree et al., TR 2014

#1. Edit following XXX and 0 values with your cytomine identifiers
cytomine_host="http://beta.cytomine.be"
cytomine_public_key="e56aa80d-9593-4636-acd7-14ad4e1d333b"
cytomine_private_key="3afc3636-cbfc-4381-abc7-9e82526e6240"
cytomine_id_software=207434390 # Software Segmentation_Model_Predict can now be used in project  ULG-LBTD-E2B-NOV2013
cytomine_working_path="/home/falight/TFE_Cytomine/Learning/tmp/segmentation_deep"
cytomine_id_project=150079801
cytomine_predict_terms=20202,4746,2171300,2171395 #id of terms to be grouped into the positive class (e.g. tumor) separated by ,
cytomine_roi_term=5735 #id of terms that will not be used (neither positive nor negative class) separated by ,
cytomine_reviewed_roi=true

# Tile and union parameters
cytomine_min_size=4000
cytomine_max_size=100000000
cytomine_zoom_level=2
cytomine_tile_size=512
cytomine_tile_min_stddev=5
cytomine_tile_max_mean=250
cytomine_tile_overlap=128
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
window_size=32
cytomine_predict_step=16
pyxit_post_classification=false
pyxit_post_classification_save_to=""
pyxit_colorspace=2
pyxit_nb_jobs=10
verbose=true

# Pyxit parameters for subwindow extraction for training
cytomine_reviewed=false
pyxit_transpose=true
pyxit_n_subwindows=10
pyxit_fixed_size=true
pyxit_interpolation=1
pyxit_min_size=0.1
pyxit_max_size=0.3

# Keras model parameters
keras_save_to="/home/falight/TFE_Cytomine/Learning/tmp/segmentation_deep/models"
keras_batch_size=128
keras_n_epochs=30
keras_shuffle=true
keras_validation_split=0.2

# Run parameters
cytomine_union=true
cytomine_postproc=false
cytomine_count=false
cytomine_mask_internal_holes=true
cytomine_predict_projects=95267849 #,132961573,82731537,118492022,181136007,160950554,150079801,151547032,148113329,155192336,151746100,155194951,180770924


python add_and_run.py --cytomine_host $cytomine_host --cytomine_public_key $cytomine_public_key --cytomine_private_key $cytomine_private_key --cytomine_base_path /api/ --cytomine_id_software $cytomine_id_software --cytomine_working_path $cytomine_working_path --cytomine_id_project $cytomine_id_project --cytomine_tile_size $cytomine_tile_size --cytomine_predict_step $cytomine_predict_step --cytomine_tile_overlap $cytomine_tile_overlap --cytomine_tile_min_stddev $cytomine_tile_min_stddev --cytomine_tile_max_mean $cytomine_tile_max_mean --cytomine_union $cytomine_union --cytomine_postproc $cytomine_postproc--cytomine_min_size $cytomine_min_size --cytomine_union_min_length $cytomine_union_min_length --cytomine_union_bufferoverlap $cytomine_union_bufferoverlap --cytomine_union_area $cytomine_union_area --cytomine_union_min_point_for_simplify $cytomine_union_min_point_for_simplify  --cytomine_union_min_point $cytomine_union_min_point --cytomine_union_max_point $cytomine_union_max_point --cytomine_union_nb_zones_width $cytomine_union_nb_zones_width --cytomine_union_nb_zones_height $cytomine_union_nb_zones_height --cytomine_mask_internal_holes $cytomine_mask_internal_holes --cytomine_count $cytomine_count --pyxit_post_classification $pyxit_post_classification --verbose $verbose --cytomine_predict_projects $cytomine_predict_projects -z $cytomine_zoom_level --cytomine_roi_term $cytomine_roi_term --cytomine_predict_terms $cytomine_predict_terms --pyxit_target_width $window_size --pyxit_target_height $window_size --pyxit_colorspace $pyxit_colorspace --pyxit_nb_jobs $pyxit_nb_jobs --pyxit_n_subwindows $pyxit_n_subwindows --pyxit_transpose $pyxit_transpose --pyxit_fixed_size $pyxit_fixed_size --pyxit_interpolation $pyxit_interpolation --cytomine_reviewed $cytomine_reviewed--cytomine_reviewed_roi $cytomine_reviewed_roi --keras_batch_size $keras_batch_size --keras_n_epochs $keras_n_epochs --keras_shuffle $keras_shuffle --keras_validation_split $keras_validation_split --keras_save_to $keras_save_to --startx $startx --starty $starty --nb_jobs $nb_jobs