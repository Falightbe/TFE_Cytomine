#!/usr/bin/env bash
# Pyxit Classification Model evaluation using algorithms in Maree et al., TR 2014

#1. Edit following XXX and 0 values with your cytomine identifiers
cytomine_host="http://beta.cytomine.be"
cytomine_public_key="e56aa80d-9593-4636-acd7-14ad4e1d333b"
cytomine_private_key="3afc3636-cbfc-4381-abc7-9e82526e6240"
cytomine_id_software=207421104 # Software Segmentation_Model_Predict can now be used in project  ULG-LBTD-E2B-NOV2013
cytomine_working_path="/home/mass/ifilesets/ULG/s121985/nobackup/Learning/segmentation_deep_model_builder/"
cytomine_id_project=150079801
cytomine_predict_terms=20202,4746,2171300,2171395 #id of terms to be grouped into the positive class (e.g. tumor) separated by ,
cytomine_excluded_terms=5735 #id of terms that will not be used (neither positive nor negative class) separated by ,

# Pyxit parameters for subwindow extraction for training
window_size=32
pyxit_colorspace=2
pyxit_nb_jobs=10
verbose=true
cytomine_reviewed=false
cytomine_zoom_level=2
pyxit_transpose=true
pyxit_n_subwindows=10
pyxit_fixed_size=true
pyxit_interpolation=1
pyxit_min_size=0.1
pyxit_max_size=0.3
nb_jobs=10

# Keras model parameters
keras_save_to="/home/mass/ifilesets/ULG/s121985/nobackup/Learning/segmentation_deep_model_builder/models"
keras_batch_size=128
keras_n_epochs=30
keras_shuffle=true
keras_validation_split=0.2

# Run parameters
cytomine_dump_annotations=false
cytomine_dump_annotation_stats=true
build_model=false
cytomine_annotation_projects=20207620,21903232,669418,21907448,155194683  #AS6 et AGAR23 et AGAR15-POUMON + AGAR25 + AGIC7



python /home/mass/ifilesets/ULG/s121985/TFE_Cytomine/Learning/segmentation_deep_model_builder/add_and_run.py \
--cytomine_host $cytomine_host --cytomine_public_key $cytomine_public_key \
--cytomine_private_key $cytomine_private_key \
--cytomine_base_path /api/ \
--cytomine_id_software $cytomine_id_software \
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