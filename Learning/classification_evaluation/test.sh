# Pyxit Classification Model evaluation using algorithms in Maree et al., TR 2014

#1. Edit following XXX and 0 values with your cytomine identifiers
cytomine_host="http://beta.cytomine.be"
cytomine_id_software=206657173 # Classification_model_builder in AGIC1
cytomine_public_key="e56aa80d-9593-4636-acd7-14ad4e1d333b"
cytomine_private_key="3afc3636-cbfc-4381-abc7-9e82526e6240"
cytomine_id_project=150079801
cytomine_working_path="/home/falight/TFE_Cytomine/Learning/tmp/classification_evaluation" #e.g. /bigdata/tmp/cytomine/enigmas
cytomine_annotation_projects=20207620,21903232,669418,21907448,155194683  #AS6 et AGAR23 et AGAR15-POUMON + AGAR25 + AGIC7
cytomine_test_projects=95267849,132961573,82731537,118492022,181136007,160950554,150079801,151547032,148113329,155192336,151746100,155194951,180770924
cytomine_predict_terms=20202,4746,2171300,2171395 #id of terms to be grouped into the positive class (e.g. tumor) separated by ,
cytomine_excluded_terms=5735 #id of terms that will not be used (neither positive nor negative class) separated by ,
cytomine_dump=false
zoom=2 #zoom_level at which annotations are dumped

#2. Edit pyxit parameter values to build models (see Maree et al. Technical Report 2014)
windowsize=16 #resized_size for subwindows
colorspace=2 #colorspace to encode pixel values
njobs=10 #number of parallel threads
interpolation=1 #interpolation method to rescale subwindows to fixed size
nbt=10 #numer of extra-trees
k=28 #tree node filterning parameter (number of tests) in extra-trees
nmin=10 #minimum node sample size in extra-trees
subw=100 #number of extracted subwindows per annotation crop image
min_size=0.1 #minimum size of subwindows (proportionnaly to image size: 0.1 means minimum size is 10% of min(width,height) of original image
max_size=0.9 #maximum size of subwindows (...)
build_models=false
dump_test_annotations=false # Jamais
test_models=true
plot_results=false
path_to_project_info="../../Project_analysis/tmp/"
cytomine_n_test_images_per_project=5
cv_k_folds=10

#3. Run
#Note:
#-Annotations will be dumped into $working_path/$id_project/zoom_level/$zoom/...
#-Model will be created into $working_path/models/$model_file
#-If you want to use only reviewed annotations, uncomment --cytomine_reviewed
#-If you want to build a model better robust to orientation, uncomment pyxit_transpose (which applies random right-angle rotation to subwindows)
#But it the orientation is related to object classes (e.g. 6 and 9 in digit classification) do not use transpose
#-Code could be modified to specify other filters (e.g. user ids, image ids, increasedarea)

python classification_model_evaluation.py --cytomine_host $cytomine_host --cytomine_public_key $cytomine_public_key --cytomine_private_key $cytomine_private_key --cytomine_base_path /api/ --cytomine_id_software $cytomine_id_software --cytomine_working_path $cytomine_working_path --cytomine_id_project $cytomine_id_project --cytomine_annotation_projects $cytomine_annotation_projects --cytomine_test_projects $cytomine_test_projects -z $zoom --cytomine_excluded_terms $cytomine_excluded_terms --cytomine_predict_terms $cytomine_predict_terms --pyxit_target_width $windowsize --pyxit_target_height $windowsize --pyxit_colorspace $colorspace --pyxit_n_jobs $njobs --pyxit_interpolation $interpolation --forest_n_estimators $nbt --forest_max_features $k --pyxit_n_subwindows $subw --svm 1 --cytomine_dump_type 1 --pyxit_transpose true --pyxit_fixed_size false --cytomine_reviewed false --verbose false --cytomine_dump $cytomine_dump --build_models $build_models --test_models $test_models --dump_test_annotations $dump_test_annotations --plot_results $plot_results --path_to_project_info $path_to_project_info --cytomine_n_test_images_per_project $cytomine_n_test_images_per_project --pyxit_save_to $cytomine_working_path/models_colorspace$colorspace --cv_k_folds $cv_k_folds
