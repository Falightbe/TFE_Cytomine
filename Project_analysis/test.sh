#JR_MODE = 1
#AREA_MODE = 2
#COLOR_MODE = 3 
cytomine_host="http://beta.cytomine.be"
cytomine_public_key="e56aa80d-9593-4636-acd7-14ad4e1d333b"
cytomine_private_key="3afc3636-cbfc-4381-abc7-9e82526e6240"
working_path="/home/falight/TFE_Cytomine/Project_analysis/tmp" 
directory="/home/falight/TFE_Cytomine/Project_analysis/data" 
modes=1,2,3		#separated by ,
roi_term=5735
analyse_roi=0 # whether we analyse predictions for ROI or not
positive_term=20202
userjob_id=206586588
project_ids=155192336
#95267849,126664109,141183500,132961573,669418,21903232,21907448,82731537,118492022,181136007,160950554,160965094,150079801,151547032,148113329,155192336,151746100,155194683,155194951,180770924,20207620,7873585   	#separated by ,


python prediction_analysis.py --cytomine_host $cytomine_host --cytomine_public_key $cytomine_public_key --cytomine_private_key $cytomine_private_key --cytomine_base_path /api/ --cytomine_working_path $working_path --project_ids $project_ids --userjob_id $userjob_id --roi_term $roi_term --positive_term $positive_term --modes $modes --directory $directory
