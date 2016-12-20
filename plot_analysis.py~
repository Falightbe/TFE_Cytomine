import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import statistics as stat

def main():
	project = sys.argv[1]
	
	df = pd.read_csv('Project_analysis/Final/%s/data.csv' %project, sep = ';')



	# Precision Recall map	
	p_ade = df.loc[df['Term ID'] == 20202].as_matrix(['Precision'])
	r_ade = df.loc[df['Term ID'] == 20202].as_matrix(['Recall'])
	p_pou = df.loc[df['Term ID'] == 5735].as_matrix(['Precision'])
	r_pou = df.loc[df['Term ID'] == 5735].as_matrix(['Recall'])

	p_ade = p_ade[~np.isnan(p_ade)]
	r_ade = r_ade[~np.isnan(r_ade)]
	p_pou = p_pou[~np.isnan(p_pou)]
	r_pou = r_pou[~np.isnan(r_pou)]
	
 	plt.scatter(r_ade, p_ade, c = 'b', label = 'Adenocarcinomes')
 	plt.scatter(r_pou, p_pou, c = 'r', label = 'Poumon')
 	plt.title('Precision-Recall map (%s)' %project)
 	plt.xlabel('Recall')
 	plt.ylabel('Precision')
 	plt.xlim([-5, 105])
 	plt.ylim([-5, 105])
 	legend = plt.legend(loc='lower right', framealpha = 0.1)
 	plt.savefig('Project_analysis/Final/%s/PR_map.png' %project )
	plt.show()
	
	
	# Statistics on precision recall
	txt = open('Project_analysis/Final/%s/statistics.txt' %project, 'w')
	
	mean_precision_ade = stat.mean(p_ade)
	var_precision_ade = stat.variance(p_ade)
	txt.write("Precision adenocarcinomes : mean = {} ; variance = {}\n".format(mean_precision_ade, var_precision_ade))
	
	mean_recall_ade = stat.mean(p_ade)
	var_recall_ade = stat.variance(p_ade)
	txt.write("Recall adenocarcinomes : mean = {} ; variance = {}\n\n".format(mean_recall_ade, var_recall_ade))
	
	mean_precision_pou = stat.mean(p_pou)
	var_precision_pou = stat.variance(p_pou)
	txt.write("Precision poumons : mean = {} ; variance = {}\n".format(mean_precision_pou, var_precision_pou))
	
	mean_recall_pou = stat.mean(p_pou)
	var_recall_pou = stat.variance(p_pou)
	txt.write("Recall poumons : mean = {} ; variance = {}\n\n".format(mean_recall_pou, var_recall_pou))
	
	# Area ratio
	job_area_ade = df.loc[df['Term ID'] == 20202].as_matrix(['Job Area'])
	rev_area_ade = df.loc[df['Term ID'] == 20202].as_matrix(['Review Area'])
	job_area_pou = df.loc[df['Term ID'] == 5735].as_matrix(['Job Area'])
	rev_area_pou = df.loc[df['Term ID'] == 5735].as_matrix(['Review Area'])
	image_id = df.loc[df['Term ID'] == 5735].as_matrix(['Image ID'])
	
#	job_area_ade = job_area_ade[~np.isnan(job_area_ade)]
#	rev_area_ade = rev_area_ade[~np.isnan(rev_area_ade)]
#	job_area_pou = job_area_pou[~np.isnan(job_area_pou)]
#	rev_area_pou = rev_area_pou[~np.isnan(rev_area_pou)]
	
	
	area_ratio_job = []
	area_ratio_rev = []
	area_ratio_diff = []
	idx = []
	for i in range(len(job_area_ade)):
		if ~np.isnan(job_area_ade[i]) and ~np.isnan(job_area_pou[i]) and ~np.isnan(rev_area_ade[i]) and ~np.isnan(rev_area_pou[i]):
			job = (job_area_ade[i]/job_area_pou[i])[0]
			rev = (rev_area_ade[i]/rev_area_pou[i])[0]
			area_ratio_job.append(job)
			area_ratio_rev.append(rev)
			area_ratio_diff.append(rev - job)
			idx.append(image_id[i])
			

	print idx	
	print 
	for i in range(len(idx)):
		print "{} : {}".format(idx[i], area_ratio_diff[i])
	print 
	
	plt.scatter(area_ratio_job, area_ratio_rev)
	plt.plot([0,1],[0,1], c = 'r')
 	plt.title('Area ratio job VS review (%s)' %project)
 	plt.xlabel('Job area ratio')
 	plt.ylabel('Reviewed area ratio')
 	plt.xlim([-0.05, 1.05])
 	plt.ylim([-0.05, 1.05])
 	plt.savefig('Project_analysis/Final/%s/ratio_map.png' %project )
	plt.show()
	
	
	mean_area_ratio_diff = stat.mean(area_ratio_diff)
	var_area_ratio_diff = stat.variance(area_ratio_diff)
	txt.write("Area ratio difference : mean = {} ; variance = {}\n".format(mean_area_ratio_diff, var_area_ratio_diff))
	
	
	
if __name__ == "__main__":main() 
