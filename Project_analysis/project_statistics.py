import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import statistics as stat

def basic_statistics(project_name, directory):
	path = directory
	df = pd.read_csv("{}/jr.csv".format(path), sep = ';')

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
 	plt.title('Precision-Recall map (%s)' %project_name)
 	plt.xlabel('Recall')
 	plt.ylabel('Precision')
 	plt.xlim([-5, 105])
 	plt.ylim([-5, 105])
 	legend = plt.legend(loc='lower right', framealpha = 0.1)
 	plt.savefig("{}/PR_map.png".format(path))
	
	
	# Statistics on precision recall
	txt = open("{}/jr.txt".format(path), 'w')
	
	mean_precision_ade = stat.mean(p_ade)
	var_precision_ade = stat.variance(p_ade)
	txt.write("Precision adenocarcinomes : mean = {} ; variance = {}\n".format(mean_precision_ade, var_precision_ade))
	
	mean_recall_ade = stat.mean(r_ade)
	var_recall_ade = stat.variance(r_ade)
	txt.write("Recall adenocarcinomes : mean = {} ; variance = {}\n\n".format(mean_recall_ade, var_recall_ade))
	
	mean_precision_pou = stat.mean(p_pou)
	var_precision_pou = stat.variance(p_pou)
	txt.write("Precision poumons : mean = {} ; variance = {}\n".format(mean_precision_pou, var_precision_pou))
	
	mean_recall_pou = stat.mean(r_pou)
	var_recall_pou = stat.variance(r_pou)
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
 	plt.title('Area ratio job VS review (%s)' %project_name)
 	plt.xlabel('Job area ratio')
 	plt.ylabel('Reviewed area ratio')
 	plt.xlim([-0.05, 1.05])
 	plt.ylim([-0.05, 1.05])
 	plt.savefig("{}/ratio_map.png".format(path))
	
	
	mean_area_ratio_diff = stat.mean(area_ratio_diff)
	var_area_ratio_diff = stat.variance(area_ratio_diff)
	txt.write("Area ratio difference : mean = {} ; variance = {}\n".format(mean_area_ratio_diff, var_area_ratio_diff))
	
	
def blob_size_statistics(project_name, directory):
	path = directory 
	df = pd.read_csv("{}/area.csv".format(path), sep = ';')
	txt = open("{}/area.txt".format(path), 'w')
	
	# Fetch data
	print df
	sizes_ade_rev = df.loc[df['Term ID'] == 20202].loc[df['Reviewed'] == True]
	sizes_ade_job = df.loc[df['Term ID'] == 20202].loc[df['Reviewed'] == False]
#	sizes_pou_rev = df.loc[df['Term ID'] == 5735].loc[df['Reviewed'] == True]
#	sizes_pou_job = df.loc[df['Term ID'] == 5735].loc[df['Reviewed'] == False]

	txt.write("Total number of annotations : {}\n\n".format(len(df)))

	# Create bins
	bins = np.array([0, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000])

	# Histogramme for Adenocarcinomes reviewed annotations
	df_ade_rev = sizes_ade_rev['Area']
	txt.write("Statistics on adenocarcinomes reviewed annotations : \n")
	txt.write(str(df_ade_rev.describe()))
	y, x = np.histogram(df_ade_rev, bins=bins)
	x = x[1:]
	hist = pd.Series(y, x)
	plt.axvline(x = 1, color = 'r', linestyle = 'dashed', linewidth=1)
	plt.axvline(x = 10, color = 'r', linestyle = 'dashed', linewidth=1)
	plt.axvline(x = 19, color = 'r', linestyle = 'dashed', linewidth=1)
	plt.axvline(x = 28, color = 'r', linestyle = 'dashed', linewidth=1)
	# Plot
	ax = hist.plot(kind='bar', width=1, alpha=0.5, color = 'r', label = 'Reviewed annotations', position = 1)
	ax.set_title('Area histogramme of adenocarcinomes reviewed annotations (%s)' %project_name)
	ax.set_xlabel('Annotation area')

	plt.tight_layout()
	plt.savefig("{}/Area_hist_ade_rev.png".format(path))
	
	
	# Histogramme for Adenocarcinomes job annotations
	df_ade_job = sizes_ade_job['Area']
	txt.write("\n\nStatistics on adenocarcinomes job annotations : \n")
	txt.write(str(df_ade_job.describe()))
	y, x = np.histogram(df_ade_job, bins=bins)
	x = x[1:]
	print bins
	print x
	hist = pd.Series(y, x)
	# Plot

	ax = hist.plot(kind='bar', width=1, alpha=0.5, color = 'b', label = 'Job annotations', position = 1)
	ax.set_title('Area histogramme of adenocarcinomes annotations (%s)' %project_name)
	ax.set_xlabel('Annotation area')
	legend = plt.legend(loc='upper right', framealpha = 0.1)
	
	plt.tight_layout()
	plt.savefig("{}/Area_hist_ade_job.png".format(path))
	
#	# Histogramme for Poumon reviewed annotations
#	df_pou_rev = sizes_pou_rev['Area']
#	print df_pou_rev.describe()
#	print
#	y, x = np.histogram(df_pou_rev, bins=bins)
#	x = x[1:]
#	hist = pd.Series(y, x)
#	# Plot
#	ax = hist.plot(kind='bar', width=1, alpha=0.5)
#	ax.set_title('Area histogramme of lung reviewed annotations')
#	ax.set_xlabel('Annotation area')
#	plt.show()
#	
#	# Histogramme for Poumon job annotations
#	df_pou_job = sizes_pou_job['Area']
#	print df_pou_job.describe()
#	print
#	y, x = np.histogram(df_pou_job, bins=bins)
#	x = x[1:]
#	hist = pd.Series(y, x)
#	# Plot
#	ax = hist.plot(kind='bar', width=1, alpha=0.5)
#	ax.set_title('Area histogramme of lung job annotations')
#	ax.set_xlabel('Annotation area')
#	plt.show()
