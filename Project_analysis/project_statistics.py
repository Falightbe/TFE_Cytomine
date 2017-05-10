import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
import os


def basic_statistics(project_name, directory, terms, ratio_comparison = 0, term_roi = None, term_positive = None):
	path = directory
	df = pd.read_csv(os.path.join(path, "image_info.csv"), sep = ';')
	precision = {}
	recall = {}
	# Precision Recall map
	print terms
	print term_positive
	print term_roi

	colour = 'b'
	for term, label in terms.items() :
		precision[term] = df.loc[df['Term ID'] == term].as_matrix(['Precision'])
		recall[term] = df.loc[df['Term ID'] == term].as_matrix(['Recall'])

		precision[term] = precision[term][~np.isnan(precision[term])]
		recall[term] = recall[term][~np.isnan(recall[term])]

		plt.scatter(recall[term], precision[term], c = colour, label = label)
		colour = 'r'

	plt.title('Precision-Recall map (%s)' % project_name)
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.xlim([-5, 105])
	plt.ylim([-5, 105])
	legend = plt.legend(loc = 'lower right', framealpha = 0.1)
	plt.savefig(os.path.join(path, "PR_map.png"))
	if ratio_comparison:
		# Statistics on precision recall
		txt = open(os.path.join(path, "image_info.txt"), 'w')

		if len(precision[term_positive]) != 0 :
			mean_precision_positive = stat.mean(precision[term_positive])
			var_precision_positive = stat.variance(precision[term_positive])
			txt.write("Precision {} : mean = {} ; variance = {}\n".format(terms[term_positive], mean_precision_positive, var_precision_positive))

			mean_recall_positive = stat.mean(recall[term_positive])
			var_recall_positive = stat.variance(recall[term_positive])
			txt.write("Recall {} : mean = {} ; variance = {}\n\n".format(terms[term_positive], mean_recall_positive, var_recall_positive))

		if len(precision[term_roi]) != 0 :
			mean_precision_roi = stat.mean(precision[term_roi])
			var_precision_roi = stat.variance(precision[term_roi])
			txt.write("Precision {} : mean = {} ; variance = {}\n".format(terms[term_roi], mean_precision_roi, var_precision_roi))

			mean_recall_roi = stat.mean(recall[term_roi])
			var_recall_roi = stat.variance(recall[term_roi])
			txt.write("Recall {} : mean = {} ; variance = {}\n\n".format(terms[term_roi], mean_recall_roi, var_recall_roi))

		# Area ratio
		predict_area_positive = df.loc[df['Term ID'] == term_positive].as_matrix(['Predicted Area'])
		review_area_positive = df.loc[df['Term ID'] == term_positive].as_matrix(['Reviewed Area'])
		predict_area_roi = df.loc[df['Term ID'] == term_roi].as_matrix(['Predicted Area'])
		review_area_roi = df.loc[df['Term ID'] == term_roi].as_matrix(['Reviewed Area'])
		image_id = df.loc[df['Term ID'] == 5735].as_matrix(['Image ID'])

		area_ratio_predict = []
		area_ratio_review = []
		area_ratio_diff = []
		idx = []
		for i in range(len(predict_area_positive)):
			if ~np.isnan(predict_area_positive[i]) and ~np.isnan(predict_area_roi[i]) and ~np.isnan(review_area_positive[i]) and ~np.isnan(review_area_roi[i]):
				predict = (predict_area_positive[i]/predict_area_roi[i])[0]
				review = (review_area_positive[i]/review_area_roi[i])[0]
				area_ratio_predict.append(predict)
				area_ratio_review.append(review)
				area_ratio_diff.append(review - predict)
				idx.append(image_id[i])

		for i in range(len(idx)):
			print "{} : {}".format(idx[i], area_ratio_diff[i])
		print


		plt.scatter(area_ratio_predict, area_ratio_review)
		plt.plot([0,1],[0,1], c = 'r')
		plt.title('Area ratio prediction VS review (%s)' %project_name)
		plt.xlabel('Prediction area ratio')
		plt.ylabel('Reviewed area ratio')
		plt.xlim([-0.05, 1.05])
		plt.ylim([-0.05, 1.05])
		plt.savefig(os.path.join(path, "ratio_map.png"))

		if len(area_ratio_diff) != 0:
			mean_area_ratio_diff = stat.mean(area_ratio_diff)
			var_area_ratio_diff = stat.variance(area_ratio_diff)
			txt.write("Area ratio difference : mean = {} ; variance = {}\n".format(mean_area_ratio_diff, var_area_ratio_diff))


def blob_size_statistics(project_name, directory):
	path = directory 
	df = pd.read_csv(os.path.join(path, "area.csv"), sep = ';')
	txt = open(os.path.join(path, "area.txt"), 'w')
	
	# Fetch data
	sizes_ade_rev = df.loc[df['Term ID'] == 20202].loc[df['Reviewed'] == True]
	sizes_ade_job = df.loc[df['Term ID'] == 20202].loc[df['Reviewed'] == False]

	txt.write("Total number of annotations : {}\n\n".format(len(df)))

	# Create bins
	bins = np.array([0, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000])

	# Histogram for Adenocarcinomes reviewed annotations
	plt.figure()
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
	plt.savefig(os.path.join(path, "area_hist_ade_rev.png"))

	# Histogram for Adenocarcinomes job annotations
	df_ade_job = sizes_ade_job['Area']
	txt.write("\n\nStatistics on adenocarcinomes job annotations : \n")
	txt.write(str(df_ade_job.describe()))
	y, x = np.histogram(df_ade_job, bins=bins)
	x = x[1:]
	hist = pd.Series(y, x)
	# Plot

	ax = hist.plot(kind='bar', width=1, alpha=0.5, color = 'b', label = 'Predicted annotations', position = 1)
	ax.set_title('Area histogramme of adenocarcinomes annotations (%s)' %project_name)
	ax.set_xlabel('Annotation area')
	legend = plt.legend(loc='upper right', framealpha = 0.1)
	
	plt.tight_layout()
	plt.savefig(os.path.join(path, "area_hist_ade_predict.png"))


def color_statistics(project_name, directory) :
	path = directory
	df = pd.read_csv(os.path.join(path, "color.csv"), sep = ';')
	txt = open(os.path.join(path, "color.txt"), 'w')

	# Fetch data
	annotation_ids = df.as_matrix(['Annotation ID'])
	annotation_ids = [str(id[0]) for id in annotation_ids]
	n_annotations = len(df)

	x = range(0, n_annotations)
	mean_H = df.as_matrix(['Mean H'])*360/255
	mean_S = df.as_matrix(['Mean S'])*100/255
	mean_V = df.as_matrix(['Mean V'])*100/255
	std_H = df.as_matrix(['Standard deviation H'])*360/255
	std_S = df.as_matrix(['Standard deviation S'])*100/255
	std_V = df.as_matrix(['Standard deviation V'])*100/255

	txt.write("Total number of annotations : {}\n\n".format(n_annotations))

	# Figure
	plt.figure()
	fig, axs = plt.subplots(nrows = 3, ncols = 1, sharex = True)

	# Plot for H component
	ax = axs[0]
	ax.errorbar(x, mean_H, std_H, linestyle = 'None', marker = 's')
	ax.set_title("Hue component")
	ax.set_xlim(left = -1, right = n_annotations + 1)
	ax.set_ylim(0, 360)


	# Plot for S component
	ax = axs[1]
	ax.errorbar(x, mean_S, std_S, linestyle = 'None', marker = 's')
	ax.set_title("Saturation component")
	ax.set_xlim(left = -1, right = n_annotations + 1)
	ax.set_ylim(0, 100)

	# Plot for S component
	ax = axs[2]

	ax.errorbar(x, mean_V, std_V, linestyle = 'None', marker = 's')
	ax.set_title("Value component")
	ax.set_xlabel("Annotations")
	ax.set_xticklabels(annotation_ids, rotation = 'vertical')
	ax.set_xlim(left = -1, right = n_annotations + 1)
	ax.set_ylim(0, 100)

	fig.suptitle("Color in HSV colorspace for project annotations (%s)" % project_name)
	fig.savefig(os.path.join(path, "color_H.png"))
