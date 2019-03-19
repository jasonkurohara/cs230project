import matplotlib.pyplot as plt
import os

'''
This file is all about the GRAPHS
'''

# Creates a plot of the training history of each model
# with respect to metric
# @param metric: string metric id (e.g. 'acc', 'loss', etc)
# @return none
# Called by display_all_plots
def plot_metric(metric, histories, legend,
	destination_folder = ""):
	# summarize history for metric
	print('called plot_metric with metric' + metric)
	title = metric.upper()
	print(title)
	for hist_tuple in histories:
		num_epochs = len(hist_tuple[0].history[metric])
		print('len of epoch_count: ' + str(len(hist_tuple[0].history[metric])))
		print('len of '+metric+': ' + str(len(hist_tuple[0].history[metric])))
		plt.plot(range(1,num_epochs+1), hist_tuple[0].history[metric])
	plt.title(title)
	plt.ylabel(metric)
	plt.xlabel('Epoch')
	plt.legend(legend, loc='upper right')
	plt.show()
	plt.savefig(destination_folder + title+'.png')
	plt.clf()

def plot_val_loss(histories, legend,
	destination_folder = ""):
	# summarize history for metric
	metric = "val_loss"
	title = "VALIDATION LOSS DURING TRAINING"
	for hist_tuple in histories:
		num_epochs = len(hist_tuple[0].history[metric])
		print('len of epoch_count: ' + str(len(hist_tuple[0].history[metric])))
		print('len of '+metric+': ' + str(len(hist_tuple[0].history[metric])))
		plt.plot(range(1,num_epochs+1), hist_tuple[0].history[metric])
	plt.title(title)
	plt.ylabel("Validation Loss")
	plt.xlabel('Epoch')
	plt.legend(legend, loc='upper right')
	plt.show()
	plt.savefig(destination_folder + title+'.png')
	plt.clf()


# Plots models' performances based on given metrics
# @param histories: list of tuples of the form 
#	(history, title)
# where history is a History object returned by model.fit()
# and title is a string identifier of that model
# @param metrics: list of string metrics to be shown
# @param name of the folder where 
def display_all_model_plots(histories, folder_name = ""):
	if not folder_name == "" and not os.path.isdir(folder_name):
		os.mkdir(folder_name)
	if not folder_name == "":
		folder_name += "/"
	legend = []
	metrics = list(histories[0][0].history.keys())
	for hist_tuple in histories:
		legend.append(hist_tuple[1])
	for metric in metrics: 
		plot_metric(metric, histories, legend,
			destination_folder = folder_name)
	plot_val_loss(histories, legend, destination_folder=folder_name)

'''
TO INCORPORATE PLOTTER (display_all_model_plots) INTO OUR ML MODEL:
- create an empty list before parameter search
- every time parameter search fits a new model, create a title
	specifying the model created, then append that as well as the 
	History object returned by model.fit as a tuple to the end 
	of your list
	(e.g. title = "Deeper={}, Wider={}, LR={}, Dropout={}".format(
                    deeper,wider,learning_rate,dropout))
    list.append((model.fit(....), title))
- after parameter search is done, pass the list of tuples to display_all_model_plots
'''