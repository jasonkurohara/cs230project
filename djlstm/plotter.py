import matplotlib.pyplot as plt

'''
This file is all about the GRAPHS
'''

# Pass in history, returned by call to model.fit()
def display_all_plots(history, deeper, wider, dropout, 
	learning_rate):
	plot_name = 'deeper={}_wider={}_dropout={}_lr={}'.format(
		deeper, wider, dropout, learning_rate)
	plot_loss(history, plot_name)
	plot_accuracy(history, plot_name)

def plot_loss(history, plot_name):
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	title = "LOSS_" + plot_name
	plt.title(title)
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	plt.savefig(title+'.png')


def plot_accuracy(history, plot_name):
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	title = "ACCURACY_" + plot_name
	plt.title('title')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	plt.savefig(title+'.png')