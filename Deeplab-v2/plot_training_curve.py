import matplotlib.pyplot as plt
import numpy as np

LOG_FILE = './log.txt'

def get_log(log):
	f = open(log, 'r')
	lines = f.readlines()
	f.close()

	loss = []
	for line in lines:
		loss.append(float(line.strip('\n').split(' ')[1]))

	return loss

def plot_iteration(log):
	loss = get_log(log)
	plt.plot(range(len(loss)), loss)
	plt.xlabel('Iteration')
	plt.ylabel('Loss')
	plt.title('Training Curve')
	plt.show()

def plot_epoch(log, num_samples, batch_size):
	"""Avg for each epoch
	num_per_epoch： number of samples in the training dataset
	batch_size: training batch size
	"""
	loss = get_log(log)
	epochs = len(loss) * batch_size // num_samples
	iters_per_epochs = num_samples // batch_size
	x = range(0, epochs+1)
	y = [loss[0]]
	for i in range(epochs):
		y.append(np.mean(np.array(loss[i*iters_per_epochs+1: (i+1)*iters_per_epochs+1])))
	plt.plot(x, y)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Training Curve')
	plt.show()

if __name__ == '__main__':
	plot_epoch(LOG_FILE, 10582, 10)