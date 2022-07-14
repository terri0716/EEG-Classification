import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.model_selection import KFold
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, lfilter

import pywt



### Import Data
def import_data():
	X_test = np.load("X_test.npy")
	y_test = np.load("y_test.npy")
	person_train_valid = np.load("person_train_valid.npy")
	X_train_valid = np.load("X_train_valid.npy")
	y_train_valid = np.load("y_train_valid.npy")
	person_test = np.load("person_test.npy")

	y_train_valid = y_train_valid - 769
	y_test = y_test - 769


	return X_test, y_test, person_train_valid, X_train_valid, y_train_valid, person_test


### Import Data Subject S
def import_data_subject(subject_num):
	# input: subject number
	# output, same 3d shape format as original samples, for only the subject

	subject_num = subject_num-1

	X_test = np.load("X_test.npy")
	y_test = np.load("y_test.npy")
	person_train_valid = np.load("person_train_valid.npy")
	X_train_valid = np.load("X_train_valid.npy")
	y_train_valid = np.load("y_train_valid.npy")
	person_test = np.load("person_test.npy")

	y_train_valid = y_train_valid - 769
	y_test = y_test - 769

	subject_train_val_index, _ = np.where(person_train_valid == subject_num)
	subject_test_index, _ = np.where(person_test == subject_num)

	X_train_valid_s = X_train_valid[subject_train_val_index, :, :]
	y_train_valid_s = y_train_valid[subject_train_val_index]
	X_test_s = X_test[subject_test_index, :, :]
	y_test_s = y_test[subject_test_index]

	return X_test_s, y_test_s, X_train_valid_s, y_train_valid_s



### Dataset Class
class EEGDataset(Dataset):
	def __init__(self, subset, transform=None):
		self.subset = subset
		self.transform = transform

	def __getitem__(self, index):
		x, y = self.subset[index]
		if self.transform:
			x = self.transform(x)
		return x, y

	def __len__ (self):
		return len(self.subset)



### Data Loader Setup
def make_dataloaders(k_train_index, k_val_index, X_train_valid, y_train_valid, X_test, y_test, cut=False, window_width=0, stride=0, num_slices=0):
	
	#if k_train_index == 0:
	#	# train/val split 
	#	train_val_num = X_train_valid.shape[0]
	#	train_num = round(train_val_num * 0.8)
	#	val_num = train_val_num - train_num
	#	k_train_index = np.sort(np.random.choice(train_val_num, train_num, replace=False))
	#	k_val_index = np.delete(np.arange(train_val_num), k_train_index)

	# train/val/test cut 
	if cut:
		X_train_uncut = X_train_valid[k_train_index]
		y_train_uncut = y_train_valid[k_train_index]
		X_val_uncut = X_train_valid[k_val_index]
		y_val_uncut = y_train_valid[k_val_index]

		X_train_f, y_train_f = cut_data(X_train_uncut, y_train_uncut, window_width, stride)
		X_val_f, y_val_f = cut_data(X_val_uncut, y_val_uncut, window_width, stride)
		X_test_f, y_test_f = cut_data(X_test, y_test, window_width, stride)
	else:
		X_train_f = X_train_valid[k_train_index]
		y_train_f = y_train_valid[k_train_index]
		X_val_f = X_train_valid[k_val_index]
		y_val_f = y_train_valid[k_val_index]
		X_test_f = X_test
		y_test_f = y_test


	# reshape X data to be shape (N, 1, 22, trial_length)
	X_train_f = X_train_f[:, np.newaxis, :, :]
	X_val_f = X_val_f[:, np.newaxis, :, :]
	X_test_f = X_test_f[:, np.newaxis, :, :]


	# train/val/test write to tensor & dataset
	X_train_tensor = torch.from_numpy(X_train_f).float()
	y_train_tensor = torch.from_numpy(y_train_f).float().long()
	X_val_tensor = torch.from_numpy(X_val_f).float()
	y_val_tensor = torch.from_numpy(y_val_f).float().long()	
	X_test_tensor = torch.from_numpy(X_test_f).float()
	y_test_tensor = torch.from_numpy(y_test_f).float().long()
	
	train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
	val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
	test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

	train_data = EEGDataset(
	    train_dataset, transform=None)
	val_data = EEGDataset(
	    val_dataset, transform=None)
	test_data = EEGDataset(
	    test_dataset, transform=None)

	#print('X_train_tensor shape: {}'.format(X_train_tensor.shape))
	#print('train_dataset size: {}'.format(len(train_dataset)))
	#print('X_val_tensor shape: {}'.format(X_val_tensor.shape))
	#print('val_dataset size: {}'.format(len(val_dataset)))

	# dataloaders dictionary
	if cut:
		# scale train batchsize accordingly to increase of sample data
		train_batchsize = 32*num_slices
		# restrict validation & test batchsize (and no shuffle), for voting/ensemble accuracy method
		val_batchsize = num_slices
		test_batchsize = num_slices	
	else:
		train_batchsize = 32
		val_batchsize = 32
		test_batchsize = 32

	dataloaders = {
		'train': torch.utils.data.DataLoader(train_data, batch_size = train_batchsize, shuffle=True),
    	'val': torch.utils.data.DataLoader(val_data, batch_size= val_batchsize, shuffle=False),
    	'test': torch.utils.data.DataLoader(test_data, batch_size= test_batchsize, shuffle=False)
	}

	return dataloaders





### Data Splicing Function 
### Used in dataloader function
def cut_data(orig_data, orig_labels, window_width, stride):
	# takes 3d array and 1d array
  	# returns 3d array and 1d array

  	N = int(orig_data.shape[0])
  	feature_size = int(orig_data.shape[1])
  	sample_num = int((orig_data.shape[2]- window_width)/stride + 1)

  	newData = np.zeros([sample_num*N, feature_size, window_width]).astype(orig_data.dtype.name)
  	newlabel = np.zeros([sample_num*N]).astype(orig_labels.dtype.name)

  	for n in range(N):
  		original2d = orig_data[n]
  		newData2d = np.zeros([sample_num, feature_size, window_width]).astype(orig_data.dtype.name)
  		newlabel[n*sample_num : n*sample_num+sample_num] = orig_labels[n]
  	
	  	for i in range(sample_num):
	  		newData2d[i] = original2d[:, i*stride : i*stride+window_width ]
	  		newData[n*sample_num+i] = newData2d[i]

  	return newData, newlabel




### Plot Function
def make_plot(num_epochs, loss_tracker, accuracy_tracker, saveplot):

	figtotal, ax = plt.subplots(1,2, figsize=(12,6))

	# Plot loss
	#fig1 = plt.figure(1)
	ax[0].plot(list(range(1,num_epochs+1)),loss_tracker['train'], label="train")
	ax[0].plot(list(range(1,num_epochs+1)),loss_tracker['val'], label="val")
	ax[0].set_title("Loss")
	ax[0].legend()

	# Plot accuracy
	#fig2 = plt.figure(2)
	ax[1].plot(list(range(1,num_epochs+1)),accuracy_tracker['train'], label="train")
	ax[1].plot(list(range(1,num_epochs+1)),accuracy_tracker['val'], label="val")
	ax[1].set_title("Accuracy")
	ax[1].legend()

	# Max training/validation accuracy
	max_train_acc = np.max(accuracy_tracker['train'])
	max_train_ind = np.argmax(accuracy_tracker['train'])
	max_val_acc = np.max(accuracy_tracker['val'])
	max_val_ind = np.argmax(accuracy_tracker['val'])

	# Current training/validation accuracy
	curr_train_acc = accuracy_tracker['train'][-1]
	curr_val_acc = accuracy_tracker['val'][-1]

	# Test accuracy
	#test_acc_approx = best_results(loss_tracker, accuracy_tracker)

	# Add text to graph
	text1 = 'Max TRAIN accuracy: {:.2f}%, at epoch {}\n'.format(max_train_acc, max_train_ind+1)
	text2 = 'Max VAL accuracy: {:.2f}%, at epoch {}\n'.format(max_val_acc, max_val_ind+1)
	text3 = 'Curr TRAIN accuracy: {:.2f}%, at epoch {}\n'.format(curr_train_acc, num_epochs)
	text4 = 'Curr VAL accuracy: {:.2f}%, at epoch {}\n'.format(curr_val_acc, num_epochs)
	#text5 = 'Approx Max TEST accuracy: {:.2f}%'.format(test_acc_approx)

	figtotal.text(0, -0.05, text1+text2+text3+text4)

	if saveplot:
		figtotal.savefig("Plot.png", bbox_inches = "tight")









### Funcitons not used in final test

### Unit Testing
def unit_test(num_units):
	# truncates imported data to only num_units each
	X_test = np.load("X_test.npy")
	y_test = np.load("y_test.npy")
	person_train_valid = np.load("person_train_valid.npy")
	X_train_valid = np.load("X_train_valid.npy")
	y_train_valid = np.load("y_train_valid.npy")
	person_test = np.load("person_test.npy")

	y_train_valid = y_train_valid - 769
	y_test = y_test - 769

	X_train_valid = X_train_valid[: num_units]
	y_train_valid = y_train_valid[: num_units]
	X_test = X_test[: num_units]
	y_test = y_test[: num_units]


	return X_train_valid, y_train_valid, X_test, y_test


### Print best results
### Used when we run test() in epoch loop
### Not used if we keep track of bestmodel
def best_results(loss_tracker, accuracy_tracker):
	max_val_acc_ind = np.argmax(accuracy_tracker['val'])

	test_acc_maxacc = accuracy_tracker['test'][max_val_acc_ind]

	test_acc_string = 'Test accuracy at the maximum validation accuracy is [{:.2f}%], at epoch {}'.format(test_acc_maxacc, max_val_acc_ind+1)

	print(test_acc_string)	

	return test_acc_maxacc
