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


def train(model, optimizer, loss_fn, dataloaders, device, epoch):
	batch_count = 0
	epoch_loss = 0
	correct = 0
	total = 0
	model.train()

	for data in dataloaders['train']:
		inputs, labels = data[0].to(device), data[1].to(device)
		batch_count += 1

		# zero gradient
		optimizer.zero_grad()
		# forward pass
		yhats = model(inputs)
		# compute loss
	
		loss = loss_fn(yhats,labels)
		loss.backward()

		optimizer.step()

		# update this epoch's loss
		epoch_loss += loss.item()
		# update this epoch's accuracy
		_, predicted = torch.max(yhats.data,1)
		total += labels.shape[0]
		correct += (predicted == labels).sum().item()

	# update this epoch's info
	epoch_loss_avg = epoch_loss/batch_count
	epoch_accuracy = correct/total*100
	if (epoch+1)%10 == 0:
		print('Epoch [{}], Training loss [{:.4f}], Training accuracy [{:.2f}%]'.format(epoch+1, epoch_loss_avg, epoch_accuracy))
		#print('Training epoch {}, Avg epoch loss {:.4f}'.format(epoch+1, epoch_loss_avg))

	return model, epoch_loss_avg, epoch_accuracy


def validate(model, optimizer, loss_fn, dataloaders, device, epoch, best_model_params, cut=False):
	# NOTE: this function is for VALIDATION only
	# NOTE ON CROP AUGMENTATION: 
	# 		crop augmentation reports regular val loss (as it is hard to copmute loss with voting method)
	#		but reports special accuracy method, with additional voting method for final classification

	batch_count = 0
	epoch_loss = 0
	correct = 0
	total = 0
	model.eval()
	
	with torch.no_grad():
		for data in dataloaders['val']:

			inputs, labels = data[0].to(device), data[1].to(device)
			batch_count += 1
			#print('CHECK: val/test getting loading one trial at a time:')
			#print('Labels should be the same: {}'.format(labels))
			#print('Only 11 datapoints should be passed in: {}'.format(labels.shape[0]))

			# zero gradient
			optimizer.zero_grad()
			# forward pass
			yhats = model(inputs)
			# compute loss
			loss = loss_fn(yhats,labels)

			# update this epoch's loss
			epoch_loss += loss.item()

			# update this epoch's accuracy
			_, predicted = torch.max(yhats.data,1)

			if cut: # predict accuracy SPECIAL METHOD				
				#print('Predicted vector of 11: {}'.format(predicted))
				# find class with most votes
				final_prediction = torch.argmax(torch.bincount(predicted))
				#print('Final prediction: {}'.format(final_prediction))
				onelabel = labels[0] # all labels should be the same, just get one

				total += 1
				correct += (final_prediction == onelabel)


			else: # predict accuracy NORMAL METHOD
				total += labels.shape[0]
				correct += (predicted==labels).sum().item()

		# update this epoch's info
		epoch_loss_avg = epoch_loss/batch_count
		epoch_accuracy = correct/total*100
		if (epoch+1)%10 == 0:
			print('Epoch [{}], Validation loss [{:.4f}], Validation accuracy [{:.2f}%]'.format(epoch+1, epoch_loss_avg, epoch_accuracy))
			#print('Validation epoch {}, Average loss {:.4f}'.format(epoch+1, epoch_loss_avg))


		# keep track of bestmodel based off of validation accuracy
		bestValidationAcc, _ = best_model_params
		if (epoch_accuracy >= bestValidationAcc):
			best_model_params = [ epoch_accuracy,model]

	return epoch_loss_avg, epoch_accuracy, best_model_params



def finaltest(bestmodel, dataloaders, device, cut=False):
	# NOTE: this function is for TEST only
	# NOTE ON CROP AUGMENTATION: 
	#		reports special accuracy method, with additional voting method for final classification
	
	correct = 0
	total = 0
	bestmodel.eval()

	with torch.no_grad():
		for data in dataloaders['test']:
			inputs, labels = data[0].to(device), data[1].to(device)
			#print('CHECK: val/test getting loading one trial at a time:')
			#print('Labels should be the same: {}'.format(labels))
			#print('Only 11 datapoints should be passed in: {}'.format(labels.shape[0]))

			# forward pass
			yhats = bestmodel(inputs)
			_, predicted = torch.max(yhats.data,1)

			if cut: # predict accuracy SPECIAL METHOD				
				#print('Predicted vector of 11: {}'.format(predicted))
				# find class with most votes
				final_prediction = torch.argmax(torch.bincount(predicted))
				#print('Final prediction: {}'.format(final_prediction))
				onelabel = labels[0] # all labels should be the same, just get one

				total += 1
				correct += (final_prediction == onelabel)


			else: # predict accuracy NORMAL METHOD
				total += labels.size(0)
				correct += (predicted==labels).sum().item()


	final_test_accuracy = 100*correct/total
	print('Test accuracy [{:.2f}%]'.format(final_test_accuracy))
	return final_test_accuracy






# functions not used in final test


def test(model, dataloaders, device, epoch, cut=False):
	# NOTE: this function is for TEST only
	# NOTE ON CROP AUGMENTATION: 
	#		reports special accuracy method, with additional voting method for final classification
	
	correct = 0
	total = 0
	model.eval()

	with torch.no_grad():
		for data in dataloaders['test']:
			inputs, labels = data[0].to(device), data[1].to(device)
			#print('CHECK: val/test getting loading one trial at a time:')
			#print('Labels should be the same: {}'.format(labels))
			#print('Only 11 datapoints should be passed in: {}'.format(labels.shape[0]))

			# forward pass
			yhats = model(inputs)
			_, predicted = torch.max(yhats.data,1)

			if cut: # predict accuracy SPECIAL METHOD				
				#print('Predicted vector of 11: {}'.format(predicted))
				# find class with most votes
				final_prediction = torch.argmax(torch.bincount(predicted))
				#print('Final prediction: {}'.format(final_prediction))
				onelabel = labels[0] # all labels should be the same, just get one

				total += 1
				correct += (final_prediction == onelabel)


			else: # predict accuracy NORMAL METHOD
				total += labels.size(0)
				correct += (predicted==labels).sum().item()


	epoch_test_accuracy = 100*correct/total
	if (epoch+1)%10 == 0:
		#print('Test accuracy: {}'.format(test_accuracy))
		print('Epoch [{}], Test accuracy [{:.2f}%]'.format(epoch+1, epoch_test_accuracy))
	return epoch_test_accuracy
