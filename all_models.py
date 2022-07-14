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




###################################################################
# DeepConv, from the paper, accepts trials size 500-1000
###################################################################
class DeepConv(nn.Module):
	def __init__(self, in_channels, classes, cnn_drop, trial_length):
		super(DeepConv, self).__init__()
		# self.output_length : length of output cnn
		self.trial_length = trial_length
		self.output_length = trial_length

		for i in range(4):
			self.output_length = int((self.output_length-9)/3)

		# conv block 1
		self.conv1a = nn.Conv2d(in_channels, 25, (1, 10))
		self.batch1a = nn.BatchNorm2d(num_features=25)
		self.conv1b = nn.Conv2d(25, 25, (22, 1))
		self.batch1 = nn.BatchNorm2d(num_features=25)
		self.maxpool1 = nn.MaxPool2d((1, 3), stride=(1, 3))
		self.drop1 = nn.Dropout(p=cnn_drop)

		# conv block 2
		self.conv2 = nn.Conv2d(25, 50, (1, 10))
		self.batch2 = nn.BatchNorm2d(num_features=50)
		self.maxpool2 = nn.MaxPool2d((1, 3), stride=(1, 3))
		self.drop2 = nn.Dropout(p=cnn_drop)

		# conv block 3
		self.conv3 = nn.Conv2d(50, 100, (1, 10))
		self.batch3 = nn.BatchNorm2d(num_features=100)
		self.maxpool3 = nn.MaxPool2d((1, 3), stride=(1, 3))
		self.drop3 = nn.Dropout(p=cnn_drop)

		# conv block 4
		self.conv4 = nn.Conv2d(100, 200, (1, 10))
		self.batch4 = nn.BatchNorm2d(num_features=200)
		self.maxpool4 = nn.MaxPool2d((1, 3), stride=(1, 3))
		#self.drop4 = nn.Dropout(p=cnn_drop)

		# fc1
		self.fc1 = nn.Linear(self.output_length*200, classes)

	def forward(self, x):
		# conv block 1
		x = self.conv1a(x)
		x = self.batch1a(x)
		x = self.conv1b(x)
		x = self.batch1(x)
		x = nn.functional.elu(x)
		x = self.maxpool1(x)
		x = self.drop1(x)

		# conv block 2
		x = self.conv2(x)
		x = self.batch2(x)
		x = nn.functional.elu(x)
		x = self.maxpool2(x)
		x = self.drop2(x)

		# conv block 3
		x = self.conv3(x)
		x = self.batch3(x)
		x = nn.functional.elu(x)
		x = self.maxpool3(x)
		x = self.drop3(x)

		# conv block 4
		x = self.conv4(x)
		x = self.batch4(x)
		x = nn.functional.elu(x)
		x = self.maxpool4(x)

		# fc 1
		x = x.view(-1,self.output_length*200)
		x = self.fc1(x)
		return x


###################################################################
# DeepConvSmall, accepts trials size 200-400
###################################################################

class DeepConvSmall(nn.Module):
	def __init__(self, in_channels, classes, cnn_drop, trial_length):
		super(DeepConvSmall, self).__init__()
		# self.output_length : length of output cnn
		self.trial_length = trial_length
		self.output_length = trial_length

		for i in range(3):
			self.output_length = int((self.output_length-9)/3)

		# conv block 1
		self.conv1a = nn.Conv2d(in_channels, 25, (1, 10))
		self.batch1a = nn.BatchNorm2d(num_features=25)
		self.conv1b = nn.Conv2d(25, 25, (22, 1))
		self.batch1 = nn.BatchNorm2d(num_features=25)
		self.maxpool1 = nn.MaxPool2d((1, 3), stride=(1, 3))
		self.drop1 = nn.Dropout(p=cnn_drop)

		# conv block 2
		self.conv2 = nn.Conv2d(25, 50, (1, 10))
		self.batch2 = nn.BatchNorm2d(num_features=50)
		self.maxpool2 = nn.MaxPool2d((1, 3), stride=(1, 3))
		self.drop2 = nn.Dropout(p=cnn_drop)

		# conv block 3
		self.conv3 = nn.Conv2d(50, 100, (1, 10))
		self.batch3 = nn.BatchNorm2d(num_features=100)
		self.maxpool3 = nn.MaxPool2d((1, 3), stride=(1, 3))
		#self.drop3 = nn.Dropout(p=cnn_drop)

		# fc1
		self.fc1 = nn.Linear(self.output_length*100, classes)

	def forward(self, x):
		# conv block 1
		x = self.conv1a(x)
		x = self.batch1a(x)
		x = self.conv1b(x)
		x = self.batch1(x)
		x = nn.functional.elu(x)
		x = self.maxpool1(x)
		x = self.drop1(x)

		# conv block 2
		x = self.conv2(x)
		x = self.batch2(x)
		x = nn.functional.elu(x)
		x = self.maxpool2(x)
		x = self.drop2(x)

		# conv block 3
		x = self.conv3(x)
		x = self.batch3(x)
		x = nn.functional.elu(x)
		x = self.maxpool3(x)

		# fc 1
		x = x.view(-1,self.output_length*100)

		x = self.fc1(x)
		return x



###################################################################
# DeepConv2, DeepConv from paper + RNN, accepts trials size 500-1000
###################################################################

class DeepConv2(nn.Module):
	def __init__(self, in_channels, classes, cnn_drop):
		super(DeepConv2, self).__init__()

		# conv block 1
		self.conv1a = nn.Conv2d(in_channels, 25, (1, 10))
		self.batch1a = nn.BatchNorm2d(num_features=25)
		self.conv1b = nn.Conv2d(25, 25, (22, 1))
		self.batch1 = nn.BatchNorm2d(num_features=25)
		self.maxpool1 = nn.MaxPool2d((1, 3), stride=(1, 3))
		self.drop1 = nn.Dropout(p=cnn_drop)

		# conv block 2
		self.conv2 = nn.Conv2d(25, 50, (1, 10))
		self.batch2 = nn.BatchNorm2d(num_features=50)
		self.maxpool2 = nn.MaxPool2d((1, 3), stride=(1, 3))
		self.drop2 = nn.Dropout(p=cnn_drop)

		# conv block 3
		self.conv3 = nn.Conv2d(50, 100, (1, 10))
		self.batch3 = nn.BatchNorm2d(num_features=100)
		self.maxpool3 = nn.MaxPool2d((1, 3), stride=(1, 3))
		self.drop3 = nn.Dropout(p=cnn_drop)

		# conv block 4
		self.conv4 = nn.Conv2d(100, 200, (1, 10))
		self.batch4 = nn.BatchNorm2d(num_features=200)
		self.maxpool4 = nn.MaxPool2d((1, 3), stride=(1, 3))
		self.drop4 = nn.Dropout(p=cnn_drop)

		self.rnn1 = nn.LSTM(200, 64, 1, batch_first=True, bidirectional=True)
		self.rnn2 = nn.LSTM(64*2, 32, 1, batch_first=True, bidirectional=True)
		self.rnn3 = nn.LSTM(32*2, 32, 1, batch_first=True, bidirectional=True)

		# fc1
		self.fc1 = nn.Linear(64, classes)

	def forward(self, x):
		# input is (N, C, H, W)
		# conv block 1
		x = self.conv1a(x)
		x = self.batch1a(x)
		x = self.conv1b(x)
		x = self.batch1(x)
		x = nn.functional.elu(x)
		x = self.maxpool1(x)
		x = self.drop1(x)

		# conv block 2
		x = self.conv2(x)
		x = self.batch2(x)
		x = nn.functional.elu(x)
		x = self.maxpool2(x)
		x = self.drop2(x)

		# conv block 3
		x = self.conv3(x)
		x = self.batch3(x)
		x = nn.functional.elu(x)
		x = self.maxpool3(x)
		x = self.drop3(x)

		# conv block 4
		x = self.conv4(x)
		x = self.batch4(x)
		x = nn.functional.elu(x)
		x = self.maxpool4(x)
		x = self.drop4(x) # (N, C, 1, W)

		# lstm (N, seq, feature)
		x = torch.squeeze(x,2)
		x = x.permute(0, 2, 1)
		x , _ = self.rnn1(x)
		x , _ = self.rnn2(x)
		x , _ = self.rnn3(x)
		x = x[:, -1, :]

		x = self.fc1(x)
		return x


###################################################################
# DeepConv2Small, DeepConv from the paper + RNN, accepts trials size 200-400
###################################################################

class DeepConv2Small(nn.Module):
	def __init__(self, in_channels, classes, cnn_drop):
		super(DeepConv2Small, self).__init__()

		# conv block 1
		self.conv1a = nn.Conv2d(in_channels, 25, (1, 10))
		self.batch1a = nn.BatchNorm2d(num_features=25)
		self.conv1b = nn.Conv2d(25, 25, (22, 1))
		self.batch1 = nn.BatchNorm2d(num_features=25)
		self.maxpool1 = nn.MaxPool2d((1, 3), stride=(1, 3))
		self.drop1 = nn.Dropout(p=cnn_drop)

		# conv block 2
		self.conv2 = nn.Conv2d(25, 50, (1, 10))
		self.batch2 = nn.BatchNorm2d(num_features=50)
		self.maxpool2 = nn.MaxPool2d((1, 3), stride=(1, 3))
		self.drop2 = nn.Dropout(p=cnn_drop)

		# conv block 3
		self.conv3 = nn.Conv2d(50, 100, (1, 10))
		self.batch3 = nn.BatchNorm2d(num_features=100)
		self.maxpool3 = nn.MaxPool2d((1, 3), stride=(1, 3))
		self.drop3 = nn.Dropout(p=cnn_drop)

		self.rnn1 = nn.LSTM(100, 64, 1, batch_first=True, bidirectional=True)
		self.rnn2 = nn.LSTM(64*2, 32, 1, batch_first=True, bidirectional=True)
		self.rnn3 = nn.LSTM(32*2, 32, 1, batch_first=True, bidirectional=True)

		# fc1
		self.fc1 = nn.Linear(64, classes)

	def forward(self, x):
		# input is (N, C, H, W)
		# conv block 1
		x = self.conv1a(x)
		x = self.batch1a(x)
		x = self.conv1b(x)
		x = self.batch1(x)
		x = nn.functional.elu(x)
		x = self.maxpool1(x)
		x = self.drop1(x)

		# conv block 2
		x = self.conv2(x)
		x = self.batch2(x)
		x = nn.functional.elu(x)
		x = self.maxpool2(x)
		x = self.drop2(x)

		# conv block 3
		x = self.conv3(x)
		x = self.batch3(x)
		x = nn.functional.elu(x)
		x = self.maxpool3(x)
		x = self.drop3(x)

		# lstm (N, seq, feature)
		x = torch.squeeze(x,2)
		x = x.permute(0, 2, 1)
		x , _ = self.rnn1(x)
		x , _ = self.rnn2(x)
		x , _ = self.rnn3(x)
		x = x[:, -1, :]

		x = self.fc1(x)
		return x





###################################################################
# Conv1
###################################################################

class Conv1(nn.Module):
	def __init__(self, dropout, trial_length):
		super(Conv1, self).__init__()

		self.linear_input = trial_length
		for i in range(2):
			self.linear_input = int((self.linear_input-9)/6)

		self.linear_input = self.linear_input*64

		self.cnn1 = nn.Sequential(
			nn.Conv2d(1, 16, (1, 10)),
			nn.BatchNorm2d(16),
			nn.ELU(),
			nn.Conv2d(16, 32, (22, 1)),
			nn.BatchNorm2d(32),
			nn.ELU(),
			nn.MaxPool2d((1,6)),
			nn.Dropout(p=dropout)
		)        
		self.cnn2 = nn.Sequential(
			nn.Conv2d(32, 64, (1, 10)),
			nn.BatchNorm2d(64),  
			nn.ELU(),
			nn.MaxPool2d((1,6)),
			nn.Dropout(p=dropout)
        )

		self.fullycon = nn.Sequential(
			nn.Linear(self.linear_input, 832),#(256, 128),
			nn.BatchNorm1d(832),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.Linear(832, 416),#(128, 64),
			nn.BatchNorm1d(416),
			nn.ReLU(),
			nn.Linear(416,4)
		)


	def forward(self, x):
		x = self.cnn1(x)
		x = self.cnn2(x)        
		x = torch.squeeze(x)
		x = x.permute(0,2,1)

		x = x.reshape(-1, self.linear_input)#(-1,256)
		x = self.fullycon(x)
		return x

###################################################################
# Conv2
###################################################################

class Conv2(nn.Module):
	def __init__(self, dropout):
		super(Conv2, self).__init__()
		self.cnn1 = nn.Sequential(
			nn.Conv2d(1, 16, (1, 10)),
			nn.BatchNorm2d(16),
			nn.ELU(),
			nn.Conv2d(16, 32, (22, 1), 1),
			nn.BatchNorm2d(32),
			nn.ELU(),
			nn.MaxPool2d((1,6)),
			nn.Dropout(p=dropout)
		)        
		self.cnn2 = nn.Sequential(
			nn.Conv2d(32, 64, (1, 10)),
			nn.BatchNorm2d(64),  
			nn.ELU(),
			nn.MaxPool2d((1,6)),
			nn.Dropout(p=dropout)
        )

		self.rnn1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
		self.rnn2 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
		self.fc = nn.Linear(32, 4)



	def forward(self, x):
		x = self.cnn1(x)
		x = self.cnn2(x)        
		x = torch.squeeze(x)
		x = x.permute(0,2,1)

		x, _ = self.rnn1(x)
		x, _ = self.rnn2(x)
		x = x[:, -1, :]
		x = self.fc(x)
		return x


