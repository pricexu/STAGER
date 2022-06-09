import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import time

class Weight_Assigner(nn.Module):
	def __init__(self, nhid, nclass, adj, dropout=0.5, topk=16):
		super(Weight_Assigner, self).__init__()
		self.adj = adj
		self.degree = 6
		self.fc1 = nn.Linear(topk, nhid)
		self.fc2 = nn.Linear(nhid, self.degree)
		self.dropout = dropout
		self.topk = topk


	def forward(self, x):

		output_pre = x
		t_total = time.time()

		x = F.softmax(x, dim=1)
		# x = x[:,torch.randperm(x.size(1))] # shuffle the input to push them get the confidence info
		x = torch.topk(x, k=self.topk, dim=1)[0]
		x = F.leaky_relu(self.fc1(x),0.1)
		x = F.dropout(x, self.dropout, training=self.training)
		x = self.fc2(x)
		x = F.dropout(x, self.dropout, training=self.training)
		weight = F.softmax(x, dim=1)
		output = weight[:,0].unsqueeze(1).expand(-1, output_pre.size(1))*output_pre

		for i in range(self.degree-1):
			output_pre = torch.spmm(self.adj, output_pre)
			output += weight[:,i+1].unsqueeze(1).expand(-1, output_pre.size(1))*output_pre
		return F.log_softmax(output, dim=1)

class Feature_Extractor(nn.Module):
	def __init__(self, nfeat, nhid, nclass, dropout):
		super(Feature_Extractor, self).__init__()

		self.preliminary_classifier = nn.Sequential(
			nn.Linear(nfeat, nhid),
			# nn.ReLU(),
			nn.LeakyReLU(negative_slope=0.1),
			nn.Dropout(p=dropout),
			nn.Linear(nhid, nclass)
		)

	def forward(self, x):
		return self.preliminary_classifier(x)
