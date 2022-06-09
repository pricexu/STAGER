import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import time


class GraphConvolution(Module):
	"""
	Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
	"""

	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input, adj):
		support = torch.mm(input, self.weight)
		output = torch.spmm(adj, support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
				+ str(self.in_features) + ' -> ' \
				+ str(self.out_features) + ')'

class GCN(nn.Module):
	def __init__(self, nfeat, nhid, nclass, dropout):
		super(GCN, self).__init__()

		self.gc1 = GraphConvolution(nfeat, nhid)
		self.gc2 = GraphConvolution(nhid, nclass)
		self.dropout = dropout

	def forward(self, x, adj):
		x = F.relu(self.gc1(x, adj))
		x = F.dropout(x, self.dropout, training=self.training)
		x = self.gc2(x, adj)
		return F.log_softmax(x, dim=1)

class SGC(nn.Module):
	def __init__(self, nfeat, nhid, nclass, dropout):
		super(SGC, self).__init__()

		self.gc1 = GraphConvolution(nfeat, nhid)
		self.gc2 = GraphConvolution(nhid, nclass)
		self.dropout = dropout

	def forward(self, x, adj):
		x = self.gc1(x, adj)
		x = self.gc2(x, adj)
		return F.log_softmax(x, dim=1)

class APPNP(nn.Module):
	def __init__(self, nfeat, nhid, nclass, dropout):
		super(APPNP, self).__init__()

		self.linear1 = nn.Linear(nfeat, nhid)
		self.linear2 = nn.Linear(nhid, nclass)
		self.dropout = dropout

	def forward(self, x, adj):
		alpha = 0.1
		x = F.relu(self.linear1(x))
		x = F.dropout(x, self.dropout, training=self.training)
		x = self.linear2(x)
		# x = F.dropout(x, self.dropout, training=self.training)

		x_k = x

		for _ in range(10):
			x_k = alpha*x + (1-alpha)*torch.spmm(adj, x_k)

		return F.log_softmax(x_k, dim=1)

class GPR_GNN(nn.Module):
	def __init__(self, nfeat, nhid, nclass, dropout):
		super(GPR_GNN, self).__init__()

		self.linear1 = nn.Linear(nfeat, nhid)
		self.linear2 = nn.Linear(nhid, nclass)
		self.dropout = dropout
		self.distance_weight = Parameter(torch.FloatTensor([1,1,1,1,1]))


	def forward(self, x, adj):
		# print(self.distance_weight)
		x = F.relu(self.linear1(x))
		x = F.dropout(x, self.dropout, training=self.training)
		x = self.linear2(x)

		output = torch.zeros_like(x)

		for i in range(5):
			x = torch.spmm(adj, x)
			output += self.distance_weight[i]*x

		return F.log_softmax(output, dim=1)

class Classifier(nn.Module):
	def __init__(self, nhid, nclass):
		super(Classifier, self).__init__()

		self.linear = nn.Linear(nhid, nclass)

	def forward(self, x):
		classification = F.log_softmax(self.linear(x), dim=1)
		return  classification

class Classifier_auxiliary(nn.Module):
	def __init__(self, nhid):
		super(Classifier_auxiliary, self).__init__()

		self.linear1 = nn.Linear(nhid, nhid)
		self.linear2 = nn.Linear(nhid, nhid)
		# self.linear3 = nn.Linear(nhid, nhid)
		# self.linear4 = nn.Linear(nhid, nhid)
		self.linear5 = nn.Linear(nhid, 1)

	def forward(self, x):
		x = F.leaky_relu(self.linear1(x),0.1)
		x = F.leaky_relu(self.linear2(x),0.1)+x
		# x = F.leaky_relu(self.linear3(x),0.01)+x
		classification = torch.sigmoid(self.linear5(x))
		return  classification

class GCN_Encoder(nn.Module):
	def __init__(self, nfeat, nhid, dropout):
		super(GCN_Encoder, self).__init__()

		self.gc1 = GraphConvolution(nfeat, nhid)
		self.gc2 = GraphConvolution(nhid, nhid)
		self.dropout = dropout

	def forward(self, x, adj):
		encoding = F.relu(self.gc1(x, adj))
		encoding = F.dropout(encoding, self.dropout, training=self.training)
		# encoding = self.gc2(encoding, adj)
		encoding = torch.spmm(adj, encoding)
		return encoding

class Weight_Assigner(nn.Module):
	def __init__(self, nhid, nclass, adj, dropout=0.5, topk=16):
		super(Weight_Distributer_new, self).__init__()
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

class Weight_Distributer_decoupled(nn.Module):
	def __init__(self, nhid, nclass, adj, dropout=0.5, topk=16):
		super(Weight_Distributer_decoupled, self).__init__()
		self.adj = adj
		self.degree = 9
		self.topk = topk
		self.g1 = nn.Sequential(
			nn.Linear(topk, nhid*2),
			# nn.LeakyReLU(negative_slope=0.01),
			nn.Tanh(),
			# nn.Dropout(p=dropout),
			nn.Linear(nhid*2, nhid),
			# nn.LeakyReLU(negative_slope=0.01),
			nn.Tanh(),
			# nn.Dropout(p=dropout),
			nn.Linear(nhid, self.degree),
			# nn.Dropout(p=dropout),
		)

	def forward(self, f_prediction, g0_prediction):

		t_total = time.time()

		g0_prediction = F.softmax(g0_prediction, dim=1)

		ranked_g0_prediction = torch.topk(g0_prediction, k=self.topk, dim=1)[0]
		weight = self.g1(ranked_g0_prediction)
		# weight = F.softmax(weight, dim=1)
		# weight = torch.tanh(weight)
		final_prediction = weight[:,0].unsqueeze(1).expand(-1, f_prediction.size(1))*f_prediction

		for i in range(self.degree-1):
			f_prediction = torch.spmm(self.adj, f_prediction)
			final_prediction += weight[:,i+1].unsqueeze(1).expand(-1, f_prediction.size(1))*f_prediction

		return F.log_softmax(final_prediction, dim=1), weight

class Learner(nn.Module):
	def __init__(self, config):
		super(Learner, self).__init__()
		self.config = config
		self.vars = nn.ParameterList()
		for i, param in enumerate(self.config):
			w = nn.Parameter(torch.ones(*param))
			torch.nn.init.kaiming_normal_(w)
			self.vars.append(w)
			self.vars.append(nn.Parameter(torch.zeros(param[0])))

	def forward(self, x, vars=None):
		if vars is None:
			vars = self.vars

		w1, b1, w2, b2 = vars
		x = F.relu(F.linear(x, w1, b1))
		x = F.linear(x, w2, b2)
		return F.log_softmax(x, dim=1)

	def zero_grad(self, vars=None):
		with torch.no_grad():
			if vars is None:
				for p in self.vars:
					if p.grad is not None:
						p.grad.zero_()
			else:
				for p in vars:
					if p.grad is not None:
						p.grad.zero_()

	def parameters(self):
		return self.vars

class Learner_ammgnn(nn.Module):
	def __init__(self, config, num_node, feature_dim):
		super(Learner_ammgnn, self).__init__()
		self.config = config
		self.vars = nn.ParameterList()
		for i, param in enumerate(self.config):
			w = nn.Parameter(torch.ones(*param))
			torch.nn.init.kaiming_normal_(w)
			self.vars.append(w)
			self.vars.append(nn.Parameter(torch.zeros(param[0])))
		self.g_phi_alpha_fc1 = nn.Linear(num_node, 16)
		self.g_phi_alpha_fc2 = nn.Linear(16, 1)
		self.g_phi_beta_fc1 = nn.Linear(num_node, 16)
		self.g_phi_beta_fc2 = nn.Linear(16, 1)

	def forward(self, x, x_alpha_beta, vars=None):
		if vars is None:
			vars = self.vars
		x_alpha_beta_prime = x_alpha_beta.transpose(0,1)
		alpha = torch.sigmoid(self.g_phi_alpha_fc2(torch.tanh(self.g_phi_alpha_fc1(x_alpha_beta_prime))))
		beta = torch.sigmoid(self.g_phi_beta_fc2(torch.tanh(self.g_phi_beta_fc1(x_alpha_beta_prime))))

		x_prime = x.transpose(0,1)
		alpha = alpha.expand(-1, x_prime.size(1))
		beta = beta.expand(-1, x_prime.size(1))
		x_prime = x_prime*(1+alpha)+beta
		x = x_prime.transpose(0,1)

		w1, b1, w2, b2 = vars
		x = F.relu(F.linear(x, w1, b1))
		x = F.linear(x, w2, b2)
		return F.log_softmax(x, dim=1)

	def zero_grad(self, vars=None):
		with torch.no_grad():
			if vars is None:
				for p in self.vars:
					if p.grad is not None:
						p.grad.zero_()
			else:
				for p in vars:
					if p.grad is not None:
						p.grad.zero_()

	def parameters(self):
		return self.vars
