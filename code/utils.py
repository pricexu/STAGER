import random
import argparse
import numpy as np
import pickle as pkl
import networkx as nx
import torch
import torch.nn as nn
import scipy.sparse as sp
import scipy.io as sio
from sklearn import preprocessing
from sklearn.metrics import f1_score
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
	torch.cuda.manual_seed(1234)

valid_num_dic = {'Amazon_clothing': 17, 'Amazon_eletronics': 36, 'dblp': 27}

def accuracy(output, labels):
	preds = output.max(1)[1].type_as(labels)
	correct = preds.eq(labels).double()
	correct = correct.sum()
	return correct / len(labels)

def accuracy_binary(output, labels):
	correct = output.eq(labels).double()
	correct = correct.sum()
	return correct / len(labels)

def f1(output, labels):
	preds = output.max(1)[1].type_as(labels)
	f1 = f1_score(labels, preds, average='weighted')
	return f1

def load_adj_features_labels(file_name):

	with np.load(file_name) as loader:
		# loader = dict(loader)
		adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
									loader['adj_indptr']), shape=loader['adj_shape'])

		features = sp.csr_matrix((loader['features_data'], loader['features_indices'],
									loader['features_indptr']), shape=loader['features_shape'])

		# features = loader.get('features')

		labels = loader.get('labels')

	return adj,features.toarray(),labels

def load_data(dataset_source, data_folder):
	n1s = []
	n2s = []
	for line in open("{}/{}_network".format(data_folder, dataset_source)):
		n1, n2 = line.strip().split('\t')
		n1s.append(int(n1))
		n2s.append(int(n2))

	num_nodes = max(max(n1s),max(n2s)) + 1
	adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
								 shape=(num_nodes, num_nodes))

	data_train = sio.loadmat("{}/{}_train.mat".format(data_folder, dataset_source))
	train_class = list(set(data_train["Label"].reshape((1,len(data_train["Label"])))[0]))


	data_test = sio.loadmat("{}/{}_test.mat".format(data_folder, dataset_source))
	class_list_test = list(set(data_test["Label"].reshape((1,len(data_test["Label"])))[0]))


	labels = np.zeros((num_nodes,1))
	labels[data_train['Index']] = data_train["Label"]
	labels[data_test['Index']] = data_test["Label"]
	labels = labels.flatten().astype(int)

	features = np.zeros((num_nodes,data_train["Attributes"].shape[1]))
	features[data_train['Index']] = data_train["Attributes"].toarray()
	features[data_test['Index']] = data_test["Attributes"].toarray()

	class_list = list(set(labels))

	id_by_class = {}
	for i in class_list:
		id_by_class[i] = []
	for id, cla in enumerate(labels):
		id_by_class[cla].append(id)
	# it is a mapping from class 2 id

	# lb = preprocessing.LabelBinarizer()
	# labels = lb.fit_transform(labels)
	#
	# adj = normalize_adj(adj + sp.eye(adj.shape[0]))
	# features = torch.FloatTensor(features)
	# labels = torch.LongTensor(np.where(labels)[1])
	#
	# adj = sparse_mx_to_torch_sparse_tensor(adj)

	class_list_valid = random.sample(train_class, valid_num_dic[dataset_source])

	class_list_train = list(set(train_class).difference(set(class_list_valid)))

	return adj, features, labels, class_list_train, class_list_valid, class_list_test, id_by_class

def task_generator_old(id_by_class, class_list, n_way, k_shot, n_query, is_train=True):

	class_selected = random.sample(class_list, n_way)

	# sample support examples
	id_support = []
	id_by_class_remain = {}
	for cla in class_selected:
		temp = random.sample(id_by_class[cla], k_shot)
		id_support.extend(temp)
		# id_remain.extend(list(set(id_by_class[cla]).difference(set(temp))))
		if cla in id_by_class_remain:
			id_by_class_remain[cla].extend(list(set(id_by_class[cla]).difference(set(temp))))
		else:
			id_by_class_remain[cla] = list(set(id_by_class[cla]).difference(set(temp)))

	# sample query examples
	id_query = []
	for cla in class_selected:
		if n_query > len(id_by_class_remain[cla]):
			id_query.extend(id_by_class_remain[cla])
		else:
			id_query.extend(random.sample(id_by_class_remain[cla], n_query))

	return np.array(id_support), np.array(id_query), class_selected

def task_generator(class2id, classes, n_way, k_shot, n_query):
	classes_selected = random.sample(classes, n_way)
	id_support = []
	id_query = []
	support_class2id = {}
	query_class2id = {}
	for class_selected in classes_selected:
		ids = class2id[class_selected]
		np.random.shuffle(ids)
		id_support.extend(ids[:k_shot])
		id_query.extend(ids[k_shot:k_shot+n_query])
		support_class2id[class_selected] = ids[:k_shot]
		query_class2id[class_selected] = ids[k_shot:k_shot+n_query]

	return np.array(id_support), np.array(id_query), support_class2id, query_class2id


def load_raw_graph(data_folder):
	bin_file = "{}/ind.{}".format(data_folder, 'graph')
	if os.path.isfile(bin_file):
		with open(bin_file, 'rb') as f:
			if sys.version_info > (3, 0):
				graph = pkl.load(f, encoding='latin1')
			else:
				graph = pkl.load(f)
	return graph

def load_data_old(data_folder):
	idx_train = list(np.loadtxt(data_folder + '/train_idx.txt', dtype=int))
	idx_val = list(np.loadtxt(data_folder + '/val_idx.txt', dtype=int))
	idx_test = list(np.loadtxt(data_folder + '/test_idx.txt', dtype=int))
	labels = np.loadtxt(data_folder + '/label.txt')

	with open(data_folder + '/meta.txt', 'r') as f:
		num_nodes, num_class, feature_dim = [int(w) for w in f.readline().strip().split()]

	graph = load_raw_graph(data_folder)
	assert len(graph) == num_nodes
	graph = nx.from_dict_of_lists(graph)

	row_ptr = []
	col_idx = []
	vals = []
	with open(data_folder + '/features.txt', 'r') as f:
		nnz = 0
		for row in f:
			row = row.strip().split()
			row_ptr.append(nnz)
			for i in range(1, len(row)):
				w = row[i].split(':')
				col_idx.append(int(w[0]))
				vals.append(float(w[1]))
			nnz += int(row[0])
		row_ptr.append(nnz)
	assert len(col_idx) == len(vals) and len(vals) == nnz and len(row_ptr) == num_nodes + 1

	features = preprocess_features(sp.csr_matrix((vals, col_idx, row_ptr), shape=(num_nodes, feature_dim)))

	features = torch.FloatTensor(np.array(features.todense()))
	adjacency_matrix = nx.to_scipy_sparse_matrix(graph)
	adjacency_matrix = sp.lil_matrix(adjacency_matrix)
	# adjacency_matrix = preprocess_adj(nx.to_scipy_sparse_matrix(graph))
	# adjacency_matrix = sparse_mx_to_torch_sparse_tensor(adjacency_matrix)
	idx_train = torch.LongTensor(idx_train)
	idx_val = torch.LongTensor(idx_val)
	idx_test = torch.LongTensor(idx_test)

	return adjacency_matrix, features, labels, idx_train, idx_val, idx_test, graph

def preprocess_features(features):
	"""Row-normalize feature matrix and convert to tuple representation"""
	rowsum = np.array(features.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	features = r_mat_inv.dot(features)

	return features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
	"""Convert a scipy sparse matrix to a torch sparse tensor."""
	sparse_mx = sparse_mx.tocoo().astype(np.float32)
	indices = torch.from_numpy(
		np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
	values = torch.from_numpy(sparse_mx.data)
	shape = torch.Size(sparse_mx.shape)
	return torch.sparse.FloatTensor(indices, values, shape)

def sparse_mx_to_torch_tensor(sparse_mx):
	"""Convert a scipy sparse matrix to a torch tensor."""
	return torch.FloatTensor(sparse_mx.astype(np.float32).toarray())

def normalize_adj(adj):
	"""Symmetrically normalize adjacency matrix."""
	adj = sp.coo_matrix(adj)
	rowsum = np.array(adj.sum(1))
	d_inv_sqrt = np.power(rowsum, -0.5).flatten()
	d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
	d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
	return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
	"""Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
	adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
	return adj_normalized

def preprocess_adj_row(adj):
	"""Row-normalize feature matrix and convert to tuple representation"""
	adj = sp.coo_matrix(adj)
	rowsum = np.array(adj.sum(1))
	d_inv_sqrt = np.power(rowsum, -1).flatten()
	d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
	d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
	adj = d_mat_inv_sqrt.dot(adj)

	return adj

def normalize_adj_tensor(adj):
	"""Symmetrically normalize adjacency tensor."""
	rowsum = torch.sum(adj,1)
	d_inv_sqrt = torch.pow(rowsum, -0.5)
	d_inv_sqrt[d_inv_sqrt == float("Inf")] = 0.
	d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
	return torch.mm(torch.mm(adj,d_mat_inv_sqrt).transpose(0,1),d_mat_inv_sqrt)

def preprocess_adj_tensor(adj):
	"""Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
	adj_normalized = normalize_adj_tensor(adj + torch.eye(adj.shape[0]).to(device))
	return adj_normalized

def cross_entropy(pred, soft_targets):
	logsoftmax = nn.LogSoftmax(dim=1)
	return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
