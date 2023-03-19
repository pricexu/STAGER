import os, sys
import time
import random
import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import block_diag

from copy import deepcopy
from utils import *
from models import *

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--meta_training_epochs', type=int, default=300,
					help='Number of epochs to meta train.')
parser.add_argument('--lr', type=float, default=0.01,
					help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0,
					help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=120,
					help='Number of hidden units.')
parser.add_argument('--topk', type=int, default=32,
					help='The number of k logits as the input of the weight assigner.')
parser.add_argument('--episodes', type=int, default=11,
					help='Number of episodes.')
parser.add_argument('--dropout', type=float, default=0.5,
					help='Dropout rate (1 - keep probability).')
parser.add_argument('--sample_weight', type=float, default=100,
					help='The sample reweighting for the novel classes.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)


def train(idx_support, idx_query_base, idx_query_novel, feature_extractor, weight_distributer, optimizer_feature_extractor,
			optimizer_weight_distributer, sample_weight):

	idx_query = torch.cat((idx_query_base, idx_query_novel),0)

	for _ in range(20):

		feature_extractor.train()
		weight_distributer.eval()
		emb = feature_extractor(features_tensor)
		classification = weight_distributer(emb)

		optimizer_feature_extractor.zero_grad()
		optimizer_weight_distributer.zero_grad()
		loss_inner = F.nll_loss(classification[idx_support], labels_tensor[idx_support], weight=sample_weight)
		loss_inner.backward()
		optimizer_feature_extractor.step()

	feature_extractor.eval()
	weight_distributer.train()
	with torch.no_grad():
		emb = feature_extractor(features_tensor)

	classification = weight_distributer(emb)

	optimizer_feature_extractor.zero_grad()
	optimizer_weight_distributer.zero_grad()
	loss_outer = F.nll_loss(classification[idx_query], labels_tensor[idx_query], weight=sample_weight)
	loss_outer.backward()
	optimizer_weight_distributer.step()

	acc_train_base = accuracy(classification[idx_query_base], labels_tensor[idx_query_base])
	acc_train_novel = accuracy(classification[idx_query_novel], labels_tensor[idx_query_novel])
	acc_train = accuracy(classification[idx_query], labels_tensor[idx_query])
	return acc_train_base, acc_train_novel, acc_train


def test(idx_support, idx_test_base, idx_test_novel, feature_extractor, weight_distributer, optimizer_feature_extractor,
			optimizer_weight_distributer, sample_weight):

	idx_query = torch.cat((idx_test_base, idx_test_novel),0)

	for _ in range(300):

		# fine tuning
		feature_extractor.train()
		weight_distributer.train()
		emb = feature_extractor(features_tensor)
		classification = weight_distributer(emb)

		optimizer_feature_extractor.zero_grad()
		optimizer_weight_distributer.zero_grad()
		loss_inner = F.nll_loss(classification[idx_support], labels_tensor[idx_support], weight=sample_weight)
		loss_inner.backward()
		optimizer_feature_extractor.step()
		optimizer_weight_distributer.step()


	feature_extractor.eval()
	weight_distributer.eval()
	emb = feature_extractor(features_tensor)
	classification = weight_distributer(emb)

	acc_test_base = accuracy(classification[idx_test_base], labels_tensor[idx_test_base])
	acc_test_novel = accuracy(classification[idx_test_novel], labels_tensor[idx_test_novel])
	acc_test = accuracy(classification[idx_query], labels_tensor[idx_query])

	return acc_test_base, acc_test_novel, acc_test


if __name__ == "__main__":

	settings = [(5,1),(5,3),(10,1),(10,3)]
	n_query = 50
	meta_test_num = 10
	meta_valid_num = 10
	episodes = args.episodes

	print()
	print('weight_decay: {}'.format(args.weight_decay))
	print('lr: {}'.format(args.lr))
	print('hidden: {}'.format(args.hidden))
	print()

	data_folder = '../data/'
	dataset_sources = ['Amazon_clothing', 'Amazon_eletronics', 'dblp', 'cora_full']
	# # of labels 77, 167, 137

	for dataset_source in dataset_sources:
		# fix the testing classes for stable performance as the difference
		# between classes are significant
		if dataset_source == 'Amazon_clothing':
			test_classes_org = [43, 30, 26, 23, 49, 15, 24, 53, 38, 47]
			validation_classes_org = [54, 9, 16, 5, 28, 48, 56, 58, 69, 68]
			training_classes_org = [22, 33, 31, 72, 40, 35, 4, 51, 25, 52, 21, 73, 45, 55, 42, 10, 18, 6, 27, 8, 64, 67, 66, 29, 65, 32, 60, 57, 46, 20, 13, 70, 36, 61, 7, 14, 17, 44, 1, 74, 19, 2, 71, 59, 63, 41, 0, 11, 39, 3, 75, 34, 37, 12, 62, 50, 76]
		elif dataset_source == 'Amazon_eletronics':
			test_classes_org = [84, 62, 151, 78, 55, 89, 103, 82, 4, 28]
			validation_classes_org = [116, 119, 21, 154, 114, 90, 98, 121, 0, 105]
			training_classes_org = [60, 94, 31, 53, 128, 6, 91, 73, 129, 5, 136, 111, 162, 74, 48, 65, 7, 10, 32, 63, 132, 97, 51, 122, 9, 146, 110, 13, 52, 155, 29, 131, 125, 143, 152, 165, 14, 95, 99, 135, 40, 148, 86, 58, 108, 137, 106, 120, 93, 44, 43, 71, 112, 70, 66, 54, 77, 102, 50, 23, 34, 37, 104, 145, 59, 67, 85, 39, 92, 160, 41, 138, 45, 22, 149, 164, 47, 117, 113, 12, 24, 56, 25, 38, 141, 17, 127, 11, 144, 166, 163, 147, 124, 115, 118, 27, 101, 100, 158, 49, 126, 83, 96, 61, 134, 16, 20, 133, 75, 8, 80, 81, 69, 76, 18, 35, 157, 139, 87, 2, 42, 150, 33, 88, 36, 161, 57, 153, 46, 79, 107, 130, 72, 15, 64, 123, 68, 26, 19, 109, 140, 30, 3, 142, 1, 159, 156]
		elif dataset_source == 'dblp':
			test_classes_org = [14, 93, 114, 52, 77, 130, 51, 2, 127, 1]
			validation_classes_org = [126, 17, 116, 109, 63, 69, 41, 8, 102, 60]
			training_classes_org = [83, 86, 97, 134, 7, 33, 28, 27, 119, 121, 118, 35, 92, 135, 44, 22, 10, 111, 59, 32, 38, 82, 117, 73, 110, 101, 36, 21, 125, 103, 71, 131, 19, 79, 123, 62, 85, 132, 43, 26, 30, 98, 37, 3, 68, 61, 13, 54, 56, 76, 107, 5, 115, 133, 70, 25, 15, 112, 67, 105, 12, 66, 64, 108, 87, 49, 106, 42, 0, 91, 20, 136, 124, 45, 40, 120, 84, 72, 90, 39, 48, 18, 99, 113, 46, 4, 57, 78, 81, 6, 16, 58, 23, 100, 55, 104, 94, 24, 88, 89, 53, 34, 11, 74, 95, 31, 128, 50, 80, 96, 9, 29, 65, 75, 47, 129, 122]
		elif dataset_source == 'cora_full':
			test_classes_org = [36, 58, 10, 26, 8, 61, 64, 16, 22, 38]
			validation_classes_org = [52, 31, 6, 24, 56, 33, 25, 41, 13, 9]
			training_classes_org = [62, 17, 21, 29, 32, 39, 2, 48, 66, 3, 45, 51, 46, 0, 14, 5, 44, 63, 11, 20, 7, 34, 53, 59, 57, 37, 30, 55, 27, 40, 19, 28, 15, 49, 23, 60]

		for n_way, k_shot in settings:
			print('************************************************************************************')
			print('************************************************************************************')
			print('Dataset: '+dataset_source)
			print('n_way: '+str(n_way))
			print('k_shot: '+str(k_shot))

			adj, features, labels = load_adj_features_labels(data_folder+dataset_source+'.npz')
			n_class = max(labels)+1
			class2id = {}
			for id, label in enumerate(labels):
				if label not in class2id:
					class2id[label] = [id]
				else:
					class2id[label].append(id)
			print('# of nodes: '+str(features.shape[0]))
			print('# of edges: '+str(adj.count_nonzero()//2))
			print('# of features: '+str(features.shape[1]))
			print('# of labels: '+str(n_class))

			test_classes = test_classes_org[:n_way]
			validation_classes = validation_classes_org[:n_way]
			training_classes = training_classes_org

			adj_normalized = preprocess_adj(adj)
			adj_normalized_tensor = sparse_mx_to_torch_sparse_tensor(adj_normalized).to(device)
			features_tensor = torch.FloatTensor(features).to(device)
			labels_tensor = torch.LongTensor(labels).to(device)

			# Train model
			t_total = time.time()

			id_support_base, id_query_base, base_support_class2id, base_query_class2id = \
				task_generator(class2id, training_classes, len(training_classes), 50, n_query*len(test_classes)//len(training_classes))
			id_support_novel, id_query_novel, novel_support_class2id, novel_query_class2id = \
				task_generator(class2id, test_classes, len(test_classes), k_shot, n_query)

			id_support_base = torch.LongTensor(id_support_base).to(device)
			id_query_base = torch.LongTensor(id_query_base).to(device)
			id_support_novel = torch.LongTensor(id_support_novel).to(device)
			id_query_novel = torch.LongTensor(id_query_novel).to(device)

			sample_weight_test = []
			for i in range(n_class):
				if i in test_classes or i in validation_classes:
					sample_weight_test.append(args.sample_weight/k_shot)
					# sample reweighting for novel classes
				else:
					sample_weight_test.append(1)
			sample_weight_test = torch.tensor(sample_weight_test).to(device)


			weight_distributer = Weight_Assigner(nhid=args.hidden,
			 			nclass=n_class,
						adj=adj_normalized_tensor,
						dropout=args.dropout,
						topk=args.topk).to(device)

			optimizer_weight_distributer = optim.Adam(weight_distributer.parameters(),
								lr=args.lr, weight_decay=args.weight_decay)

			for episode in range(episodes):

				# for every epoch, retrain feature_extractor from scratch
				feature_extractor = Feature_Extractor(nfeat=features.shape[1],
							nhid=args.hidden,
							nclass=n_class,
							dropout=args.dropout).to(device)
				optimizer_feature_extractor = optim.Adam(feature_extractor.parameters(),
									lr=args.lr, weight_decay=args.weight_decay)

				pseudo_novel_classes = random.sample(training_classes, len(test_classes))
				pseudo_base_classes = [x for x in training_classes if x not in pseudo_novel_classes]

				sample_weight_train = []
				for i in range(n_class):
					if i in pseudo_novel_classes:
						# also sample rewighting during the imbalanced episodic training
						sample_weight_train.append(40/k_shot)
					else:
						sample_weight_train.append(1)
				sample_weight_train = torch.tensor(sample_weight_train).to(device)


				id_support_pseudo_base, id_query_pseudo_base, pseudo_base_support_class2id, pseudo_base_query_class2id = \
					task_generator(base_support_class2id, pseudo_base_classes, len(pseudo_base_classes), 40, 10)
				id_support_pseudo_novel, id_query_pseudo_novel, pseudo_novel_support_class2id, pseudo_novel_query_class2id = \
					task_generator(base_support_class2id, pseudo_novel_classes, len(pseudo_novel_classes), k_shot, 10)

				id_support_pseudo_base = torch.LongTensor(id_support_pseudo_base).to(device)
				id_query_pseudo_base = torch.LongTensor(id_query_pseudo_base).to(device)
				id_support_pseudo_novel = torch.LongTensor(id_support_pseudo_novel).to(device)
				id_query_pseudo_novel = torch.LongTensor(id_query_pseudo_novel).to(device)

				id_support_pseudo = torch.cat((id_support_pseudo_base, id_support_pseudo_novel), 0)

				meta_train_acc = []
				meta_train_base_acc = []
				meta_train_novel_acc = []
				for _ in range(10):
					# repeat experiments for 10 runs

					acc_train_base, acc_train_novel, acc_train = train(id_support_pseudo, id_query_pseudo_base, id_query_pseudo_novel, feature_extractor, weight_distributer,
								optimizer_feature_extractor, optimizer_weight_distributer, sample_weight_train)
					meta_train_base_acc.append(acc_train_base.cpu())
					meta_train_novel_acc.append(acc_train_novel.cpu())
					meta_train_acc.append(acc_train.cpu())
				if episode > 0 and episode % 5 == 0:
					print("-------Episode {}-------".format(episode))

					# testing
					meta_test_acc = []
					meta_test_base_acc = []
					meta_test_novel_acc = []
					for _ in range(meta_test_num):

						tmp_weight_distributer = deepcopy(weight_distributer).to(device)
						tmp_feature_extractor = Feature_Extractor(nfeat=features.shape[1],
									nhid=args.hidden,
									nclass=n_class,
									dropout=args.dropout).to(device)
						tmp_optimizer_weight_distributer = optim.Adam(tmp_weight_distributer.parameters(),
											lr=args.lr, weight_decay=args.weight_decay)
						tmp_optimizer_feature_extractor = optim.Adam(tmp_feature_extractor.parameters(),
											lr=args.lr, weight_decay=args.weight_decay)

						acc_test_base, acc_test_novel, acc_test = test(torch.cat((id_support_base,id_support_novel),0),
							id_query_base, id_query_novel, tmp_feature_extractor, tmp_weight_distributer,
							tmp_optimizer_feature_extractor, tmp_optimizer_weight_distributer, sample_weight_test)
						meta_test_base_acc.append(acc_test_base.cpu())
						meta_test_novel_acc.append(acc_test_novel.cpu())
						meta_test_acc.append(acc_test.cpu())

					print("Meta-Test_Acc_Base_Mean: {:.3f}, Meta-Test_Acc_Base_Std: {:.3f}".format(np.array(meta_test_base_acc).mean(axis=0),
																				np.array(meta_test_base_acc).std(axis=0)))
					print("Meta-Test_Acc_Novel_Mean: {:.3f}, Meta-Test_Acc_Novel_Std: {:.3f}".format(np.array(meta_test_novel_acc).mean(axis=0),
																				np.array(meta_test_novel_acc).std(axis=0)))
					print("Meta-Test_Acc_Mean: {:.3f}, Meta-Test_Acc_Std: {:.3f}".format(np.array(meta_test_acc).mean(axis=0),
																				np.array(meta_test_acc).std(axis=0)))
					print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
			print()
			print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
