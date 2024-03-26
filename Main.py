from Train_Model import *
from Data_Process import *
import torch
from torch import nn as nn
from scipy.sparse import coo_matrix
import numpy as np
from numpy import diag
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import Adam
from torch.nn import MSELoss
from Metrics import *
import os
import visdom
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

batch_size = 4096
BATCH_SIZE = batch_size
epochs = 100
para = {
    'lr': 0.0001,
    'train': 0.1
}


def test(model, users_list):
    all_precision_20, all_recall_20, all_precision_10, all_recall_10 = [], [], [], []
    all_precision_30,all_precision_40,all_precision_50 = [], [], []
    all_recall_30,all_recall_40,all_recall_50= [], [], []
    all_ndcg_20,all_ndcg_30, all_ndcg_40, all_ndcg_50 = [], [], [], []
    features = data.get_features()
    count = 0
    for j in range(BATCH_SIZE):
        id = users_list[np.random.randint(data.n_users, size=1)[0]]
        item_list = list(set(range(data.n_items)) - set(data.train_items[id]))
        item_features = []
        not_in_features = np.ones(shape=18)
        for i in range(len(item_list)):
            if item_list[i] in features.keys():
                item_features_data = features[str(item_list[i])]
                item_features_data = np.asarray(item_features_data)
                item_features.append(item_features_data)
            else:
                item_features.append(not_in_features)
        features_input = torch.tensor(item_features, dtype=torch.float32).cuda()
        features_input = vector_norm(features_input)

        users = [id for j in range(len(item_list))]
        users = torch.tensor(users).cuda()
        items = torch.tensor(item_list).cuda()
        pred = model.predict(users, items, features_input)

        _, item_key = pred.sort(descending=True)
        item_key = item_key.cpu().int()
        item_top20 = item_key[:20]
        item_top10 = item_key[:10]
        item_top30 = item_key[:30]
        item_top40 = item_key[:40]
        item_top50 = item_key[:50]
        item_list = np.array(item_list)
        pred_top20 = item_list[item_top20]
        pred_top10 = item_list[item_top10]
        pred_top30 = item_list[item_top30]
        pred_top40 = item_list[item_top40]
        pred_top50 = item_list[item_top50]
        actual = data.test_set[id]
        precision_20 = precisionk(actual, pred_top20)
        precision_30 = precisionk(actual, pred_top30)
        precision_40 = precisionk(actual, pred_top40)
        precision_50 = precisionk(actual, pred_top50)
        recall_20 = recallk(actual, pred_top20)
        recall_30 = recallk(actual, pred_top30)
        recall_40 = recallk(actual, pred_top40)
        recall_50 = recallk(actual, pred_top50)
        all_precision_20.append(precision_20)
        all_recall_20.append(recall_20)

        precision_10 = precisionk(actual, pred_top10)
        recall_10 = recallk(actual, pred_top10)
        all_precision_10.append(precision_10)
        all_precision_30.append(precision_30)
        all_precision_40.append(precision_40)
        all_precision_50.append(precision_50)
        all_recall_10.append(recall_10)
        all_recall_30.append(recall_30)
        all_recall_40.append(recall_40)
        all_recall_50.append(recall_50)

        ndcg_20 = ndcgk(actual, pred_top20, 20)
        ndcg_30 = ndcgk(actual, pred_top30, 30)
        ndcg_40 = ndcgk(actual, pred_top40, 40)
        ndcg_50 = ndcgk(actual, pred_top50, 50)
        all_ndcg_20.append(ndcg_20)
        all_ndcg_30.append(ndcg_30)
        all_ndcg_40.append(ndcg_40)
        all_ndcg_50.append(ndcg_50)

    return np.mean(all_precision_20), np.mean(all_recall_20), np.mean(all_precision_10), np.mean(all_recall_10), \
           np.mean(all_recall_30), np.mean(all_recall_40), np.mean(all_recall_50), np.mean(all_precision_30), \
           np.mean(all_precision_40), np.mean(all_precision_50), np.mean(all_ndcg_20), np.mean(all_ndcg_30), \
           np.mean(all_ndcg_40), np.mean(all_ndcg_50)


def vector_norm(data):

    one = torch.ones(size=[1, 1], dtype=torch.float32).cuda()

    cap_data = data + one
    return F.normalize(cap_data, dim=1)


def main():

    model = MIMA(n_users=None, n_items=None, n_features=None, embedding_dim=None, weight_size=None, dropout_list=None, norm_adj=None, batch_size=None,decay=None).cuda()
    optim = Adam(model.parameters(), lr=para['lr'])

    lossfn = model.BPR_loss
    features = data.get_features()

    for i in range(epochs):

        loss_value = 0
        mf_loss_value, reg_loss_value = 0.0, 0.0
        t0 = time()
        for j in range(n_epochs // batch_size + 1):
            users, pos_items, neg_items = data.sample()
            users = torch.tensor(users).cuda()
            pos_items = torch.tensor(pos_items).type(torch.long).cuda()
            neg_items = torch.from_numpy(np.array(neg_items)).type(torch.long).cuda()
            optim.zero_grad()
            mf_loss, reg_loss = lossfn(users, pos_items, neg_items, features)
            loss = mf_loss + reg_loss
            loss.backward()
            optim.step()
            loss_value += loss.item()
            mf_loss_value += mf_loss.item()
            reg_loss_value += reg_loss.item()

        if (i + 1) % 10 != 0:
            str1 = 'epoch: %d loss_value=%.5f' % (i, loss_value)

            continue

        t1 = time()

        user_to_test = list(data.test_set.keys())
        precision_20, recall_20, precision_10, recall_10, recall_30, recall_40, recall_50, precision_30, \
        precision_40, precision_50, ndcg_20, ndcg_30, ndcg_40, ndcg_50 = test(model, user_to_test)
        t2 = time()
        str2 = 'epoch: %d loss_value=%.2f precision_10=%.5f precision_20=%.5f  recall_20=%.5f recall_30=%.5f recall_40=%.5f ' \
               'recall_50=%.5f ndcg_20=%.5f' % (
               i, loss_value, precision_10, precision_20, recall_20, recall_30, recall_40, recall_50, ndcg_20)

        print(str2)

if __name__ == '__main__':
    cur_dir = os.getcwd()
    path = ''
    data = Data(path, batch_size)
    user_nums, item_nums = data.get_num_users_items()
    feature = data.get_features()
    USER_NUM, ITEM_NUM = user_nums, item_nums
    n_epochs = data.get_trainNum()
    plain_adj, norm_adj, mean_adj = data.get_adj_mat()
    main()
