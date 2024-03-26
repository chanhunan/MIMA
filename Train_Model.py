import torch
import torch.nn as nn
import torch.nn.functional as F
from Data_Process import *
import math
import torch
import pandas as pd
from Feature_Interaction_Model import *
from Metapath_Aggregation_Model import *

class MIMA(nn.Module):
    def __init__(self, n_users, n_items, n_features, embedding_dim, weight_size, dropout_list, norm_adj, batch_size,
                 decay):
        super().__init__()
        self.name = 'MIMA'
        self.n_users = n_users
        self.n_items = n_items
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_layers = len(self.weight_size)
        self.dropout_list = nn.ModuleList()
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()
        self.batch_size = batch_size
        self.norm_adj = norm_adj
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float()
        self.norm_adj = self.norm_adj.cuda()
        self.decay = decay
        self.line_user = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.line_item = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.feature_line = torch.nn.Linear(self.embedding_dim, 256)
        self.u_f_embeddings = None
        self.i_f_embeddings = None
        self.feature_f_embedding = None
        self.weight_size = [self.embedding_dim] + self.weight_size

        self.attention_layer = SelfAttention(self.n_features, self.embedding_dim, self.n_features, 0)
        for i in range(self.n_layers):
            self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
            self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
            self.dropout_list.append(nn.Dropout(dropout_list[i]))

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.feature_embedding = nn.Embedding(n_features, embedding_dim)
        self._init_weight_()

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.feature_embedding.weight)

    def forward(self, adj, features_data):

        atten_socore = self.feature_line(self.attention_layer(features_data))

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        ego_embeddings = torch.cat((ego_embeddings, atten_socore), dim=0)

        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
            ego_embeddings = sum_embeddings + bi_embeddings
            ego_embeddings = self.dropout_list[i](ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]
            all_embeddings=Math_Path_Layer(in_dim=len(all_embeddings), heads=None,self_attention=None)

        all_embeddings = torch.cat(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        return u_g_embeddings, i_g_embeddings, atten_socore

    def predict(self, userIDx, itemIDx, features):
        with torch.no_grad():
            if self.u_f_embeddings is not None:
                uEmbd, iEmbd, features_score = self.u_f_embeddings, self.i_f_embeddings, self.feature_embedding.weight
            else:
                uEmbd, iEmbd, features_score = self.forward(self.norm_adj, self.feature_embedding.weight)
                self.u_f_embeddings, self.i_f_embeddings = uEmbd, iEmbd,

            uembd = uEmbd[userIDx]

            iembd = iEmbd[itemIDx]

            features_attention = features_score
            features = torch.mm(features, features_attention)

            uembd = torch.cat((uembd, features), 1)
            iembd = torch.cat((iembd, features), 1)

            prediction = torch.sum(torch.mul(uembd, iembd), dim=1)
        return prediction

    def vector_norm(self, data):

        one = torch.ones(size=[1, 1], dtype=torch.float32).cuda()
        cap_data = data + one
        return F.normalize(cap_data, dim=1)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def BPR_loss(self, users, pos_item, neg_item, features):
        uEmbd, iEmbd, features_attention = self.forward(self.norm_adj, self.feature_embedding.weight)

        self.u_f_embeddings, self.i_f_embeddings = uEmbd, iEmbd

        uembd = uEmbd[users]
        pos_feature = []
        not_in_features = np.ones(shape=18)
        for item in pos_item:
            itemname = str(item)
            if itemname in features.keys():
                pos_feature.append(features[itemname])
            else:
                pos_feature.append(not_in_features)
        neg_feature = []
        for item in neg_item:
            itemname = str(item)
            if itemname in features.keys():
                neg_feature.append(features[itemname])
            else:
                neg_feature.append(not_in_features)

        pos_feature = np.asarray(pos_feature, dtype=float)
        neg_feature = np.asarray(neg_feature, dtype=float)
        pos_features = torch.tensor(pos_feature, dtype=torch.float32).cuda()
        pos_features = self.vector_norm(pos_features)
        pos_features = torch.mm(pos_features, features_attention)
        neg_features = torch.tensor(neg_feature, dtype=torch.float32).cuda()
        neg_features = self.vector_norm(neg_features)
        neg_features = torch.mm(neg_features, features_attention)
        uembd_1 = torch.cat((uembd, pos_features), 1)
        pos_iembd = iEmbd[pos_item]

        pos_iembd = torch.cat((pos_iembd, pos_features), 1)
        neg_iembd = iEmbd[neg_item]
        neg_iembd = torch.cat((neg_iembd, neg_features), 1)

        pos_score = torch.sum(torch.mul(uembd_1, pos_iembd), dim=1)
        neg_score = torch.sum(torch.mul(uembd_1, neg_iembd), dim=1)
        regularizer = torch.sum(uembd ** 2) / 2.0 + torch.sum(pos_iembd ** 2) / 2.0 + torch.sum(neg_iembd ** 2) / 2.0

        maxi = torch.log(torch.sigmoid(pos_score - neg_score))

        mf_loss = torch.mean(maxi) * -1.0

        reg_loss = self.decay * regularizer / self.batch_size

        return mf_loss, reg_loss


