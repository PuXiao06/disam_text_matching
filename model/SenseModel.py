import torch
import torch.nn as nn
from model.modeling_albert import AlbertEmbeddings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SenseModel(nn.Module):
    def __init__(self, config):
        super(SenseModel, self).__init__()

        # self.filter_s1_all = nn.ModuleList([nn.Conv2d(1, dim, (size, dim)) for size in filter_size])
        # self.filter_s2_all = nn.ModuleList([nn.Conv2d(1, dim, (size, dim)) for size in filter_size])
        #
        # self.filter_s1_tag = nn.ModuleList([nn.Conv2d(1, dim, (size, dim)) for size in filter_size])
        # self.filter_s2_tag = nn.ModuleList([nn.Conv2d(1, dim, (size, dim)) for size in filter_size])
        #
        # self.cnn_s = nn.ModuleList([nn.Conv2d(1, dim, (size, dim)) for size in filter_size])
        # self.cnn_tag = nn.ModuleList([nn.Conv2d(1, dim, (size, dim)) for size in filter_size])

        self.embedding = AlbertEmbeddings(config)


    def loss_generate(self, tag_distribution, tag_context_total, tag_words_emb, mask):
        m = nn.BatchNorm1d(4, affine=True)
        tag_distribution = m(tag_distribution.cpu()).to(device)

        mask_boolean = mask.eq(0)  # find the item that should be masked (which should be valued -1e9)
        tag_distr_masked = torch.squeeze(tag_distribution).to(device).masked_fill(mask_boolean, -1e9).to(device)

        # tag_attn = self.l2norm(tag_distr_masked)
        tag_attn = nn.Softmax(dim=-1)(tag_distr_masked)

        tag_context = torch.unsqueeze(tag_attn[:, 0], -1).to(device) * tag_words_emb[:, 0, :].to(device) + \
                      torch.unsqueeze(tag_attn[:, 1], -1).to(device) * tag_words_emb[:, 1, :].to(device) + \
                      torch.unsqueeze(tag_attn[:, 2], -1).to(device) * tag_words_emb[:, 2, :].to(device) + \
                      torch.unsqueeze(tag_attn[:, 3], -1).to(device) * tag_words_emb[:, 3, :].to(device)

        return tag_attn, tag_context

    def context_loss_generate(self, x, tag_1, tag_2, tag_3, tag_4, tag_mask, sense_label_mask, token_type_ids, type_op):
        X_ori_emb = self.embedding(x)
        positive_idx = torch.nonzero(tag_mask == True).T  # 2维度【batch_idx, word_idx】

        S1_Tag_1 = tag_1[positive_s1_idx[0], positive_s1_idx[1]]  # [True_个数, def句子里word的idx]
        S1_Tag_2 = tag_2[positive_s1_idx[0], positive_s1_idx[1]]
        S1_Tag_3 = tag_3[positive_s1_idx[0], positive_s1_idx[1]]
        S1_Tag_4 = tag_4[positive_s1_idx[0], positive_s1_idx[1]]

        x1_tag_mask_filtered = tag_mask[positive_s1_idx[0], positive_s1_idx[1]]

        s1_tag1_emb = self.embed_matrix(S1_Tag_1)  # [True个数，sen, dim]
        s1_tag2_emb = self.embed_matrix(S1_Tag_2)
        s1_tag3_emb = self.embed_matrix(S1_Tag_3)
        s1_tag4_emb = self.embed_matrix(S1_Tag_4)

        S1_temp_emb = X_1_ori_emb[positive_s1_idx[0]]  # [对应True个数, sen, dim]
        # s1_gru = torch.squeeze(self.gru(S1_temp_emb.transpose(0, 1))[1]) #[对应True个数，dim]

        # CNN-based
        tag_word_filtered = tag_word[positive_s1_idx[0], positive_s1_idx[1]]
        tag_word_filtered_emb = self.embed_matrix(tag_word_filtered)

        dropout = nn.Dropout(p=drop_prob)
        pool_cnn_s1 = []
        pool_tag1 = []
        pool_tag2 = []
        pool_tag3 = []
        pool_tag4 = []
        for i, conv in enumerate(self.cnn_s):
            s_2d = nn.MaxPool2d((max_len - filter_size[i] + 1, 1))
            tag_2d = nn.MaxPool2d((sense_len - filter_size[i] + 1, 1))
            # For S1
            temp_s1 = conv(torch.unsqueeze(S1_temp_emb, 1))
            h_s1 = torch.relu(temp_s1)
            h_s1_dropout = dropout(h_s1)
            temp_pooled_s1 = s_2d(h_s1_dropout)
            pool_cnn_s1.append(torch.squeeze(temp_pooled_s1))
            # For S1_tag1
            temp = self.cnn_tag[i](torch.unsqueeze(s1_tag1_emb, 1))
            h_s1_tag1 = torch.relu(temp)
            temp_pooled_s1_tag1 = tag_2d(dropout(h_s1_tag1))
            pool_tag1.append(torch.squeeze(temp_pooled_s1_tag1))
            # For S1_tag2
            h_s1_tag2 = torch.relu(self.cnn_tag[i](torch.unsqueeze(s1_tag2_emb, 1)))
            temp_pooled_s1_tag2 = tag_2d(dropout(h_s1_tag2))
            pool_tag2.append(torch.squeeze(temp_pooled_s1_tag2))
            # For S1_tag3
            h_s1_tag3 = torch.relu(self.cnn_tag[i](torch.unsqueeze(s1_tag3_emb, 1)))
            temp_pooled_s1_tag3 = tag_2d(dropout(h_s1_tag3))
            pool_tag3.append(torch.squeeze(temp_pooled_s1_tag3))
            # For S1_tag4
            h_s1_tag4 = torch.relu(self.cnn_tag[i](torch.unsqueeze(s1_tag4_emb, 1)))
            temp_pooled_s1_tag4 = tag_2d(dropout(h_s1_tag4))
            pool_tag4.append(torch.squeeze(temp_pooled_s1_tag4))

        pool_s1 = torch.cat(pool_cnn_s1, -1)

        s1_tag_total = torch.ones(4, pool_s1.shape[0], dim * 5)
        s1_tag_total[0, :, :] = torch.cat(pool_tag1, -1)
        s1_tag_total[1, :, :] = torch.cat(pool_tag2, -1)
        s1_tag_total[2, :, :] = torch.cat(pool_tag3, -1)
        s1_tag_total[3, :, :] = torch.cat(pool_tag4, -1)
        s1_tag_total = s1_tag_total.transpose(0, 1)
        att_weight_s1 = torch.matmul(torch.unsqueeze(pool_s1.to(device), 1),
                                     s1_tag_total.transpose(1, 2).to(device)).to(device)

        att_weight_s1 = torch.squeeze(att_weight_s1)

        s1_tag_att, tag_context = self.loss_generate(att_weight_s1, s1_tag_total, tag_word_filtered_emb,
                                                     x1_tag_mask_filtered)

        return s1_tag_att, tag_context


    def forward(self, X_1, X1_Tag_1, X1_Tag_2, X1_Tag_3, X1_Tag_4, X_2, X2_Tag_1, X2_Tag_2, X2_Tag_3, X2_Tag_4, tag_mask, tag_label, token_type_ids, drop_prob):

        # X_1_ori_emb = AlbertEmbeddings.word_embeddings(X_1)
        # X_2_ori_emb = AlbertEmbeddings.word_embeddings(X_2)

        s1_att, s1_context = self.context_loss_generate(X_1, X1_Tag_1, X1_Tag_2, X1_Tag_3, X1_Tag_4, tag_mask, tag_label, token_type_ids)
        s2_att, s2_context = self.context_loss_generate(X_2, X2_Tag_1, X2_Tag_2, X2_Tag_3, X2_Tag_4, tag_mask, tag_label, token_type_ids)

        context_X1_emb = X_1_ori_emb
        positive_s1_idx = torch.nonzero(X1_sense_label_mask == True).T
        context_X1_emb[positive_s1_idx[0], positive_s1_idx[1]] = s1_context

        context_X2_emb = X_2_ori_emb
        positive_s2_idx = torch.nonzero(X2_sense_label_mask == True).T
        context_X2_emb[positive_s2_idx[0], positive_s2_idx[1]] = s2_context

        pool_s1 = []
        pool_s2 = []

        pool_s1_tag = []
        pool_s2_tag = []
        dropout = nn.Dropout(p=drop_prob)

        for i, conv1 in enumerate(self.filter_s1_all):
            m_2d = nn.MaxPool2d((max_len - filter_size[i] + 1, 1))
            # For S1
            h_s1 = torch.relu(conv1(torch.unsqueeze(X_1_ori_emb, 1)))
            h_s1_dropout = dropout(h_s1)
            temp_pooled_s1 = m_2d(h_s1_dropout)
            pool_s1.append(torch.squeeze(temp_pooled_s1))
            # For S1 tag
            h_s1_tag = torch.relu(self.filter_s1_tag[i](torch.unsqueeze(context_X1_emb, 1)))
            h_s1_tag_dropout = dropout(h_s1_tag)
            temp_pooled_s1_tag = m_2d(h_s1_tag_dropout)
            pool_s1_tag.append(torch.squeeze(temp_pooled_s1_tag))

            # For S2
            h_s2 = torch.relu(self.filter_s2_all[i](torch.unsqueeze(X_2_ori_emb, 1)))
            h_s2_dropout = dropout(h_s2)
            temp_pooled_s2 = m_2d(h_s2_dropout)
            pool_s2.append(torch.squeeze(temp_pooled_s2))

            # For S2 tag
            h_s2_tag = torch.relu(self.filter_s2_tag[i](torch.unsqueeze(context_X2_emb, 1)))
            h_s2_tag_dropout = dropout(h_s2_tag)
            temp_pooled_s2_tag = m_2d(h_s2_tag_dropout)
            pool_s2_tag.append(torch.squeeze(temp_pooled_s2_tag))


        pool_s1 = torch.cat(pool_s1, -1) #(len(window) + 1) * len(filter)
        pool_s2 = torch.cat(pool_s2, -1)

        pool_1_tag = torch.cat(pool_s1_tag, -1)
        pool_2_tag = torch.cat(pool_s2_tag, -1)

        pool_1 = (pool_s1 + pool_1_tag)/2
        pool_2 = (pool_s2 + pool_2_tag)/2


        polled_sub = torch.abs(torch.subtract(pool_1, pool_2))
        polled_mul = torch.multiply(pool_1, pool_2)

        final_tensor = torch.cat([pool_1, pool_2, pool_1 + pool_2, polled_sub, polled_mul], 1)

        # Add dropout
        dropout_value = dropout(final_tensor)
        # 全连接层
        estimation = self.full_connect(dropout_value)

        return estimation, s1_att, s2_att

    def loss_calculate(estimation_att, label_mask, label_idx):
        label_onehot_index = torch.sparse.torch.eye(4)

        estimate_max_value, estimate_label_idx = torch.max(estimation_att, -1)
        temp = estimation_att
        label_available = torch.masked_select(label_idx, label_mask)

        # 尝试加入onehot label identical loss：
        estimate_onehot = label_onehot_index.index_select(0, estimate_label_idx.cpu()).to(device)
        label_onehot = label_onehot_index.index_select(0, label_available.cpu()).to(device)

        # available_estimate_att_max = torch.max(available_estimate_att, dim=1)[0]
        identical_idx = torch.eq(estimate_label_idx, label_available)

        acc = len(identical_idx[identical_idx == True].view(-1)) / len(identical_idx.view(-1))

        # 只用预测为1的identical case做loss，舍掉预测为0的：
        identical_idx = torch.eq(estimate_label_idx, label_available)
        identical_att_score = torch.masked_select(estimate_max_value, identical_idx)

        identical_value_sum = -(torch.log(identical_att_score).sum() / len(identical_att_score))

        # return identical_value_sum, acc, estimate_label_idx, identical_idx, identical_att_score
        return estimation_att, label_available, acc
