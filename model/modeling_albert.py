"""PyTorch ALBERT model. """
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import os
import sys

import torch
import numpy as np
import torch.nn.functional
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from .modeling_utils import PreTrainedModel, prune_linear_layer
from .configuration_albert import AlbertConfig
from .file_utils import add_start_docstrings
from .self_attention import InnerSelfAttention
import time

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'albert-base': "",
    'albert-large': "",
    'albert-xlarge': "",
    'albert-xxlarge': "",
}
def load_tf_weights_in_albert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
                     "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    if not os.path.exists(tf_path+'/checkpoint'):
        tf_path = tf_path + "/variables/variables"
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    for name, array in zip(names, arrays):
        name = name.replace("attention_1","attention")
        name = name.replace("ffn_1","ffn")
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            elif re.fullmatch(r'[A-Za-z]+_+[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] in ['LayerNorm', 'attention', 'ffn'] and len(l) >= 2:
                l = ["_".join(l[:-1])]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]

        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}
AlbertLayerNorm = torch.nn.LayerNorm

class AlbertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(AlbertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        self.LayerNorm = AlbertLayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cnn_dropout = nn.Dropout(p=0.1)

        self.tag_embeddings = nn.Embedding(config.tag_vocab_size, 768)
        self.cnn_s = nn.ModuleList([nn.Conv2d(1, config.embedding_size, (size, config.embedding_size)) for size in [1, 2, 3, 4, 5]])
        self.cnn_tag = nn.ModuleList([nn.Conv2d(1, config.embedding_size, (size, config.embedding_size)) for size in [1, 2, 3, 4, 5]])
        self.att_norm = nn.BatchNorm1d(4, affine=True)
        self.cnn_projection = nn.Linear(config.embedding_size * 5, config.embedding_size * 5)
        self.cnn_s_projection = nn.Linear(config.embedding_size * 5, config.embedding_size * 5)
        self.projection_embedding = nn.Linear(config.embedding_size, config.embedding_size)

        self.temp_self = InnerSelfAttention(hidden_size=config.embedding_size, p=0.1)


    def calculate_context_fast(self, word_embedding, input_ids, attention_mask, token_type_ids, input_tag_ids, input_def_ids, input_tag_label):

        # tag_mask = input_tag_ids
        # For X1 sentence:
        shape = input_ids.shape
        filter_size = [1, 2, 3, 4, 5]
        position_filter = token_type_ids + attention_mask

        input_ids_X1_position = torch.nonzero(position_filter == 1).T # [2, 670]
        input_ids_X2_position = torch.nonzero(position_filter == 2).T # [2, 670]

        min_position_X1 = torch.min(input_ids_X1_position[1])
        max_position_X1 = torch.max(input_ids_X1_position[1]) + 1
        min_position_X2 = torch.min(input_ids_X2_position[1])
        max_position_X2 = torch.max(input_ids_X2_position[1]) + 1

        # source sentence extraction
        input_ids_X1_emb = word_embedding[:, min_position_X1:max_position_X1]
        input_ids_X2_emb = word_embedding[:, min_position_X2:max_position_X2]

        def_ids_X1 = torch.zeros([shape[0], 128, 4, 20], dtype=torch.int).to(device)
        # tag_mask_X1 = torch.zeros([shape[0], 128, 4], dtype=torch.int8).to(device)
        tag_ids_X1 = torch.zeros([shape[0], 128, 4], dtype=torch.int).to(device)
        tag_label_X1 = torch.ones([shape[0], 128], dtype=torch.long).to(device) * 5
        #########
        def_ids_X2 = torch.zeros([shape[0], 128, 4, 20], dtype=torch.int).to(device)
        # tag_mask_X2 = torch.zeros([shape[0], 128, 4], dtype=torch.int8).to(device)
        tag_ids_X2 = torch.zeros([shape[0], 128, 4], dtype=torch.int).to(device)
        tag_label_X2 = torch.ones([shape[0], 128], dtype=torch.long).to(device) * 5

        # 去掉S2或S1的信息
        tag_label_X1[input_ids_X1_position[0], input_ids_X1_position[1]] = input_tag_label[input_ids_X1_position[0], input_ids_X1_position[1]] # [16, 128]
        tag_ids_X1[input_ids_X1_position[0], input_ids_X1_position[1]] = input_tag_ids[input_ids_X1_position[0], input_ids_X1_position[1]]
        # tag_mask_X1[input_ids_X1_position[0], input_ids_X1_position[1]] = tag_mask[input_ids_X1_position[0], input_ids_X1_position[1]]
        def_ids_X1[input_ids_X1_position[0], input_ids_X1_position[1]] = input_def_ids[input_ids_X1_position[0], input_ids_X1_position[1]]
        ########
        tag_label_X2[input_ids_X2_position[0], input_ids_X2_position[1]] = input_tag_label[input_ids_X2_position[0], input_ids_X2_position[1]] # [16, 128]
        tag_ids_X2[input_ids_X2_position[0], input_ids_X2_position[1]] = input_tag_ids[input_ids_X2_position[0], input_ids_X2_position[1]]
        # tag_mask_X2[input_ids_X2_position[0], input_ids_X2_position[1]] = tag_mask[input_ids_X2_position[0], input_ids_X2_position[1]]
        def_ids_X2[input_ids_X2_position[0], input_ids_X2_position[1]] = input_def_ids[input_ids_X2_position[0], input_ids_X2_position[1]]

        tag_label_X1_filter_position = torch.nonzero(tag_label_X1 != 5).T
        tag_label_X2_filter_position = torch.nonzero(tag_label_X2 != 5).T
        # 提取出有tag的词
        tag_label_X1_have_value = tag_label_X1[tag_label_X1_filter_position[0], tag_label_X1_filter_position[1]] #[92]
        def_ids_X1_have_value = def_ids_X1[tag_label_X1_filter_position[0], tag_label_X1_filter_position[1]]  #[92, 4, 50]
        # tag_mask_X1_have_value = tag_mask_X1[tag_label_X1_filter_position[0], tag_label_X1_filter_position[1]] #[92, 4]
        tag_ids_X1_have_value = tag_ids_X1[tag_label_X1_filter_position[0],  tag_label_X1_filter_position[1]] #[92, 4]
        #######
        tag_label_X2_have_value = tag_label_X2[tag_label_X2_filter_position[0], tag_label_X2_filter_position[1]] #[92]
        def_ids_X2_have_value = def_ids_X2[tag_label_X2_filter_position[0], tag_label_X2_filter_position[1]]  #[92, 4, 50]
        # tag_mask_X2_have_value = tag_mask_X2[tag_label_X2_filter_position[0], tag_label_X2_filter_position[1]] #[92, 4]
        tag_ids_X2_have_value = tag_ids_X2[tag_label_X2_filter_position[0],  tag_label_X2_filter_position[1]] #[92, 4]

        X1_emb_ori = input_ids_X1_emb[tag_label_X1_filter_position[0]]  # [92, 50]
        X1_def_emb_ori = self.word_embeddings(def_ids_X1_have_value) # [92, 4, sen_len, 128]
        X1_tag_emb = self.tag_embeddings(tag_ids_X1_have_value) #[92, 4, 128]

        X1_emb = self.temp_self(X1_emb_ori)[0]
        temp_shape = X1_def_emb_ori.shape
        X1_def_emb_convert = torch.reshape(X1_def_emb_ori, (temp_shape[0]*temp_shape[1], temp_shape[2], temp_shape[3]))
        X1_def_emb = torch.reshape(self.temp_self(X1_def_emb_convert)[0], (temp_shape[0], temp_shape[1], temp_shape[2], temp_shape[3]))

        X2_emb_ori = input_ids_X2_emb[tag_label_X2_filter_position[0]]  # [92, 50]
        X2_def_emb_ori = self.word_embeddings(def_ids_X2_have_value) # [92, 4, sen_len, 128]
        X2_tag_emb = self.tag_embeddings(tag_ids_X2_have_value) #[92, 4, 128]

        X2_emb = self.temp_self(X2_emb_ori)[0]
        # X2_emb = self.temp_self(X1_emb_ori)[0]
        temp_shape = X2_def_emb_ori.shape
        X2_def_emb_convert = torch.reshape(X2_def_emb_ori, (temp_shape[0]*temp_shape[1], temp_shape[2], temp_shape[3]))
        X2_def_emb = torch.reshape(self.temp_self(X2_def_emb_convert)[0], (temp_shape[0], temp_shape[1], temp_shape[2], temp_shape[3]))

        pool_cnn_s1, pool_cnn_s2 = [], []
        pool_x1_tag1, pool_x1_tag2, pool_x1_tag3, pool_x1_tag4 = [], [], [], []
        pool_x2_tag1, pool_x2_tag2, pool_x2_tag3, pool_x2_tag4 = [], [], [], []

        temp_0 = torch.unsqueeze(X1_def_emb[:, 0, :, :], 1)
        temp_1 = torch.unsqueeze(X1_def_emb[:, 1, :, :], 1)
        temp_2 = torch.unsqueeze(X1_def_emb[:, 2, :, :], 1)
        temp_3 = torch.unsqueeze(X1_def_emb[:, 3, :, :], 1)

        temp_0_2 = torch.unsqueeze(X2_def_emb[:, 0, :, :], 1)
        temp_1_2 = torch.unsqueeze(X2_def_emb[:, 1, :, :], 1)
        temp_2_2 = torch.unsqueeze(X2_def_emb[:, 2, :, :], 1)
        temp_3_2 = torch.unsqueeze(X2_def_emb[:, 3, :, :], 1)

        sen_len_1 = max_position_X1 - min_position_X1
        sen_len_2 = max_position_X2 - min_position_X2

        for i, conv in enumerate(self.cnn_s):
            s1_2d = nn.MaxPool2d((sen_len_1 - filter_size[i] + 1, 1))
            s2_2d = nn.MaxPool2d((sen_len_2 - filter_size[i] + 1, 1))
            tag_2d = nn.MaxPool2d((20 - filter_size[i] + 1, 1))
            # For S1/S2
            # print(X1_emb.shape)
            temp_s1 = conv(torch.unsqueeze(X1_emb, 1))
            h_s1 = torch.relu(temp_s1)
            h_s1_dropout = self.cnn_dropout(h_s1)
            temp_pooled_s1 = s1_2d(h_s1_dropout)
            pool_cnn_s1.append(torch.squeeze(temp_pooled_s1))
            #####
            # print(X2_emb.shape)
            temp_s2 = conv(torch.unsqueeze(X2_emb, 1))
            h_s2 = torch.relu(temp_s2)
            h_s2_dropout = self.cnn_dropout(h_s2)
            temp_pooled_s2 = s2_2d(h_s2_dropout)
            pool_cnn_s2.append(torch.squeeze(temp_pooled_s2))

            # For S1_tag1
            h_s1_tag1 = torch.relu(self.cnn_tag[i](temp_0))
            temp_pooled_s1_tag1 = tag_2d(self.cnn_dropout(h_s1_tag1))
            pool_x1_tag1.append(torch.squeeze(temp_pooled_s1_tag1))
            ######
            h_s2_tag1 = torch.relu(self.cnn_tag[i](temp_0_2))
            temp_pooled_s2_tag1 = tag_2d(self.cnn_dropout(h_s2_tag1))
            pool_x2_tag1.append(torch.squeeze(temp_pooled_s2_tag1))

            # For S1_tag2
            h_s1_tag2 = torch.relu(self.cnn_tag[i](temp_1))
            temp_pooled_s1_tag2 = tag_2d(self.cnn_dropout(h_s1_tag2))
            pool_x1_tag2.append(torch.squeeze(temp_pooled_s1_tag2))
            #####
            h_s2_tag2 = torch.relu(self.cnn_tag[i](temp_1_2))
            temp_pooled_s2_tag2 = tag_2d(self.cnn_dropout(h_s2_tag2))
            pool_x2_tag2.append(torch.squeeze(temp_pooled_s2_tag2))

            # For S1_tag3
            h_s1_tag3 = torch.relu(self.cnn_tag[i](temp_2))
            temp_pooled_s1_tag3 = tag_2d(self.cnn_dropout(h_s1_tag3))
            pool_x1_tag3.append(torch.squeeze(temp_pooled_s1_tag3))
            ########
            h_s2_tag3 = torch.relu(self.cnn_tag[i](temp_2_2))
            temp_pooled_s2_tag3 = tag_2d(self.cnn_dropout(h_s2_tag3))
            pool_x2_tag3.append(torch.squeeze(temp_pooled_s2_tag3))

            # For S1_tag4
            h_s1_tag4 = torch.relu(self.cnn_tag[i](temp_3))
            temp_pooled_s1_tag4 = tag_2d(self.cnn_dropout(h_s1_tag4))
            pool_x1_tag4.append(torch.squeeze(temp_pooled_s1_tag4))
            #######
            h_s2_tag4 = torch.relu(self.cnn_tag[i](temp_3_2))
            temp_pooled_s2_tag4 = tag_2d(self.cnn_dropout(h_s2_tag4))
            pool_x2_tag4.append(torch.squeeze(temp_pooled_s2_tag4))

        pool_s1 = self.cnn_s_projection(torch.cat(pool_cnn_s1, -1))
        if len(pool_s1.shape) == 1:
            pool_s1 = torch.unsqueeze(pool_s1, 0)
            print('get it s1')
        # print('s1: ' + '{}'.format(pool_s1.shape))
        pool_s2 = self.cnn_s_projection(torch.cat(pool_cnn_s2, -1))
        if len(pool_s2.shape) == 1:
            pool_s2 = torch.unsqueeze(pool_s2, 0)
            print('get it s2')
        # print('s2: ' + '{}'.format(pool_s2.shape))
        # print(pool_s2.shape)
        embed = X1_emb.shape
        s1_tag_total = torch.ones(4, pool_s1.shape[0], embed[-1] * 5)
        s1_tag_total[0] = self.cnn_projection(torch.cat(pool_x1_tag1, -1))
        s1_tag_total[1] = self.cnn_projection(torch.cat(pool_x1_tag2, -1))
        s1_tag_total[2] = self.cnn_projection(torch.cat(pool_x1_tag3, -1))
        s1_tag_total[3] = self.cnn_projection(torch.cat(pool_x1_tag4, -1))

        s1_tag_total = s1_tag_total.transpose(0, 1).to(device)
        temp_temp = torch.squeeze(torch.matmul(torch.unsqueeze(pool_s1, 1), s1_tag_total.transpose(1, 2)))
        if len(temp_temp.shape) == 1:
            temp_temp = torch.unsqueeze(temp_temp, 0)
            attention_weight_1 = temp_temp
            print('extend att s1')
        else:
            attention_weight_1 = self.att_norm(temp_temp)
        # mask_boolean = tag_mask_X1_have_value.eq(0)  # find the item that should be masked (which should be valued -1e9)
        tag_distr_masked = torch.squeeze(attention_weight_1).masked_fill(tag_ids_X1_have_value.eq(0), -1e9)
        tag_attn_1 = nn.Softmax(dim=-1)(tag_distr_masked)  #【92, 4】

        max_value, max_id = tag_attn_1.max(-1)
        temp_tag_1 = torch.nn.functional.one_hot(max_id, num_classes=4)
        # temp_tag_1 = tag_attn_1.detach()
        # tag_context_1 = torch.unsqueeze(tag_attn_1[:, 0], -1) * X1_tag_emb[:, 0, :] + \
        #               torch.unsqueeze(tag_attn_1[:, 1], -1) * X1_tag_emb[:, 1, :] + \
        #               torch.unsqueeze(tag_attn_1[:, 2], -1) * X1_tag_emb[:, 2, :] + \
        #               torch.unsqueeze(tag_attn_1[:, 3], -1) * X1_tag_emb[:, 3, :]
        tag_context_1 = torch.unsqueeze(temp_tag_1[:, 0], -1) * X1_tag_emb[:, 0, :] + \
                      torch.unsqueeze(temp_tag_1[:, 1], -1) * X1_tag_emb[:, 1, :] + \
                      torch.unsqueeze(temp_tag_1[:, 2], -1) * X1_tag_emb[:, 2, :] + \
                      torch.unsqueeze(temp_tag_1[:, 3], -1) * X1_tag_emb[:, 3, :]
        # a, b = tag_context_1.shape
        # temp = ((torch.rand(a,b) - 0.5) * 2) / 100000000
        # tag_context_1 += temp
        # S2
        s2_tag_total = torch.ones(4, pool_s2.shape[0], embed[-1] * 5)
        s2_tag_total[0] = self.cnn_projection(torch.cat(pool_x2_tag1, -1))
        s2_tag_total[1] = self.cnn_projection(torch.cat(pool_x2_tag2, -1))
        s2_tag_total[2] = self.cnn_projection(torch.cat(pool_x2_tag3, -1))
        s2_tag_total[3] = self.cnn_projection(torch.cat(pool_x2_tag4, -1))

        s2_tag_total = s2_tag_total.transpose(0, 1).to(device)
        temp_temp_2 = torch.squeeze(torch.matmul(torch.unsqueeze(pool_s2, 1), s2_tag_total.transpose(1, 2)))
        if len(temp_temp_2.shape) == 1:
            temp_temp_2 = torch.unsqueeze(temp_temp_2, 0)
            att_weight_s2 = temp_temp_2
            print('extend att s2')
        else:
            att_weight_s2 = self.att_norm(temp_temp_2)
        # mask_boolean = tag_mask_X2_have_value.eq(0)  # find the item that should be masked (which should be valued -1e9)
        tag_distr_masked = torch.squeeze(att_weight_s2).masked_fill(tag_ids_X2_have_value.eq(0), -1e9)
        tag_attn_2 = nn.Softmax(dim=-1)(tag_distr_masked)  #【92, 4】

        max_value, max_id = tag_attn_2.max(-1)
        temp_tag_2 = torch.nn.functional.one_hot(max_id,num_classes=4)
        # print(temp_tag_2)
        # tag_context_2 = torch.unsqueeze(tag_attn_2[:, 0], -1) * X2_tag_emb[:, 0, :] + \
        #               torch.unsqueeze(tag_attn_2[:, 1], -1) * X2_tag_emb[:, 1, :] + \
        #               torch.unsqueeze(tag_attn_2[:, 2], -1) * X2_tag_emb[:, 2, :] + \
        #               torch.unsqueeze(tag_attn_2[:, 3], -1) * X2_tag_emb[:, 3, :]    # [45, 768]

        tag_context_2 = torch.unsqueeze(temp_tag_2[:, 0], -1) * X2_tag_emb[:, 0, :] + \
                      torch.unsqueeze(temp_tag_2[:, 1], -1) * X2_tag_emb[:, 1, :] + \
                      torch.unsqueeze(temp_tag_2[:, 2], -1) * X2_tag_emb[:, 2, :] + \
                      torch.unsqueeze(temp_tag_2[:, 3], -1) * X2_tag_emb[:, 3, :]    # [45, 768]

        total_tag_filter_position = torch.cat([tag_label_X1_filter_position, tag_label_X2_filter_position], -1)   # [2, 45]
        # print(tag_context_1.shape)
        # print(tag_context_2.shape)
        total_tag_context = torch.cat([tag_context_1, tag_context_2], 0)
        total_tag_attn = torch.cat([tag_attn_1, tag_attn_2], 0)
        total_tag_label = torch.cat([tag_label_X1_have_value, tag_label_X2_have_value], 0)

        return total_tag_attn, total_tag_context, total_tag_filter_position, total_tag_label


    # def forward(self, input_ids, token_type_ids=None, position_ids=None):
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, \
                    input_tag_ids=None, input_def_ids=None, input_tag_label=None, input_tag_mask=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)

        tag_attn, tag_context, tag_filter_position, tag_label = self.calculate_context_fast(words_embeddings, input_ids, attention_mask, token_type_ids, \
                                                                        input_tag_ids, input_def_ids, input_tag_label)

        # tag_context_project = self.projection_embedding(tag_context)
        # shape = words_embeddings.shape
        # context_embeddings = torch.zeros(shape[0], shape[1], shape[2]).to(device)
        # context_embeddings[tag_filter_position[0], tag_filter_position[1]] = tag_context_project
        # # context_output_project = self.projection_embedding(tag_context_project)
        # words_embeddings += context_embeddings


        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)


        return embeddings, tag_attn, tag_context, tag_filter_position, tag_label

class AlbertSelfAttention(nn.Module):
    def __init__(self, config):
        super(AlbertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

class AlbertSelfOutput(nn.Module):
    def __init__(self, config):
        super(AlbertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class AlbertAttention(nn.Module):
    def __init__(self, config):
        super(AlbertAttention, self).__init__()
        self.self = AlbertSelfAttention(config)
        self.output = AlbertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,self_outputs)
        return outputs

class AlbertOutput(nn.Module):
    def __init__(self, config):
        super(AlbertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class AlbertIntermediate(nn.Module):
    def __init__(self, config):
        super(AlbertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = AlbertOutput(config)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        intermediate_output = self.dense(hidden_states)
        intermediate_output = self.intermediate_act_fn(intermediate_output)
        output = self.output(intermediate_output)
        return output

class AlbertFFN(nn.Module):
    def __init__(self, config):
        super(AlbertFFN, self).__init__()
        self.intermediate = AlbertIntermediate(config)

    def forward(self, attention_output):
        output = self.intermediate(attention_output)
        return output

class AlbertLayer(nn.Module):
    def __init__(self, config):
        super(AlbertLayer, self).__init__()
        self.attention = AlbertAttention(config)
        self.ffn = AlbertFFN(config)
        self.LayerNorm = AlbertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm_1 = AlbertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self.LayerNorm(attention_outputs[0] + hidden_states)
        ffn_output = self.ffn(attention_output)
        ffn_output = self.LayerNorm_1(ffn_output+attention_output)
        outputs = (ffn_output,) + attention_outputs[1:] # add attentions if we output them
        return outputs

class AlbertGroup(nn.Module):
    def __init__(self, config):
        super(AlbertGroup, self).__init__()
        self.inner_group_num = config.inner_group_num
        self.inner_group = nn.ModuleList([AlbertLayer(config) for _ in range(config.inner_group_num)])

    def forward(self, hidden_states, attention_mask, head_mask):
        layer_attentions = ()
        layer_hidden_states = ()
        for inner_group_idx in range(self.inner_group_num):
            layer_module = self.inner_group[inner_group_idx]
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask)
            hidden_states = layer_outputs[0]
            layer_attentions = layer_attentions + (layer_outputs[1],)
            layer_hidden_states = layer_hidden_states + (hidden_states,)
        return (layer_hidden_states, layer_attentions)

class AlbertTransformer(nn.Module):
    def __init__(self, config):
        super(AlbertTransformer, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_groups = config.num_hidden_groups
        self.group = nn.ModuleList([AlbertGroup(config) for _ in range(config.num_hidden_groups)])

    def forward(self, hidden_states, attention_mask, head_mask):
        all_hidden_states = ()
        all_attentions = ()
        for layer_idx in range(self.num_hidden_layers):
            if self.output_hidden_states and layer_idx == 0:
                all_hidden_states = all_hidden_states + (hidden_states,)
            group_idx = int(layer_idx / self.num_hidden_layers * self.num_hidden_groups)
            layer_module = self.group[group_idx]
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[layer_idx])
            hidden_states = layer_outputs[0][-1]
            if self.output_attentions:
                all_attentions = all_attentions + layer_outputs[1]
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + layer_outputs[0]
        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class AlbertEncoder(nn.Module):
    def __init__(self, config):
        super(AlbertEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.embedding_size = config.embedding_size
        self.embedding_hidden_mapping_in = nn.Linear(self.embedding_size, self.hidden_size)
        self.transformer = AlbertTransformer(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        if self.embedding_size != self.hidden_size:
            prev_output = self.embedding_hidden_mapping_in(hidden_states)
        else:
            prev_output = hidden_states
        outputs = self.transformer(prev_output, attention_mask, head_mask)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class AlbertPooler(nn.Module):
    def __init__(self, config):
        super(AlbertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class AlbertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(AlbertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = AlbertLayerNorm(config.embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class AlbertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(AlbertLMPredictionHead, self).__init__()
        self.transform = AlbertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.embedding_size,config.vocab_size,bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class AlbertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(AlbertOnlyMLMHead, self).__init__()
        self.predictions = AlbertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class AlbertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(AlbertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

class AlbertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(AlbertPreTrainingHeads, self).__init__()
        self.predictions = AlbertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class AlbertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = AlbertConfig
    pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_albert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, AlbertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


ALBERT_START_DOCSTRING = r"""    The ALBERT model was proposed in
    `ALBERT: A Lite BERT for Self-supervised Learning of Language Representations`_
    by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut. 
    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.
    .. _`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
        https://arxiv.org/abs/1909.11942
    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module
    Parameters:
        config (:class:`~transformers.ALbertConfig`): Model configuration class with all the parameters of the model. 
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ALBERT_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, ALBERT input sequence should be formatted with [CLS] and [SEP] tokens as follows:
            (a) For sequence pairs:
                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``
                ``token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1``
            (b) For single sequences:
                ``tokens:         [CLS] the dog is hairy . [SEP]``
                ``token_type_ids:   0   0   0   0  0     0   0``
            ALBert is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.
            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""



@add_start_docstrings("The bare Albert Model transformer outputting raw hidden-states without any specific head on top.",
                      ALBERT_START_DOCSTRING, ALBERT_INPUTS_DOCSTRING)
class AlbertModel(AlbertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    """

    def __init__(self, config):
        super(AlbertModel, self).__init__(config)

        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertEncoder(config)
        self.pooler = AlbertPooler(config)
        # self.tag_embeddings = nn.Embedding(config.tag_size, 768)
        self.projection_embedding = nn.Linear(768, 768)
        self.fusion_projection = nn.Linear(768, 768)
        self.init_weights()


    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, \
                input_tag_ids=None, input_def_ids=None, input_tag_label=None, input_tag_mask=None):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # embedding_output, \
        # tag_attn_1, tag_label_1, \
        # tag_attn_2, tag_label_2, \
        # tag_label_X1_filter_position, tag_label_X2_filter_position, \
        # tag_context_1, tag_context_2 = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, \
        #                                      input_tag_ids=input_tag_ids, input_def_ids=input_def_ids, input_tag_label=input_tag_label, input_tag_mask=input_tag_mask)

        embedding_output, \
        tag_attn, tag_context, tag_filter_position, tag_label = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, \
                                             input_tag_ids=input_tag_ids, input_def_ids=input_def_ids, input_tag_label=input_tag_label, input_tag_mask=input_tag_mask)


        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]    # [batch, sen, 768]

        # #add tag_embedding
        context_output = encoder_outputs[0]
        context_output[tag_filter_position[0], tag_filter_position[1]] = tag_context
        context_output_project = self.projection_embedding(context_output)
        sequence_output = sequence_output + context_output_project
        # sequence_output = self.fusion_projection(sequence_output)
        # #end
        # tag_context_project = self.projection_embedding(tag_context)
        # shape = sequence_output.shape
        # context_embeddings = encoder_outputs[0]
        # # context_embeddings = torch.zeros(shape[0], shape[1], shape[2]).to(device)
        # context_embeddings[tag_filter_position[0], tag_filter_position[1]] = tag_context
        # context_output_project = self.projection_embedding(context_embeddings)
        # sequence_output += context_output_project

        # add tag embedding
        # positive_tag_idx = torch.nonzero(tag_input_ids != 0).T
        # tag_embedding_output = self.tag_embeddings(tag_input_ids)
        # tag_embedding_filtered = tag_embedding_output[positive_tag_idx[0], positive_tag_idx[1]]
        # temp_sequence_output = sequence_output
        # temp_sequence_output[positive_tag_idx[0], positive_tag_idx[1]] = tag_embedding_filtered
        # temp_sequence_output = self.projection_embedding(temp_sequence_output)
        # sequence_output = sequence_output + temp_sequence_output
        # tag embedding END
        # print(context_output_project.shape)
        torch.save(encoder_outputs[0], './' + str(time.time()) + '.pth')
        pooled_output = self.pooler(sequence_output)  #[batch, 768]

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:]  # add hidden_states and attentions if they are here  # [16, sen, 768], [16, 768]
        return outputs, tag_attn, tag_label  # sequence_output, pooled_output, (hidden_states), (attentions)

@add_start_docstrings("""Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
                      ALBERT_START_DOCSTRING, ALBERT_INPUTS_DOCSTRING)
class AlbertForSequenceClassification(AlbertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config):
        super(AlbertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = AlbertModel(config)
        self.dropout = nn.Dropout(0.1 if config.hidden_dropout_prob == 0 else config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, \
                input_tag_ids=None, input_def_ids=None, input_tag_label=None, input_tag_mask=None):

        # outputs = self.bert(input_ids,
        #                     attention_mask=attention_mask,
        #                     token_type_ids=token_type_ids,
        #                     position_ids=position_ids,
        #                     head_mask=head_mask,
        #                     tag_input_ids=tag_input_ids)

        outputs, \
        tag_attn, tag_label = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            input_tag_ids=input_tag_ids, input_def_ids=input_def_ids, input_tag_label=input_tag_label, input_tag_mask=input_tag_mask)

        # print(tag_attn)
        # 计算tag预测的acc

        identical_idx = torch.eq(torch.max(tag_attn, -1)[1], tag_label)

        acc = len(identical_idx[identical_idx == True].view(-1)) / len(identical_idx.view(-1))
        # END

        # print('tag acc: ' + '{}'.format(acc))

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output+0.1)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss_tag = loss_fct(tag_attn, tag_label)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) + loss_tag

                print('tag acc:{:,.5f}, loss_tag:{:,.5f}'.format(acc, loss_tag))

            outputs = (loss,) + outputs



        return outputs, loss_tag, acc  # (loss), logits, (hidden_states), (attentions)

