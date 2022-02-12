'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import tensorflow as tf
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from utility.helper import *
from utility.batch_test import *
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt

class NGCF(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = 'ngcf'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.intera_func = args.intera_func   #interaction function between user and item embeddings (for modelling user preference)
        self.layers_aggr = args.layers_aggr #layers aggregator

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 100

        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size) #converted string to list?
        #Calling eval() method will execute all preceding operations that produce the inputs needed for the operation that produces this tensor.
        self.n_layers = len(self.weight_size)

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0] # L2 regularization strength

        self.verbose = args.verbose

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''

        #A placeholder is simply a variable that we will assign data to at a later date.
        # It allows us to create our operations and build our computation graph, without needing the data.
        # In TensorFlow terminology, we then feed data into the graph through these placeholders.

        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        # dropout: node dropout (adopted on the ego-networks);
        #          ... since the usage of node dropout have higher computational cost,
        #          ... please use the 'node_dropout_flag' to indicate whether use such technique.
        #          message dropout (adopted on the convolution operations).
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
        """
        #alg_type It specifies the type of graph convolutional layer.
        if self.alg_type in ['ngcf']:
            self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['gcmc']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()

        ################# liyuli 5_11_2021
        elif self.alg_type in ['gat']:
            self.ua_embeddings, self.ia_embeddings = self._create_gat_embed()

        elif self.alg_type in ['variant1']:
            self.ua_embeddings, self.ia_embeddings = self._create_variant1_embed()

        elif self.alg_type in ['variant2']:
            self.ua_embeddings, self.ia_embeddings = self._create_variant2_embed()

        elif self.alg_type in ['variant3']:
            self.ua_embeddings, self.ia_embeddings = self._create_variant3_embed()

        elif self.alg_type in ['variant4']:
            self.ua_embeddings, self.ia_embeddings = self._create_variant4_embed()

        # elif self.alg_type in ['sgat']:  #simplified gat
        #     self.ua_embeddings, self.ia_embeddings = self._create_sgat_embed()

        elif self.alg_type in ['mf']:  #matrix factorization
            self.ua_embeddings, self.ia_embeddings = self._create_mf_embed()
        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        """
        *********************************************************
        Inference for the testing phase.
        """
        #self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)

        # select from different interaction function (inner_product OR neural_network OR fusion) ####yuli_li 13_11_2021
        if self.intera_func == 'inner_prod':
            self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)

        elif self.intera_func == 'neural':
            l1_output_list=[]
            for k in range(sum(self.weight_size_list)):
                score_p1 = tf.matmul(self.u_g_embeddings, self.weights['w%d_mlp_l1_p1'% k])
                score_p2 = tf.matmul(self.pos_i_g_embeddings, self.weights['w%d_mlp_l1_p2'% k])
                l1_output_col = score_p1 + tf.transpose(score_p2, [1, 0])
                l1_output_list.append(tf.reshape(l1_output_col,[-1,1]))

            l1_output = tf.concat(l1_output_list,axis=1) + self.weights['b_mlp_l1'] #layer1 # (8*N,4*N)
            tem = tf.nn.leaky_relu(l1_output)  # nonlinear layer

            l2_output = tf.matmul(tem, self.weights['W_mlp_l2']) + self.weights['b_mlp_l2']  # (4*N,2*N)
            tem = tf.nn.leaky_relu(l2_output)  # nonlinear layer

            l3_output = tf.matmul(tem, self.weights['W_mlp_l3']) + self.weights['b_mlp_l3']  # (2*N,N)
            tem = tf.nn.leaky_relu(l3_output)  # nonlinear layer

            l4_output = tf.matmul(tem, self.weights['W_mlp_l4']) + self.weights['b_mlp_l4']  # (N,1)
            self.batch_ratings = l4_output

            #self.batch_ratings = tf.reshape(scores,[self.users.shape[0],-1])

        #fusion of inner_product and neural ineraction(MLP)
        #refer as Fusion of GMF and MLP in NCF paper
        elif self.intera_func == 'fusion':
            #part1 : 3 layer mlp
            l1_output_list=[]
            for k in range(sum(self.weight_size_list)):
                score_p1 = tf.matmul(self.u_g_embeddings, self.weights['w%d_mlp_l1_p1'% k])
                score_p2 = tf.matmul(self.pos_i_g_embeddings, self.weights['w%d_mlp_l1_p2'% k])
                l1_output_col = score_p1 + tf.transpose(score_p2, [1, 0])
                l1_output_list.append(tf.reshape(l1_output_col,[-1,1]))

            l1_output = tf.concat(l1_output_list,axis=1) + self.weights['b_mlp_l1'] #layer1 # (8*N,4*N)
            tem = tf.nn.leaky_relu(l1_output)  # nonlinear layer

            l2_output = tf.matmul(tem, self.weights['W_mlp_l2']) + self.weights['b_mlp_l2']  # (4*N,2*N)
            tem = tf.nn.leaky_relu(l2_output)  # nonlinear layer

            l3_output = tf.matmul(tem, self.weights['W_mlp_l3']) + self.weights['b_mlp_l3']  # (2*N,N)
            mlp_part = tf.nn.leaky_relu(l3_output)  # nonlinear layer


            #part2: general inner product part
            expanded_u_embeddings = tf.expand_dims(self.u_g_embeddings,axis=1) #add a axis for broadcasting., size (u,1,4*64)
            tem = tf.multiply(expanded_u_embeddings,self.pos_i_g_embeddings) # (u,1,4*64)+ (i,4*64) = size (u,i,4*64)

            #attempt1
            #list_tem = tf.unstack(tem,axis=0)  # list of (i,4*64)
            ##this way doesn't work, ValueError: Cannot infer num from shape (?, ?, 256)
            #element_w_prod_part= tf.concat(list_tem,axis=0)  #size (u*i,4*64)


            #attempt2
            element_w_prod_part= tf.reshape(tem, [-1,tem.shape.as_list()[-1]])
            #element_w_prod_part = tf.reshape(tem, [-1, sum(self.weight_size_list)])

            concatenation = tf.concat([0.4*mlp_part,0.6*element_w_prod_part],axis=1) #weighted concatenation #size(u*i, 5*64)

            self.batch_ratings = tf.matmul(concatenation, self.weights['W_fusion']) #size (u*i,1)



        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss + self.reg_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss) #adam optimizer
        #self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)  # SGD optimizer

    #initialize all trainable parameters
    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()
        initializer2 = tf.random_uniform_initializer(maxval=0.1, minval=-0.1)
        initializer3 = tf.random_uniform_initializer(maxval=1, minval=0.01)

        #tf.Variable objects store mutable tf.Tensor-like values accessed during training to make automatic
        # differentiation easier. Operations on Variables are "watched" by the
        #The collections of variables can be encapsulated into layers or models, along with methods that operate on them

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size
        #list of embedding sizes (from the initial embeddings to the outputs of the last layer)


        for k in range(self.n_layers):
            all_weights['W_gc_%d' %k] = tf.Variable(
                initializer2([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' %k] = tf.Variable(
                initializer2([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer2([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer2([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer2([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer2([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)

            #add specifically for variant4 (concatenation updating rule)
            all_weights['W_squeeze_%d' % k] = tf.Variable(
                initializer2([2*self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_squeeze_%d' % k)
            all_weights['b_squeeze_%d' % k] = tf.Variable(
                initializer2([1, self.weight_size_list[k + 1]]), name='b_squeeze_%d' % k)

            #  specifically for GAT
            all_weights['a_1_%d' % k] = tf.Variable(
            initializer2([self.weight_size_list[k+1], 1]), name = 'a_1_%d' % k)
            all_weights['a_2_%d' % k] = tf.Variable(
            initializer2([self.weight_size_list[k+1], 1]), name = 'a_2_%d' % k)

        # For nueral network (mlp) interaction function

        #layer 1: 4*64 + 4*64—> 4*64
        for k in range(sum(self.weight_size_list)):
            all_weights['w%d_mlp_l1_p1'% k] = tf.Variable(
                initializer2([sum(self.weight_size_list), 1]), name='w%d_mlp_l1_p1'% k) #(64*4,1)
            all_weights['w%d_mlp_l1_p2'% k] = tf.Variable(
                initializer2([sum(self.weight_size_list), 1]), name='w%d_mlp_l1_p2'% k)

        all_weights['b_mlp_l1'] = tf.Variable(
                initializer2([1, sum(self.weight_size_list)]), name='b_mlp_l1')

        #layer 2: 4*64—> 2*64
        all_weights['W_mlp_l2'] = tf.Variable(
            initializer2([sum(self.weight_size_list),sum(self.weight_size_list[-2:])]), name='W_mlp_l2') #(64*4, 2*64)
        all_weights['b_mlp_l2'] = tf.Variable(initializer([1,sum(self.weight_size_list[-2:])]), name='b_mlp_l2')

        # layer 3: 2*64—> 64
        all_weights['W_mlp_l3'] = tf.Variable(
            initializer2([sum(self.weight_size_list[-2:]),self.weight_size_list[-1]]), name='W_mlp_l3')  # (2*64,64)
        all_weights['b_mlp_l3'] = tf.Variable(initializer([1,self.weight_size_list[-1]]), name='b_mlp_l3')

        # layer 4: 64—> 1
        all_weights['W_mlp_l4'] = tf.Variable(
            initializer2([self.weight_size_list[-1], 1]), name='W_mlp_l4')  # (64,1)
        all_weights['b_mlp_l4'] = tf.Variable(initializer([1]), name='b_mlp_l4')

        #specifically for fusion of MLP and inner product
        all_weights['W_fusion'] = tf.Variable(
            initializer2([sum(self.weight_size_list)+self.weight_size_list[-1], 1]), name='W_fusion')  # (4*64+64,1)


        ##########for weighted average of embedding layers
        all_weights['weights'] = tf.Variable(initializer3([self.n_layers+1]), name='layer_weights',constraint=lambda t: tf.clip_by_value(t, 0.01, 1))

        return all_weights

    #split huge adj_matrix into chunks for efficiency (without node dropout)
    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            #A_fold_hat.append(X[start:end])
        return A_fold_hat

    #split huge adj_matrix into chunks for efficiency (with specified node dropout rate)
    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            #temp = self.X[start:end]
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat



##proposal 1: GNN layers with different aggregation rules
    #Component 1 of ngcf
    def _create_variant1_embed(self):
        # Generate a set of adjacency sub-matrix.
        # split the A_hat into fold parts for the memory efficiency
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
                # split the A_hat into fold parts for the memory efficiency

            # # sum messages of neighbors.
            # side_embeddings = tf.concat(temp_embed, 0) #concatenate list of tensor objects along Dimension 0
            # # transformed sum messages of neighbors.
            # sum_embeddings = tf.nn.leaky_relu(
            #     tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            #
            # # bi messages of neighbors.
            # bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # # transformed bi messages of neighbors.
            # bi_embeddings = tf.nn.leaky_relu(
            #     tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])
            #
            # # non-linear activation.
            # ego_embeddings = sum_embeddings + bi_embeddings

            ############# my version

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)  # concatenate list of tensor objects along Dimension 0

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(side_embeddings, ego_embeddings)

            # transformed bi messages of neighbors.
            bi_embeddings = tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k]

            #  add up and then apply ReLU(non-linear activation)
            ego_embeddings = tf.nn.leaky_relu(bi_embeddings)

            #############
            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            # norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [ego_embeddings]

        if self.layers_aggr == 'concat':
            all_embeddings = tf.concat(all_embeddings,
                                       1)  # concatenate embeddings from different layers along Dimension 1
        elif self.layers_aggr == 'avg':
            all_embeddings = sum(all_embeddings) / len(all_embeddings)
        elif self.layers_aggr == 'weighted_avg':
            all_embeddings = tf.stack(all_embeddings, axis=2)  # list of L (N,K)—>(N,K,L)

            all_embeddings = tf.math.multiply(all_embeddings, self.weights['weights'])  # (N,K,L)*(L,)—>(N,K,L)
            # broadcasting weights when conducting element-wise multiplication
            all_embeddings = tf.reduce_sum(all_embeddings, axis=2) / tf.reduce_sum(
                self.weights['weights'])  # weighted average

        elif self.layers_aggr == 'weighted_concat':
            all_embeddings = tf.stack(all_embeddings, axis=2)  # list of L (N,K)—>(N,K,L)
            # normlized_weights = self.weights['weights']/tf.reduce_sum(self.weights['weights'])  #报错,输出结果不是(L,) 了?

            all_embeddings = tf.math.multiply(all_embeddings,
                                              tf.math.sqrt(self.weights['weights']))  # (N,K,L)*(L,)—>(N,K,L)
            # broadcasting weights when conducting element-wise multiplication

            all_embeddings = all_embeddings / tf.math.sqrt(tf.reduce_sum(self.weights['weights']))
            all_embeddings = tf.unstack(all_embeddings, axis=2)  # (N,K,L) —> list of L (N,K)
            all_embeddings = tf.concat(all_embeddings,
                                       1)  # concatenate embeddings from different layers along Dimension 1
        elif self.layers_aggr == 'last':
            all_embeddings = all_embeddings[-1]

        all_embeddings = tf.concat(all_embeddings, 1)  # concatenate embeddings from different layers along Dimension 1
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    #Component 2 of ngcf
    def _create_variant2_embed(self):

        # Generate a set of adjacency sub-matrix.
        # split the A_hat into fold parts for the memory efficiency
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
                # split the A_hat into fold parts for the memory efficiency

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)  # concatenate list of tensor objects along Dimension 0


            # transformed sum messages of neighbors.
            sum_embeddings = tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k]



            #  add up and then apply ReLU(non-linear activation)
            ego_embeddings = tf.nn.leaky_relu(sum_embeddings)

            #############
            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            # norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [ego_embeddings]

        if self.layers_aggr == 'concat':
            all_embeddings = tf.concat(all_embeddings,
                                       1)  # concatenate embeddings from different layers along Dimension 1
        elif self.layers_aggr == 'avg':
            all_embeddings = sum(all_embeddings) / len(all_embeddings)
        elif self.layers_aggr == 'weighted_avg':
            all_embeddings = tf.stack(all_embeddings, axis=2)  # list of L (N,K)—>(N,K,L)

            all_embeddings = tf.math.multiply(all_embeddings, self.weights['weights'])  # (N,K,L)*(L,)—>(N,K,L)
            # broadcasting weights when conducting element-wise multiplication
            all_embeddings = tf.reduce_sum(all_embeddings, axis=2) / tf.reduce_sum(
                self.weights['weights'])  # weighted average

        elif self.layers_aggr == 'weighted_concat':
            all_embeddings = tf.stack(all_embeddings, axis=2)  # list of L (N,K)—>(N,K,L)
            # normlized_weights = self.weights['weights']/tf.reduce_sum(self.weights['weights'])  #报错,输出结果不是(L,) 了?

            all_embeddings = tf.math.multiply(all_embeddings,
                                              tf.math.sqrt(self.weights['weights']))  # (N,K,L)*(L,)—>(N,K,L)
            # broadcasting weights when conducting element-wise multiplication

            all_embeddings = all_embeddings / tf.math.sqrt(tf.reduce_sum(self.weights['weights']))
            all_embeddings = tf.unstack(all_embeddings, axis=2)  # (N,K,L) —> list of L (N,K)
            all_embeddings = tf.concat(all_embeddings,
                                       1)  # concatenate embeddings from different layers along Dimension 1
        elif self.layers_aggr == 'last':
            all_embeddings = all_embeddings[-1]

        all_embeddings = tf.concat(all_embeddings, 1)  # concatenate embeddings from different layers along Dimension 1
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    # W_1(e_i ||e_u )
    def _create_variant3_embed(self):
        # Generate a set of adjacency sub-matrix.
        # split the A_hat into fold parts for the memory efficiency
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            item_part = tf.concat(temp_embed, 0)
            # item_part = tf.matmul(bi_norm_adj, ego_embeddings) #use above for efficiency

            bi_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj)  # bidirectionally normalized adjacency matrix (without self_loop)
            user_part = tf.math.multiply( ego_embeddings, tf.sparse.reduce_sum(bi_norm_adj, axis=1,keepdims=True)) #elementwise multiplication, with broadcast

            sum_part_I = tf.matmul(user_part, self.weights['W_gc_%d' % k])
            sum_part_II = tf.matmul(item_part, self.weights['W_bi_%d' % k]) + self.weights['b_gc_%d' % k]
            sum_embeddings = sum_part_I+ sum_part_II


            #  add up and then apply ReLU(non-linear activation)
            ego_embeddings = tf.nn.leaky_relu(sum_embeddings)

            #############
            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            # norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [ego_embeddings]

        if self.layers_aggr == 'concat':
            all_embeddings = tf.concat(all_embeddings,
                                       1)  # concatenate embeddings from different layers along Dimension 1
        elif self.layers_aggr == 'avg':
            all_embeddings = sum(all_embeddings) / len(all_embeddings)
        elif self.layers_aggr == 'weighted_avg':
            all_embeddings = tf.stack(all_embeddings, axis=2)  # list of L (N,K)—>(N,K,L)

            all_embeddings = tf.math.multiply(all_embeddings, self.weights['weights'])  # (N,K,L)*(L,)—>(N,K,L)
            # broadcasting weights when conducting element-wise multiplication
            all_embeddings = tf.reduce_sum(all_embeddings, axis=2) / tf.reduce_sum(
                self.weights['weights'])  # weighted average

        elif self.layers_aggr == 'weighted_concat':
            all_embeddings = tf.stack(all_embeddings, axis=2)  # list of L (N,K)—>(N,K,L)
            # normlized_weights = self.weights['weights']/tf.reduce_sum(self.weights['weights'])  #报错,输出结果不是(L,) 了?

            all_embeddings = tf.math.multiply(all_embeddings,
                                              tf.math.sqrt(self.weights['weights']))  # (N,K,L)*(L,)—>(N,K,L)
            # broadcasting weights when conducting element-wise multiplication

            all_embeddings = all_embeddings / tf.math.sqrt(tf.reduce_sum(self.weights['weights']))
            all_embeddings = tf.unstack(all_embeddings, axis=2)  # (N,K,L) —> list of L (N,K)
            all_embeddings = tf.concat(all_embeddings,
                                       1)  # concatenate embeddings from different layers along Dimension 1
        elif self.layers_aggr == 'last':
            all_embeddings = all_embeddings[-1]

        all_embeddings = tf.concat(all_embeddings, 1)  # concatenate embeddings from different layers along Dimension 1
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    #modified GAT layers
    #the sparse implementation is inspired by the official implementation of GAT
    def _create_gat_embed(self):

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        ## the original GAT design
        # for k in range(self.n_layers):
        #
        #     #project node embeddings into another hidden space
        #     h_embeddings = tf.matmul(ego_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k]
        #
        #     #define two trainable parameter vectors in each GAT layer , a_1, a_2
        #     attention_part_I = tf.matmul(h_embeddings, self.weights['a_1_%d' % k])
        #     attention_part_II = tf.matmul(h_embeddings, self.weights['a_2_%d' % k])
        #
        #     # just retain attentions to first-order neighboring nodes and self by masking attentions to all other nodes
        #
        #     # Failure 1: 构建full-attention matrix then mask it
        #     #utilizing broadcasting (N+M, 1) + (1,N+M ) to generate attention_matrix (N+M, N+M)
        #     #这个地方，attention_martix 会不会太大了? 如何转成sparse implementation??
        #     #attention_matrix = tf.sparse.from_dense(attention_part_I + attention_part_II.T)
        #
        #     # connectivity_mask = adj_self_loop - adj_self_loop.power(-1)   #之后放到外面# adj_matrix selected as 'plain' and then add self-connection (eye matrix)
        #     # attention_matrix = attention_matrix + connectivity_mask.todense()
        #     # connectivity mask will put -inf on all locations where there are no edges, after applying the softmax
        #     # this will result in attention scores being computed only for existing edges
        #     # ValueError: Cannot create a tensor proto whose content is larger than 2GB.
        #
        #
        #     # Failure2: through array/tensor selection to construct sparse attention matrix
        #     # adj_self_loop = self.norm_adj+ sp.eye(self.norm_adj.shape[0])
        #     # adj_self_loop = adj_self_loop.tocoo() #convert to A sparse matrix in COOrdinate format.
        #     # data = []
        #     # for row,col in zip(adj_self_loop.row,adj_self_loop.col):
        #     #     data.append(attention_part_I[row]+attention_part_II[col])
        #     #
        #     # adj_self_loop.data = data
        #     # adj_self_loop = adj_self_loop.tocsr() #to Compressed Sparse Row matrix
        #     # attention_matrix = adj_self_loop
        #
        #
        #     # Attempt3, succeed
        #     adj_self_loop = self.norm_adj + sp.eye(self.norm_adj.shape[0])
        #     #convert sp_mat to sp_tensor
        #     adj_self_loop = self._convert_sp_mat_to_sp_tensor(adj_self_loop)
        #
        #     # Component-wise multiplies a SparseTensor by a dense Tensor. (broadcasting),broadcasts the dense side to the sparse side, but not the other direction.
        #     attention_part_I = adj_self_loop.__mul__(attention_part_I)
        #     attention_part_II = adj_self_loop.__mul__(tf.transpose(attention_part_II, [1, 0]))
        #     # result should also be sparse ? because adj_self_loop is sparse
        #
        #
        #     attention_part_I = adj_self_loop.__mul__(attention_part_I)
        #     attention_part_II = adj_self_loop.__mul__(tf.transpose(attention_part_II, [1, 0]))
        #
        #     logits  = tf.sparse_add(attention_part_I, attention_part_II) #At least one operand must be sparse.
        #
        #
        #     # apply ReLU(non-linear activation) , tf.nn.leaky_relu() accepts – A `Tensor`
        #     attention_matrix = tf.SparseTensor(indices=logits.indices,
        #                                        values=tf.nn.leaky_relu(logits.values),
        #                                        dense_shape=logits.dense_shape)
        #
        #     # apply softmax on the axis 1 # with the catch that the implicitly zero elements do not participate.
        #     attention_matrix = tf.sparse.softmax(attention_matrix)
        #
        #
        #     #node dropout
        #     if self.node_dropout_flag:
        #         n_nonzero_temp = attention_matrix.values.shape  # Number of non-zero entries
        #         attention_matrix = self._dropout_sparse(attention_matrix, 1 - self.node_dropout[0], n_nonzero_temp)
        #
        #
        #     ego_embeddings = tf.sparse_tensor_dense_matmul(attention_matrix,h_embeddings) # returns a dense matrix (pseudo-code in dense np.matrix notation)
        #
        #     #apply ReLU(non-linear activation) (can apply directly, because the ego_embeddings is dense matrix)
        #     ego_embeddings = tf.nn.leaky_relu(ego_embeddings)
        #
        #     #############
        #     # message dropout.
        #     ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k]) # keep_rate = 1-mess_drop_rate
        #
        #     all_embeddings += [ego_embeddings]



        # #modified GAT for recommendation task
        for k in range(self.n_layers):

            #project node embeddings into another hidden space
            #h_embeddings = tf.matmul(ego_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k]

            #cancel transformation, inspired by xxxGCN
            h_embeddings = ego_embeddings

            #define two trainable parameter vectors in each GAT layer , a_1, a_2
            attention_part_I = tf.matmul(h_embeddings, self.weights['a_1_%d' % k])
            attention_part_II = tf.matmul(h_embeddings, self.weights['a_2_%d' % k])

            bi_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj)  # bidirectionally normalized adjacency matrix (without self_loop)

            ## implementation 1, apply softmax on att_mat firstly, then * GCN decay, then softmax again
            # plain_adj_mat =  tf.SparseTensor(indices=bi_norm_adj.indices,
            #                                    values=tf.ones(bi_norm_adj.values.shape),
            #                                    dense_shape=bi_norm_adj.dense_shape)
            #
            # attention_part_I = plain_adj_mat.__mul__(attention_part_I)
            # attention_part_II = plain_adj_mat.__mul__(tf.transpose(attention_part_II, [1, 0]))
            # logits = tf.sparse_add(attention_part_I, attention_part_II)  # At least one operand must be sparse.
            # #apply ReLU(non-linear activation) , tf.nn.leaky_relu() accepts – A `Tensor`, (去掉试一试)
            # attention_matrix = tf.SparseTensor(indices=logits.indices,
            #                                    values=tf.nn.leaky_relu(logits.values),
            #                                    dense_shape=logits.dense_shape)
            # #apply softmax on the axis 1 # with the catch that the implicitly zero elements do not participate.
            # attention_matrix = tf.sparse.softmax(attention_matrix)
            # combined_at_mat = tf.SparseTensor(indices=attention_matrix.indices,
            #                                    values=(attention_matrix.values * bi_norm_adj.values),
            #                                    dense_shape=attention_matrix.dense_shape)
            # attention_matrix=combined_at_mat
            # attention_matrix = tf.sparse.softmax(attention_matrix)
            ###############

            # implementing2,  att_mat * GCN decay firstly, then apply softmax
            attention_part_I = bi_norm_adj.__mul__(attention_part_I) #element-wise multiply
            attention_part_II = bi_norm_adj.__mul__(tf.transpose(attention_part_II, [1, 0]))

            logits  = tf.sparse_add(attention_part_I, attention_part_II) #At least one operand must be sparse.


            # apply ReLU(non-linear activation) , tf.nn.leaky_relu() accepts – A `Tensor`, (去掉试一试)
            attention_matrix = tf.SparseTensor(indices=logits.indices,values=tf.nn.leaky_relu(logits.values),dense_shape=logits.dense_shape)


            # implementing3,  cancel LeakyRelU on the top of implement 2
            #attention_matrix = logits

            # # implementing4, use plain adj_matrix and Mean-Pooling normalize attention matrix
            # rowsum = tf.sparse.reduce_sum(logits,axis=1)
            #
            # tf.sparse.
            # d_inv = np.power(rowsum, -1).flatten()
            # d_inv[np.isinf(d_inv)] = 0.  # change infinite values to 0
            # d_mat_inv = sp.diags(d_inv)
            #
            # norm_adj = d_mat_inv.dot(adj)


            # apply softmax on the axis 1 # with the catch that the implicitly zero elements do not participate.
            attention_matrix = tf.sparse.softmax(attention_matrix)




            #node dropout
            if self.node_dropout_flag:
                n_nonzero_temp = attention_matrix.values.shape  # Number of non-zero entries
                attention_matrix = self._dropout_sparse(attention_matrix, 1 - self.node_dropout[0], n_nonzero_temp)


            ego_embeddings = tf.sparse_tensor_dense_matmul(attention_matrix,h_embeddings) # returns a dense matrix (pseudo-code in dense np.matrix notation)

            #apply ReLU(non-linear activation) (can apply directly, because the ego_embeddings is dense matrix)
            #ego_embeddings = tf.nn.leaky_relu(ego_embeddings) #cancel nonlinear, inspired by xxxGCN

            #############
            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k]) # keep_rate = 1-mess_drop_rate

            # normalize the distribution of embeddings.(去掉试一试)
            #norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [ego_embeddings]

        if  self.layers_aggr =='concat':
            all_embeddings = tf.concat(all_embeddings, 1) #concatenate embeddings from different layers along Dimension 1
        elif self.layers_aggr =='avg':
            all_embeddings = sum(all_embeddings)/len(all_embeddings)
        elif self.layers_aggr =='weighted_avg':
            all_embeddings = tf.stack(all_embeddings,axis=2) #list of L (N,K)—>(N,K,L)

            all_embeddings = tf.math.multiply(all_embeddings,self.weights['weights'])#(N,K,L)*(L,)—>(N,K,L)
            #broadcasting weights when conducting element-wise multiplication
            all_embeddings = tf.reduce_sum(all_embeddings, axis=2) / tf.reduce_sum(self.weights['weights']) #weighted average

        elif self.layers_aggr =='weighted_concat':
            all_embeddings = tf.stack(all_embeddings,axis=2) #list of L (N,K)—>(N,K,L)
            #normlized_weights = self.weights['weights']/tf.reduce_sum(self.weights['weights'])  #报错,输出结果不是(L,) 了?

            all_embeddings = tf.math.multiply(all_embeddings,tf.math.sqrt(self.weights['weights'])) #(N,K,L)*(L,)—>(N,K,L)
            #broadcasting weights when conducting element-wise multiplication

            all_embeddings = all_embeddings/tf.math.sqrt(tf.reduce_sum(self.weights['weights']))
            all_embeddings = tf.unstack(all_embeddings,axis=2) #(N,K,L) —> list of L (N,K)
            all_embeddings = tf.concat(all_embeddings, 1) #concatenate embeddings from different layers along Dimension 1
        elif self.layers_aggr == 'last':
            all_embeddings = all_embeddings[-1]

        all_embeddings = tf.concat(all_embeddings, 1)  # concatenate embeddings from different layers along Dimension 1
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

##proposal 2: GNN layers with  different updating rules

    #another Single-interaction updater (concatenation)
    def _create_variant4_embed(self):
        # Generate a set of adjacency sub-matrix.
        # split the A_hat into fold parts for the memory efficiency
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
                # split the A_hat into fold parts for the memory efficiency

            # # sum messages of neighbors.
            # side_embeddings = tf.concat(temp_embed, 0) #concatenate list of tensor objects along Dimension 0
            # # transformed sum messages of neighbors.
            # sum_embeddings = tf.nn.leaky_relu(
            #     tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            #
            # # bi messages of neighbors.
            # bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # # transformed bi messages of neighbors.
            # bi_embeddings = tf.nn.leaky_relu(
            #     tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])
            #
            # # non-linear activation.
            # ego_embeddings = sum_embeddings + bi_embeddings

            ############# my version

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)  # concatenate list of tensor objects along Dimension 0
            # transformed sum messages of neighbors.
            sum_embeddings = tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k]

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k]

            ##context_representation
            neg_embeddings = sum_embeddings + bi_embeddings

            #"concatenation" updating rule
            # Concatenates the list of tensors values along dimension axis, W3*(e_u || e_N_u)
            tem = tf.matmul(tf.concat([ego_embeddings,neg_embeddings], axis=1), self.weights['W_squeeze_%d' % k]) + self.weights['b_squeeze_%d' % k]

            #  add up and then apply ReLU(non-linear activation)
            ego_embeddings = tf.nn.leaky_relu(tem)



            #############
            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        if self.layers_aggr == 'concat':
            all_embeddings = tf.concat(all_embeddings,
                                       1)  # concatenate embeddings from different layers along Dimension 1
        elif self.layers_aggr == 'avg':
            all_embeddings = sum(all_embeddings) / len(all_embeddings)
        elif self.layers_aggr == 'weighted_avg':
            all_embeddings = tf.stack(all_embeddings, axis=2)  # list of L (N,K)—>(N,K,L)

            all_embeddings = tf.math.multiply(all_embeddings, self.weights['weights'])  # (N,K,L)*(L,)—>(N,K,L)
            # broadcasting weights when conducting element-wise multiplication
            all_embeddings = tf.reduce_sum(all_embeddings, axis=2) / tf.reduce_sum(
                self.weights['weights'])  # weighted average

        elif self.layers_aggr == 'weighted_concat':
            all_embeddings = tf.stack(all_embeddings, axis=2)  # list of L (N,K)—>(N,K,L)
            # normlized_weights = self.weights['weights']/tf.reduce_sum(self.weights['weights'])  #报错,输出结果不是(L,) 了?

            all_embeddings = tf.math.multiply(all_embeddings,
                                              tf.math.sqrt(self.weights['weights']))  # (N,K,L)*(L,)—>(N,K,L)
            # broadcasting weights when conducting element-wise multiplication

            all_embeddings = all_embeddings / tf.math.sqrt(tf.reduce_sum(self.weights['weights']))
            all_embeddings = tf.unstack(all_embeddings, axis=2)  # (N,K,L) —> list of L (N,K)
            all_embeddings = tf.concat(all_embeddings,
                                       1)  # concatenate embeddings from different layers along Dimension 1
        elif self.layers_aggr == 'last':
            all_embeddings = all_embeddings[-1]

        all_embeddings = tf.concat(all_embeddings, 1)  # concatenate embeddings from different layers along Dimension 1
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    # multiple-interaction updater (concatenation+add)
    def _create_variant5_embed(self):
        # Generate a set of adjacency sub-matrix.
        # split the A_hat into fold parts for the memory efficiency
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
                # split the A_hat into fold parts for the memory efficiency

            # # sum messages of neighbors.
            # side_embeddings = tf.concat(temp_embed, 0) #concatenate list of tensor objects along Dimension 0
            # # transformed sum messages of neighbors.
            # sum_embeddings = tf.nn.leaky_relu(
            #     tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            #
            # # bi messages of neighbors.
            # bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # # transformed bi messages of neighbors.
            # bi_embeddings = tf.nn.leaky_relu(
            #     tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])
            #
            # # non-linear activation.
            # ego_embeddings = sum_embeddings + bi_embeddings

            ############# my version

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)  # concatenate list of tensor objects along Dimension 0
            # transformed sum messages of neighbors.
            sum_embeddings = tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k]

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k]

            ##context_representation
            neg_embeddings = sum_embeddings + bi_embeddings

            # "concatenation" updating rule
            # Concatenates the list of tensors values along dimension axis, W3*(e_u || e_N_u)
            tem = tf.matmul(tf.concat([ego_embeddings, neg_embeddings], axis=1), self.weights['W_squeeze_%d' % k]) + \
                  self.weights['b_squeeze_%d' % k]

            #  add up and then apply ReLU(non-linear activation)
            ego_embeddings = tf.nn.leaky_relu(tem)

            #############
            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        if self.layers_aggr == 'concat':
            all_embeddings = tf.concat(all_embeddings,
                                       1)  # concatenate embeddings from different layers along Dimension 1
        elif self.layers_aggr == 'avg':
            all_embeddings = sum(all_embeddings) / len(all_embeddings)
        elif self.layers_aggr == 'weighted_avg':
            all_embeddings = tf.stack(all_embeddings, axis=2)  # list of L (N,K)—>(N,K,L)

            all_embeddings = tf.math.multiply(all_embeddings, self.weights['weights'])  # (N,K,L)*(L,)—>(N,K,L)
            # broadcasting weights when conducting element-wise multiplication
            all_embeddings = tf.reduce_sum(all_embeddings, axis=2) / tf.reduce_sum(
                self.weights['weights'])  # weighted average

        elif self.layers_aggr == 'weighted_concat':
            all_embeddings = tf.stack(all_embeddings, axis=2)  # list of L (N,K)—>(N,K,L)
            # normlized_weights = self.weights['weights']/tf.reduce_sum(self.weights['weights'])  #报错,输出结果不是(L,) 了?

            all_embeddings = tf.math.multiply(all_embeddings,
                                              tf.math.sqrt(self.weights['weights']))  # (N,K,L)*(L,)—>(N,K,L)
            # broadcasting weights when conducting element-wise multiplication

            all_embeddings = all_embeddings / tf.math.sqrt(tf.reduce_sum(self.weights['weights']))
            all_embeddings = tf.unstack(all_embeddings, axis=2)  # (N,K,L) —> list of L (N,K)
            all_embeddings = tf.concat(all_embeddings,
                                       1)  # concatenate embeddings from different layers along Dimension 1
        elif self.layers_aggr == 'last':
            all_embeddings = all_embeddings[-1]

        all_embeddings = tf.concat(all_embeddings, 1)  # concatenate embeddings from different layers along Dimension 1
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

#proposal 3 (layer-aggregators) is integrated in each type of GNN layer
#proposal 4 (neural interaction function) is at the batch_test and BPR loss part
################################
    # MF baseline
    def _create_mf_embed(self): #matrix factorization
        u_g_embeddings, i_g_embeddings = self.weights['user_embedding'], self.weights['item_embedding']
        return u_g_embeddings, i_g_embeddings

    #original NGCF baseline
    def _create_ngcf_embed(self):
        # Generate a set of adjacency sub-matrix.
        # split the A_hat into fold parts for the memory efficiency
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
                # split the A_hat into fold parts for the memory efficiency

            # # sum messages of neighbors.
            # side_embeddings = tf.concat(temp_embed, 0) #concatenate list of tensor objects along Dimension 0
            # # transformed sum messages of neighbors.
            # sum_embeddings = tf.nn.leaky_relu(
            #     tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            #
            # # bi messages of neighbors.
            # bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # # transformed bi messages of neighbors.
            # bi_embeddings = tf.nn.leaky_relu(
            #     tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])
            #
            # # non-linear activation.
            # ego_embeddings = sum_embeddings + bi_embeddings

            ############# my version

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)  # concatenate list of tensor objects along Dimension 0
            # transformed sum messages of neighbors.
            sum_embeddings = tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k]

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, (side_embeddings - ego_embeddings) )  #because side_embeddings = (L+I)E, we just need LE, so minus E
            # transformed bi messages of neighbors.
            bi_embeddings = tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k]

            #  add up and then apply ReLU(non-linear activation)
            ego_embeddings = tf.nn.leaky_relu(sum_embeddings + bi_embeddings)


            #############
            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        if self.layers_aggr == 'concat':
            all_embeddings = tf.concat(all_embeddings,
                                       1)  # concatenate embeddings from different layers along Dimension 1
        elif self.layers_aggr == 'avg':
            all_embeddings = sum(all_embeddings) / len(all_embeddings)
        elif self.layers_aggr == 'weighted_avg':
            all_embeddings = tf.stack(all_embeddings, axis=2)  # list of L (N,K)—>(N,K,L)

            all_embeddings = tf.math.multiply(all_embeddings, self.weights['weights'])  # (N,K,L)*(L,)—>(N,K,L)
            # broadcasting weights when conducting element-wise multiplication
            all_embeddings = tf.reduce_sum(all_embeddings, axis=2) / tf.reduce_sum(
                self.weights['weights'])  # weighted average

        elif self.layers_aggr == 'weighted_concat':
            all_embeddings = tf.stack(all_embeddings, axis=2)  # list of L (N,K)—>(N,K,L)
            # normlized_weights = self.weights['weights']/tf.reduce_sum(self.weights['weights'])  #报错,输出结果不是(L,) 了?

            all_embeddings = tf.math.multiply(all_embeddings,
                                              tf.math.sqrt(self.weights['weights']))  # (N,K,L)*(L,)—>(N,K,L)
            # broadcasting weights when conducting element-wise multiplication

            all_embeddings = all_embeddings / tf.math.sqrt(tf.reduce_sum(self.weights['weights']))
            all_embeddings = tf.unstack(all_embeddings, axis=2)  # (N,K,L) —> list of L (N,K)
            all_embeddings = tf.concat(all_embeddings,
                                       1)  # concatenate embeddings from different layers along Dimension 1
        elif self.layers_aggr == 'last':
            all_embeddings = [all_embeddings[-1]]

        all_embeddings = tf.concat(all_embeddings, 1)  # concatenate embeddings from different layers along Dimension 1
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    #classic GCN layers
    def _create_gcn_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        ##split the A_hat into fold parts for the memory efficiency
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' %k]) + self.weights['b_gc_%d' %k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    #GCMC layers
    def _create_gcmc_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        #split the A_hat into fold parts for the memory efficiency

        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = []

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            # convolutional layer.
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # dense layer.
            mlp_embeddings = tf.matmul(embeddings, self.weights['W_mlp_%d' %k]) + self.weights['b_mlp_%d' %k]
            mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [mlp_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    #create BPR loss
    def create_bpr_loss(self, users, pos_items, neg_items):

        if self.intera_func == 'inner_prod':
            pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
            neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        ##proposal 4 (neural interaction function - MLP)
        elif self.intera_func == 'neural':
            #拼起来
            W_l1_p1 = tf.concat([self.weights['w%d_mlp_l1_p1' % k] for k in range(sum(self.weight_size_list))],axis=1)
            W_l1_p2 = tf.concat([self.weights['w%d_mlp_l1_p2' % k] for k in range(sum(self.weight_size_list))], axis=1)
            W_mlp_l1 = tf.concat([W_l1_p1,W_l1_p2],axis=0)

            pos_concat = tf.concat([users, pos_items], axis=1)
            neg_concat = tf.concat([users, neg_items], axis=1)

            tem = tf.matmul(pos_concat, W_mlp_l1) + self.weights['b_mlp_l1'] #first layer
            tem = tf.nn.leaky_relu(tem)  #nonlinear layer
            tem = tf.matmul(tem, self.weights['W_mlp_l2']) + self.weights['b_mlp_l2'] #second layer
            tem = tf.nn.leaky_relu(tem)  # nonlinear layer
            tem = tf.matmul(tem, self.weights['W_mlp_l3']) + self.weights['b_mlp_l3']  # third layer
            tem = tf.nn.leaky_relu(tem)  # nonlinear layer
            pos_scores = tf.matmul(tem, self.weights['W_mlp_l4']) + self.weights['b_mlp_l4']  #fourth layer

            tem = tf.matmul(neg_concat, W_mlp_l1) + self.weights['b_mlp_l1'] #first layer
            tem = tf.nn.leaky_relu(tem)  # nonlinear layer
            tem = tf.matmul(tem, self.weights['W_mlp_l2']) + self.weights['b_mlp_l2']  # second layer
            tem = tf.nn.leaky_relu(tem)  # nonlinear layer
            tem = tf.matmul(tem, self.weights['W_mlp_l3']) + self.weights['b_mlp_l3']  # third layer
            tem = tf.nn.leaky_relu(tem)  # nonlinear layer
            neg_scores = tf.matmul(tem, self.weights['W_mlp_l4']) + self.weights['b_mlp_l4']  # fourth layer

        ##proposal 4 (neural interaction function - fusion of MLP and genralized inner product)
        elif self.intera_func == 'fusion':
            #拼起来
            W_l1_p1 = tf.concat([self.weights['w%d_mlp_l1_p1' % k] for k in range(sum(self.weight_size_list))],axis=1)
            W_l1_p2 = tf.concat([self.weights['w%d_mlp_l1_p2' % k] for k in range(sum(self.weight_size_list))], axis=1)
            W_mlp_l1 = tf.concat([W_l1_p1,W_l1_p2],axis=0)

            pos_concat = tf.concat([users, pos_items], axis=1)
            pos_elem_prod = tf.multiply(users, pos_items) #element-wise product part

            neg_concat = tf.concat([users, neg_items], axis=1)
            neg_elem_prod = tf.multiply(users, neg_items) #element-wise product part

            #positive part
            tem = tf.matmul(pos_concat, W_mlp_l1) + self.weights['b_mlp_l1'] #first layer
            tem = tf.nn.leaky_relu(tem)  #nonlinear layer
            tem = tf.matmul(tem, self.weights['W_mlp_l2']) + self.weights['b_mlp_l2'] #second layer
            tem = tf.nn.leaky_relu(tem)  # nonlinear layer
            tem = tf.matmul(tem, self.weights['W_mlp_l3']) + self.weights['b_mlp_l3']  # third layer
            tem = tf.nn.leaky_relu(tem)  # nonlinear layer

            tem = tf.concat([0.4*tem,0.6*pos_elem_prod],axis=1) #weighted concatenation
            pos_scores = tf.matmul(tem, self.weights['W_fusion'])

            #negative part
            tem = tf.matmul(neg_concat, W_mlp_l1) + self.weights['b_mlp_l1'] #first layer
            tem = tf.nn.leaky_relu(tem)  # nonlinear layer
            tem = tf.matmul(tem, self.weights['W_mlp_l2']) + self.weights['b_mlp_l2']  # second layer
            tem = tf.nn.leaky_relu(tem)  # nonlinear layer
            tem = tf.matmul(tem, self.weights['W_mlp_l3']) + self.weights['b_mlp_l3']  # third layer
            tem = tf.nn.leaky_relu(tem)  # nonlinear layer

            tem = tf.concat([0.4*tem, 0.6*neg_elem_prod], axis=1)
            neg_scores = tf.matmul(tem, self.weights['W_fusion'])

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        # why other parameters W and b are not regularized????

        regularizer = regularizer/self.batch_size
        
        # In the first version, we implement the bpr loss via the following codes:
        # We report the performance in our paper using this implementation.
        # maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        # mf_loss = tf.negative(tf.reduce_mean(maxi))
        
        ## In the second version, we implement the bpr loss via the following codes to avoid 'NAN' loss during training:
        ## However, it will change the training performance and training performance.
        ## Please retrain the model and do a grid search for the best experimental setting.
        mf_loss = tf.reduce_sum(tf.nn.softplus(-(pos_scores - neg_scores))) #this loss

        emb_loss = self.decay * regularizer



        #regularization loss of all model parameters (excepting for embeddings)

        # reg_loss = tf.constant(0.0, tf.float32, [1])
        reg_loss = tf.nn.l2_loss(self.weights['weights'])

        ## parameters W and b
        for k in range(self.n_layers):
             reg_loss += (tf.nn.l2_loss(self.weights['W_gc_%d' %k]) + tf.nn.l2_loss(self.weights['b_gc_%d' %k]) + tf.nn.l2_loss(self.weights['W_mlp_%d' % k]) + tf.nn.l2_loss(self.weights['b_mlp_%d' % k]) + tf.nn.l2_loss(self.weights['a_1_%d' % k]) + tf.nn.l2_loss(self.weights['a_2_%d' % k]))

        #regularization for nueral interaction function
        #first layer parameters
        for k in range(sum(self.weight_size_list)):
            reg_loss += (tf.nn.l2_loss(self.weights['w%d_mlp_l1_p1' % k]) + tf.nn.l2_loss(self.weights['w%d_mlp_l1_p2' % k]))

        #2-4 layers
        for k in range(2,5):
            reg_loss += (tf.nn.l2_loss(self.weights['W_mlp_l%d' % k]) + tf.nn.l2_loss(self.weights['b_mlp_l%d' % k]) )

        #fusion weights
        reg_loss += tf.nn.l2_loss(self.weights['W_fusion'])

        reg_loss = self.decay * reg_loss

        return mf_loss, emb_loss, reg_loss

    #auxiliary functions
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose() #row-major order
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _convert_sp_tensor_to_sp_mat(self, X): #define by yuli_li
        data=X.values#.numpy()
        indices = X.indices#.numpy() #tensor is not subscriptable
        shape = X.dense_shape#.numpy()
        coo = sp.coo_matrix((data,(indices[:,0],indices[:,1])),shape=shape)
        return coo.tocsr()

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape) #keep_prob + [0-1] uniform distribution
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool) #0，1 mask, floor_round to 0 or 1
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob) #retained entries are multiply by 1/keep_prob

#funtion for loading pretrained (user and item) embeddings
def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

#training loop + validation + testing
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj, bi_norm_adj = data_generator.get_adj_mat() #geennrate adj_matrices in different form

    #assign according to "adj_type" argument
    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')

    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')
        # unidirectionally normalized adjacency matrix (with self_loop)

    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        print('use the gcmc adjacency matrix')
        #unidirectionally normalized adjacency matrix (without self_loop)

    elif args.adj_type == 'bi_norm':
        config['norm_adj'] = bi_norm_adj
        print('use the bidirectionally normalized adjacency matrix')
        # bidirectionally normalized adjacency matrix (without self_loop)

    #ngcf (by default), where each decay factor between two connected nodes is set as 1 /(out degree of the node),
    #each node is also assigned with 1 for self-connections
    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')

    t0 = time()

    if args.pretrain == -1: # 0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    model = NGCF(data_config=config, pretrain_data=pretrain_data)
    #instantiation of the model class, the computational graph will be built(launched)

    """
    *********************************************************
    Save the model parameters.
    """
    saver = tf.train.Saver()

    if args.save_flag == 1: #0: Disable model saver, 1: Activate model saver
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)



    config = tf.ConfigProto() #The ConfigProto protocol buffer exposes various configuration options for a session
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    #config a seesion,A Session object encapsulates the environment in which Operation objects are executed,
    # and Tensor objects are evaluated. Launch the graph in a session.

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    if args.pretrain == 1: # 0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models
        layer = '-'.join([str(l) for l in eval(args.layer_size)])

        pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                        str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))


        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from pretrained model.
            if args.report != 1:
                users_to_test = list(data_generator.test_set.keys())
                ret = test(sess, model, users_to_test, batch_test_flag=True, drop_flag=True)
                cur_best_pre_0 = ret['recall'][0]

                pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                               'ndcg=[%.5f, %.5f], MRR=[%.5f]' % \
                               (ret['recall'][0], ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1],
                                ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1],
                                ret['rr'] )
                            # ( highest value, lowest value )
                print(pretrain_ret)
        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')

    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')

    """
    *********************************************************
    Get the performance w.r.t. different sparsity levels.
    """
    #report=0: Disable performance report w.r.t. sparsity levels, report=1: Show performance report w.r.t. sparsity levels
    if args.report == 1:
        assert args.test_flag == 'full'
        users_to_test_list, split_state = data_generator.get_sparsity_split()
        users_to_test_list.append(list(data_generator.test_set.keys()))
        split_state.append('all')

        report_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(report_path)
        f = open(report_path, 'w')
        f.write(
            'embed_size=%d, lr=%.4f, layer_size=%s, keep_prob=%s, regs=%s, loss_type=%s, adj_type=%s\n'
            % (args.embed_size, args.lr, args.layer_size, args.keep_prob, args.regs, args.loss_type, args.adj_type))

        for i, users_to_test in enumerate(users_to_test_list):
            ret = test(sess, model, users_to_test,batch_test_flag=True,drop_flag=True)

            final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s], MRR=[%s]" % \
                         ('\t'.join(['%.5f' % r for r in ret['recall']]),
                          '\t'.join(['%.5f' % r for r in ret['precision']]),
                          '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                          '\t'.join(['%.5f' % r for r in ret['ndcg']]),
                          '%.5f'%(ret['rr']))
            print(final_perf)

            f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        f.close()
        exit()

    """
    *********************************************************
    Train loops
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0 #counter for the successive decreasing evaluations on the test set, trigger early stop
    should_stop = False

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = sess.run([model.opt, model.loss, model.mf_loss, model.emb_loss, model.reg_loss],
                               feed_dict={model.users: users, model.pos_items: pos_items,
                                          model.node_dropout: eval(args.node_dropout),
                                          model.mess_dropout: eval(args.mess_dropout),
                                          model.neg_items: neg_items})


            loss += batch_loss  #total loss on all training pairs (as many as the numbers of training interactions)
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            reg_loss += batch_reg_loss


        loss_loger.append(loss) #log training loss in each epoch

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 20 epochs;
        if (epoch + 1) % 20 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, reg_loss)
                print(perf_str)
            continue  # It continues with the next cycle of the nearest enclosing for loop.

        #test on validation set for each 20 epoches
        t2 = time()
        users_to_test = list(data_generator.test_set.keys())


        if args.intera_func in ['neural','fusion']:
            #when adopt neural interaction fucntion, we have to adopt split items in to batches for avoid overflowing
            ret = test(sess, model, users_to_test,batch_test_flag=True, drop_flag=True)
        else:
            ret = test(sess, model, users_to_test, batch_test_flag=False, drop_flag=True)

        t3 = time()

        # loss_loger.append(loss) #moved upwards
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f], MRR=[%.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1],ret['rr'])
            print(perf_str)


        if args.alg_type == 'gat':
            flag_step = 5 #assign same tolerant steps (for triggering early stop) when using gat
        else:
            flag_step = 5

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=flag_step)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for 5 successive evaluations.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
        #if args.save_flag == 1:          # temporary modification for keeping training
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

####save embedings for further usage
    user_embed, item_embed = sess.run([model.weights['user_embedding'], model.weights['item_embedding']])
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    ensureDir(pretrain_path)
    np.savez(pretrain_path, user_embed=user_embed, item_embed=item_embed)

    recs = np.array(rec_loger) #2-D array , each row represents: recal@20 ,40, 60, 80, 100
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    """
    ################ploting part ################ yuli_li
    """
    # plot with various axes scales
    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    plt.plot(np.arange(len(loss_loger)), loss_loger, 'r-')  # show the training loss
    plt.ylabel('training loss(BPR)')
    plt.xlabel('epoch')
    plt.title('training,lr=%.5f,batch_size=%d'%(args.lr,args.batch_size))
    plt.grid(True)

    # log
    plt.subplot(122)
    plt.plot(np.arange(20, len(loss_loger)+1, 20), recs[:,0], 'ro-',label='recall@20')  # show the recall@20 on the validation set
    plt.plot(np.arange(20, len(loss_loger)+1, 20), ndcgs[:,0], 'go-', label='ndcg@20')  # show the ndcg@20 on the validation set
    plt.plot(np.arange(20, len(loss_loger)+1, 20), pres[:,0], 'bo-',label='precison@20')  # show the precison@20 on the validation set
    plt.plot(np.arange(20, len(loss_loger)+1, 20), hit[:,0], 'yo-', label='hit@20')  # show the hit@20 on the validation set
    plt.ylabel('validation metrics')
    plt.xlabel('epoch')
    plt.legend()
    plt.title('validation')
    plt.grid(True)
    # plt.show()
    plt.savefig('./loss_record.png')
    ###############################################

    best_rec_0 = max(recs[:, 0])

    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s], MRR=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]),
                  '%.5f'%(ret['rr']))
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write(
        'embed_size=%d, lr=%.5f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
        % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs,
           args.adj_type, final_perf))
    f.close()
