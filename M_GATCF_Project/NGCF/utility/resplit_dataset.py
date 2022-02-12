
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import matplotlib.pyplot as plt

loss_loger = np.random.randint(2000,3000,100)
rec_loger = np.random.rand(10)
ndcg_loger = np.random.rand(10)
lr=0.0002
batch_size=3072
################ploting part ################ yuli_li
# plot with various axes scales
plt.figure(figsize=(14,8))
plt.subplot(121)
plt.plot(np.arange(len(loss_loger)),loss_loger,'r-') #show the training loss
plt.ylabel('training loss')
plt.xlabel('epoch')
plt.title('training,lr=%.5f,batch_size=%d' % (lr, batch_size))
plt.grid(True)


# log
plt.subplot(122)
plt.plot(np.arange(10,len(loss_loger)+1,10), rec_loger, 'ro-',label='recall')  # show the recall on the validation set
plt.plot(np.arange(10, len(loss_loger)+1, 10), ndcg_loger, 'go-',label='ndcg')  # show the ndcg on the validation set
plt.ylabel('validation metrics')
plt.xlabel('epoch')
plt.legend()
plt.title('validation')
plt.grid(True)
# plt.show()
plt.savefig('./loss_record.png')

# train_file = '../Data/gowalla/train.txt'
# test_file = '../Data/gowalla/test.txt'
#
# #initialization
# n_users = 0
# n_items = 0
# exist_users = []
# n_train = 0
# n_test = 0
#
# with open(train_file) as f:
#     for l in f.readlines():
#         if len(l) > 0:
#             l = l.strip('\n').split(' ')
#             items = [int(i) for i in l[1:]]
#             uid = int(l[0])
#             exist_users.append(uid)
#             n_items = max(n_items, max(items))
#             n_users = max(n_users, uid)
#             n_train += len(items)
#
# with open(test_file) as f:
#     for l in f.readlines():
#         if len(l) > 0:
#             l = l.strip('\n')
#             try:
#                 items = [int(i) for i in l.split(' ')[1:]]
#             except Exception:
#                 continue
#             n_items = max(n_items, max(items))
#             n_test += len(items)
#
#
# n_items += 1 #index begins from 0
# n_users += 1 #index begins from 0
#
#
# #determine the number of items and users and ,therefore, the shape of adj_matrix
# R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
#
# train_items = {}
# test_set = {}
#
# with open(train_file) as f_train:
#     with open(test_file) as f_test:
#         for l in f_train.readlines():
#             if len(l) == 0: break
#             l = l.strip('\n')
#             items = [int(i) for i in l.split(' ')]
#             uid, train_items = items[0], items[1:]
#
#             for i in train_items:
#                 R[uid, i] = 1.
#
#             train_items[uid] = train_items
#
#         for l in f_test.readlines():
#             if len(l) == 0: break
#             l = l.strip('\n')
#             try:
#                 items = [int(i) for i in l.split(' ')]
#             except Exception:
#                 continue
#
#             uid, test_items = items[0], items[1:]
#             test_set[uid] = test_items
