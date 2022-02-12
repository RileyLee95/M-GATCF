'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import *
import tensorflow as tf
import multiprocessing
import heapq

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)

USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1]) #sorted by scores in ascending order
    item_score.reverse() #sorted by scores in descending order
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def get_rr(r): #reciprocal rank, yuli_li
    try:
        rank_of_first_p = r.index(1)+1    #rank starts from 1,but index starts from 0
        rr = 1/rank_of_first_p
    except Exception:
        rr = 0   #if there is no true positive item returned within top 100 , rr =0
    return rr

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)#returns top K scored items' IDs

    r = [] # boolean marks indicating if each of top-K scored items is a truly positive item in the test set
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    rr= get_rr(r)  #yuli_li

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc,'rr':rr}


def test_one_user(x):

    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]

    #user u's interacted items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []

    #user u's interacted items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items)) #all items except positive items appeared in training set

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def test(sess, model, users_to_test, drop_flag=False, batch_test_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0., 'rr': 0.}

    pool = multiprocessing.Pool(cores)

    #u_batch_size = BATCH_SIZE * 2  #why?? no reason....
    #i_batch_size = BATCH_SIZE

    #yuli_li, for making sure tensors will not be too big when using nuearl interaction function
    # batch size can not be wholly divided by n_test_users(1000) and n_items (40322),
    if args.intera_func == 'neural':
        u_batch_size = 602
        i_batch_size = 1024

    if args.intera_func == 'fusion':
        u_batch_size = 602
        i_batch_size = 1024

    else:
        u_batch_size = BATCH_SIZE * 2  # why?? no reason....
        i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        #a slice of test_users, if "end" index is larger than the length, doesn't matter

        if batch_test_flag:

            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)

                if drop_flag == False:
                    i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})

                else:
                    i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch,
                                                                model.node_dropout: [0.]*len(eval(args.layer_size)),
                                                                model.mess_dropout: [0.]*len(eval(args.layer_size))})

                #mistake 1
                #if args.intera_func == 'neural':
                    # i_rate_batch = tf.reshape(i_rate_batch, [len(user_batch), -1])  # it will just create a symbolic tensor
                    # rate_batch[:, i_start: i_end] = np.array(i_rate_batch) #NotImplementedError: Cannot convert a symbolic Tensor (

               #Right way to do it
                i_rate_batch = np.array(i_rate_batch)       #convert to ndarray, don't built graph, in fact it's already a adarray..

                if args.intera_func in ['neural','fusion']:
                    i_rate_batch =  i_rate_batch.reshape((len(user_batch), -1))   # 临时，yuli_li

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            item_batch = range(ITEM_NUM)

            if drop_flag == False:
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                              model.pos_items: item_batch})

            else:
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                              model.pos_items: item_batch,
                                                              model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                              model.mess_dropout: [0.] * len(eval(args.layer_size))})

            #mistake 1
            #if args.intera_func == 'neural':
                ## if select "neural interaction function", returns tensor size= ( len(user_batch)*len(item_batch), 1)
                #rate_batch = tf.reshape(rate_batch, [-1,ITEM_NUM])  # it will just create a symbolic tensor

            if args.intera_func in ['neural','fusion']:
                rate_batch = np.array(rate_batch)  # convert to ndarray, don't built graph, in fact it's already a ndarray....
                # if select "neural interaction function", returns ndarray size= ( len(user_batch)*len(item_batch), 1)
                rate_batch = rate_batch.reshape((-1,ITEM_NUM))

        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid) #apply test_one_user function on each element of user_batch_rating_uid
        count += len(batch_result)

        for re in batch_result:
            #each element in batch_result is evaluation performance of a user
            # formatted as {'precision':array[pre@k1, pre@k2,...,pre@kn], 'recall':array[...] ...}
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users
            result['rr'] += re['rr'] / n_test_users #mean reciprocal rank

    assert count == n_test_users
    pool.close()
    return result
