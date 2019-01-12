from mxnet import gluon
import mxnet as mx
from mxnet import autograd as ag
from mxnet import ndarray as nd
import numpy as np
import pandas as pd
import time
import math
import random
from nn import net_node, predict_label

def count_max(x):
    return x.value_counts().index[0]

class Bagging:
    def __init__(self, train_features, train_labels, num_of_nn):
        self.features = train_features
        self.labels = train_labels
        self.num_of_nn = int(num_of_nn)
        self.num_of_sample = int(self.features.shape[0])
        self.train_index = np.array([])

    def create_node(self):
        self.node_list = []
        for i in np.arange(self.num_of_nn):
            random_array = np.random.randint(0, self.num_of_sample,
                                             self.num_of_sample - int(self.num_of_sample / self.num_of_nn))
            contain_array = np.arange(i, self.num_of_sample, self.num_of_nn)
            train_index = np.unique(np.append(random_array, contain_array))
            self.train_index = np.unique(np.append(self.train_index, train_index))
            features = self.features.iloc[train_index, :]
            labels = self.labels[train_index]
            features = nd.array(features)
            labels = nd.array(labels)
            print('node : %d start' % (i))
            param_path = './params/net%d.params' % i
            net = net_node(features, labels, param_path)
            print('node : %d finished' % (i))
            self.node_list.append(net)
        return self.node_list

    # plurality voting
    def predict(self, test_features):
        df = pd.DataFrame()
        print('start predict')
        for i, nn in enumerate(self.node_list):
            test_features = nd.array(test_features)
            predict = nn(test_features)
            col_name = 'nn' + str(i)
            predict = predict_label(predict)
            df[col_name] = predict.asnumpy()
        predict_list = df.apply(count_max, axis=1)
        return predict_list


if __name__ == '__main__':
    from load_data import train_features, train_labels, test_id, test_features, split_train_data, evaliate
    from pandas import DataFrame

    # tfs, tls, vfs, vls = split_train_data()
    # i = 0
    # mean_acc = 0
    # for (features, labels, vfeatures, vlabels) in zip(tfs, tls, vfs, vls):
    #     print('bag : ', i, ' start')
    #     bag = Bagging(train_features=train_features,
    #                     train_labels=train_labels,
    #                     num_of_nn=20)
    #     bag.create_node()
    #     vfeatures = nd.array(vfeatures)
    #     predict = bag.predict(vfeatures)
    #     accuracy = evaliate(predict, vlabels)
    #     mean_acc += accuracy
    #     print('bag ', i, ' :', accuracy)
    #     i += 1
    #     break
    # print('mean acc : ', mean_acc / 10)
    bag = Bagging(train_features=train_features,
                        train_labels=train_labels,
                        num_of_nn=20)
    bag.create_node()
    test_features = nd.array(test_features)
    predict = bag.predict(test_features)
    print(predict)
    predict += 1
    data = {'Cover_Type': predict.astype(int).tolist()}
    result = pd.DataFrame(data, columns=['Cover_Type'], index=test_id.astype(int))
    result.to_csv('result.csv')
    print(result)