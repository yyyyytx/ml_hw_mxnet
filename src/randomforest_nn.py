import numpy as np
import pandas as pd
import time
import math
import random
from nn import net_node, predict_label
from mxnet import ndarray as nd

def count_max(x):
    return x.value_counts().index[0]

class Random_Forest:
    def __init__(self,train_features,train_labels,num_of_nn):
        self.features = train_features
        self.labels = train_labels
        self.num_of_nn = int(num_of_nn)
        self.num_of_sample = int(self.features.shape[0])
        self.num_of_features = int(self.features.shape[1])
        self.train_index = np.array([])
        self.features_index_list = []

    def create_node(self):
        self.node_list=[]
        for i in np.arange(self.num_of_nn):
            feature_index = np.array(random.sample(range(self.num_of_features),5))
            self.features_index_list.append(feature_index)
            random_array = np.random.randint(0,self.num_of_sample,self.num_of_sample)
            contain_array = np.arange(i,self.num_of_sample,self.num_of_nn)
            train_index = np.unique(np.append(random_array,contain_array))
            self.train_index = np.unique(np.append(self.train_index,train_index))
            # features = self.features.iloc[train_index,feature_index]
            features = self.features.iloc[:,feature_index]
            labels = self.labels
            features = nd.array(features)
            labels = nd.array(labels)
            param_path = './params/rf%d.params' % i
            net = net_node(features, labels, param_path)
            # ct = DecisionTreeClassifier().fit(features,labels)
            self.node_list.append(net)
        return self.node_list

    def predict(self,test_features):
        df=pd.DataFrame()
        print('start predict')
        for i,nn in enumerate(self.node_list):
            features = nd.array(test_features.iloc[:,self.features_index_list[i]])
            predict = nn(features)
            col_name = 'nn' + str(i)
            predict = predict_label(predict)
            df[col_name] = predict.asnumpy()
        predict_list=df.apply(count_max,axis=1)
        return predict_list

if __name__ == '__main__':
    from load_data import train_features, train_labels, test_id, test_features, split_train_data, evaliate
    from pandas import DataFrame

    tfs, tls, vfs, vls = split_train_data()
    i = 0
    mean_acc = 0
    for (features, labels, vfeatures, vlabels) in zip(tfs, tls, vfs, vls):
        print('bag : ', i, ' start')
        bag = Random_Forest(train_features=train_features,
                        train_labels=train_labels,
                        num_of_nn=20)
        bag.create_node()
        # vfeatures = nd.array(vfeatures)
        predict = bag.predict(vfeatures)
        accuracy = evaliate(predict, vlabels)
        mean_acc += accuracy
        print('bag ', i, ' :', accuracy)
        i += 1
        break
    print('mean acc : ', mean_acc / 10)
    # bag = Bagging(train_features=train_features,
    #                     train_labels=train_labels,
    #                     num_of_nn=20)
    # bag.create_node()
    # test_features = nd.array(test_features)
    # predict = bag.predict(test_features)
    # print(predict)
    # predict += 1
    # data = {'Cover_Type': predict.astype(int).tolist()}
    # result = pd.DataFrame(data, columns=['Cover_Type'], index=test_id.astype(int))
    # result.to_csv('result.csv')
    # print(result)