from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import numpy as np
import pandas as pd
import time
import math
import random



def count_max(x):
    return x.value_counts().index[0]

class Random_Forest:
    def __init__(self, train_features, train_labels, num_of_tree):
        self.features = train_features
        self.labels = train_labels
        self.num_of_tree = int(num_of_tree)
        self.num_of_sample = int(self.features.shape[0])
        self.num_of_features = int(self.features.shape[1])
        self.train_index = np.array([])
        self.features_index_list = []

    def create_tree(self):
        self.tree_list = []
        # print('start create tree, total ',self.num_of_tree,' trees')
        for i in np.arange(self.num_of_tree):
            # print('create tree',i)
            feature_index = np.array(random.sample(range(self.num_of_features), int(math.log2(self.num_of_features))))
            self.features_index_list.append(feature_index)
            # feature_index = np.random.randint(0, self.num_of_features,int(math.log2(self.num_of_features)))
            random_array = np.random.randint(0, self.num_of_sample, self.num_of_sample)
            contain_array = np.arange(i, self.num_of_sample, self.num_of_tree)
            train_index = np.unique(np.append(random_array, contain_array))
            self.train_index = np.unique(np.append(self.train_index, train_index))
            features = self.features.iloc[train_index, feature_index]
            labels = self.labels[train_index]
            ct = DecisionTreeClassifier().fit(features, labels)
            self.tree_list.append(ct)
        # print('tree create finished')
        return self.tree_list

    def predict(self, test_features):
        # predict_list = np.array([])
        df = pd.DataFrame()
        print('start predict')
        for i, ct in enumerate(self.tree_list):
            predict = ct.predict(test_features.iloc[:, self.features_index_list[i]])
            col_name = 'tree' + str(i)
            df[col_name] = predict

        start = time.time()
        predict_list = df.apply(count_max, axis=1)
        end = time.time()
        # print('predict ',test_features.shape[0],' examples time :',end-start)
        # print('predict finished')
        return predict_list


class Bagging:
    def __init__(self, train_features, train_labels, num_of_tree):
        self.features = train_features
        self.labels = train_labels
        self.num_of_tree = int(num_of_tree)
        self.num_of_sample = int(self.features.shape[0])
        self.train_index = np.array([])

    def create_tree(self):
        self.tree_list = []
        for i in np.arange(self.num_of_tree):
            random_array = np.random.randint(0, self.num_of_sample,
                                             self.num_of_sample - int(self.num_of_sample / self.num_of_tree))
            contain_array = np.arange(i, self.num_of_sample, self.num_of_tree)
            train_index = np.unique(np.append(random_array, contain_array))
            self.train_index = np.unique(np.append(self.train_index, train_index))
            features = self.features.iloc[train_index, :]
            labels = self.labels[train_index]
            ct = DecisionTreeClassifier().fit(features, labels)
            self.tree_list.append(ct)
        return self.tree_list

    # plurality voting
    def predict(self, test_features):
        # predict_list = np.array([])
        df = pd.DataFrame()
        print('start predict')
        for i, ct in enumerate(self.tree_list):
            predict = ct.predict(test_features)
            col_name = 'tree' + str(i)
            df[col_name] = predict

        start = time.time()
        predict_list = df.apply(count_max, axis=1)
        end = time.time()
        # print('predict ',test_features.shape[0],' examples time :',end-start)
        # print('predict finished')
        return predict_list


if __name__ == '__main__':
    from load_data import train_features, train_labels, test_id, test_features, split_train_data, evaliate

    from dimension_reduction import PCA_features, dimension_split_train_data
    from pandas import DataFrame

    pca_features = PCA_features(out_number=7, tfeatures=train_features)
    features = DataFrame(pca_features)
    tfs, tls, vfs, vls = dimension_split_train_data(train_features=features, train_labels=train_labels)
    # tfs, tls, vfs, vls = split_train_data()
    i = 0
    mean_acc = 0
    for (features, labels, vfeatures, vlabels) in zip(tfs, tls, vfs, vls):
        print('bag : ', i, ' start')
        bag = Bagging(train_features=features,
                      train_labels=labels,
                      num_of_tree=100)
        bag.create_tree()
        predict = bag.predict(vfeatures)
        accuracy = evaliate(predict, vlabels)
        mean_acc += accuracy
        print('bag ', i, ' :', accuracy)
        i += 1

        # clf=RandomForestClassifier(n_estimators=200)
        # clf.fit(features,labels)
        # predict=clf.predict(vfeatures)
        # accuracy = evaliate(predict, vlabels)
        # mean_acc += accuracy
        #
        # i += 1
        # print('forest ', i, ' :', accuracy)

    print('mean acc : ', mean_acc / 10)



    '''
    # final
    bag = Random_Forest(train_features=train_features,
                        train_labels=train_labels,
                        num_of_tree=20)
    tree_list = bag.create_tree()
    predict = bag.predict(test_features)
    print(predict)
    predict += 1
    data = {'Cover_Type': predict.astype(int).tolist()}
    result = pd.DataFrame(data, columns=['Cover_Type'], index=test_id.astype(int))
    result.to_csv('result.csv')
    print(result)
    '''

