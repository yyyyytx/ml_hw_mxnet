import pandas as pd
import numpy as np
import time

def load_data(file_name):
    data_frame = pd.read_csv(file_name,dtype=np.float)
    ids = data_frame.pop('Id')
    # print(data_frame.columns)
    return data_frame,ids

def load_train_data():
    train_file_path = '../input/train.csv'
    train_data,ids = load_data(train_file_path)
    train_labels = train_data.pop('Cover_Type')
    train_labels = train_labels-1
    train_features = train_data.iloc[:,:10]
    train_features = train_features.assign(Wilderness_Area = np.NaN)
    train_features = train_features.assign(Soil_Type = np.NaN)

    start = time.time()

    for i in np.arange(1,5):
        colum_name = 'Wilderness_Area%d' % (i)
        train_features.loc[train_data[colum_name] == 1, 'Wilderness_Area'] = i

    for i in np.arange(1,41):
        colum_name = 'Soil_Type%d' % (i)
        train_features.loc[train_data[colum_name] == 1, 'Soil_Type'] = i

    end = time.time()
    print('load train data time :',end - start)
    return train_labels,train_data

def load_test_data():
    test_file_path = '../input/test.csv'
    test_data,ids = load_data(test_file_path)
    test_features = test_data.iloc[:,:10]
    test_features = test_features.assign(Wilderness_Area = np.NaN)
    test_features = test_features.assign(Soil_Type = np.NaN)
    start = time.time()
    for i in np.arange(1,5):
        colum_name = 'Wilderness_Area%d' % (i)
        test_features.loc[test_data[colum_name] == 1, 'Wilderness_Area'] = i

    for i in np.arange(1,41):
        colum_name = 'Soil_Type%d' % (i)
        test_features.loc[test_data[colum_name] == 1, 'Soil_Type'] = i
    end = time.time()
    print('Load test data time:',end-start)
    return test_features,ids



train_labels,train_features = load_train_data()

def split_train_data():
    num_of_bin = int(train_features.shape[0]/10)
    tfeatures=[]
    tlabels=[]
    vfeatures=[]
    vlabels=[]
    for i in range(10):
        if i==9:
            vf = train_features.loc[i * num_of_bin:]
            vl = train_labels.loc[i * num_of_bin:]
            tf = train_features.drop(range(i*num_of_bin,train_features.shape[0]))
            tl = train_labels.drop(range(i*num_of_bin,train_features.shape[0]))
        else:
            vf = train_features.loc[i*num_of_bin:(i+1)*num_of_bin]
            vl = train_labels.loc[i*num_of_bin:(i+1)*num_of_bin]
            tf = train_features.drop(range(i * num_of_bin, (i+1)*num_of_bin))
            tl = train_labels.drop(range(i * num_of_bin, (i+1)*num_of_bin))
        tf.index=range(tf.shape[0])
        vf.index=range(vf.shape[0])
        tl.index=range(tl.count())
        vl.index=range(vl.count())
        tfeatures.append(tf)
        tlabels.append(tl)
        vfeatures.append(vf)
        vlabels.append(vl)
    return tfeatures,tlabels,vfeatures,vlabels

test_features,test_id = load_test_data()


def evaliate(predict_labels,true_labels):
    # total_num = true_labels.count()
    total_num=len(true_labels)
    correct_num=np.sum(predict_labels==true_labels)
    return correct_num/total_num

if __name__ == '__main__':
    # from desicion_tree import information_entropy, information_gain
    # print(train_labels.value_counts())
    # print('entropy',information_entropy(train_labels))
    # sub_list = [train_labels[train_labels == 0.0],
    #             train_labels[train_labels == 1.0],
    #             train_labels[train_labels == 2.0],
    #             train_labels[train_labels == 3.0],
    #             train_labels[train_labels == 4.0],
    #             train_labels[train_labels == 5.0],
    #             train_labels[train_labels == 6.0]]
    # print('information gain',information_gain(train_labels,sub_list))
    print(len(train_features.columns))
    # print(test_features.columns)
    # print(train_features.head(10))
    # print(train_labels.head(10))

    # test=pd.DataFrame(test,columns=['1','2','3','4'])
    # print(test.index)
    # print(test)
    # row1=test.head(1)
    # test=test.drop(range(0,2,1))
    # print(test)
    # tfs,tls,vfs,vls=split_train_data()
    # for tf in tfs:
    #     print(tf.index)
    # print(train_labels.count())
    # test1=np.array([1,2,3,4,5,6,7,7,7,7])
    # test2=np.array([1,2,3,4,5,6,7,8,8,8])
    # print(np.sum(test1==test2))
    print(train_features.head(5))
    print(train_labels[:5])
