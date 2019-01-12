from mxnet import gluon
import mxnet as mx
from mxnet import autograd as ag
from mxnet import ndarray as nd
import os

batch_size = 256
n_class = 7
lr = 0.01


# def net_node()

def features_net(features, labels, param=None):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(100))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Activation(activation='relu'))

        net.add(gluon.nn.Dense(100))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Activation(activation='relu'))

        net.add(gluon.nn.Dense(100))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Activation(activation='relu'))

        net.add(gluon.nn.Dense(7))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Activation(activation='relu'))

        net.add(gluon.nn.Dense(n_class))
    net.initialize()

    if os.path.exists(param):
        net.load_parameters(param)
        print('load success')
    else:
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
        for epoch in range(100):
            total_loss = 0.
            data_iter = data_iteration(features, labels)
            for feature, label in data_iter:
                with ag.record():
                    output = net(feature)
                    loss = softmax_cross_entropy(output, label)
                loss.backward()
                trainer.step(batch_size)
                total_loss += nd.mean(loss).asscalar()
        net.save_parameters(param)
        print('save success')
    return net

def inter_output(net, output_layer):
    inter_net = gluon.nn.Sequential()
    with inter_net.name_scope():
        for i in range(output_layer):
            inter_net.add(net[i])
    return inter_net

def net_node(features, labels, param=None):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(100))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Activation(activation='relu'))

        net.add(gluon.nn.Dense(100))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Activation(activation='relu'))

        net.add(gluon.nn.Dense(100))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Activation(activation='relu'))

        net.add(gluon.nn.Dense(n_class))
    net.initialize()

    if os.path.exists(param):
        net.load_parameters(param)
        print('load success')
    else:
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
        for epoch in range(100):
            total_loss = 0.
            data_iter = data_iteration(features, labels)
            for feature, label in data_iter:
                with ag.record():
                    output = net(feature)
                    loss = softmax_cross_entropy(output, label)
                loss.backward()
                trainer.step(batch_size)
                total_loss += nd.mean(loss).asscalar()
        net.save_parameters(param)
        print('save success')

    return net

def train(features, labels, params_path):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(100))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Activation(activation='relu'))

        net.add(gluon.nn.Dense(100))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Activation(activation='relu'))
        # # net.add(gluon.nn.Dropout(0.5))
        #


        net.add(gluon.nn.Dense(100))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Activation(activation='relu'))

        net.add(gluon.nn.Dense(7))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Activation(activation='relu'))

        net.add(gluon.nn.Dense(n_class))
    net.initialize()

    if os.path.exists(params_path):
        net.load_parameters(params_path)
        print('load success')
    else:
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
        for epoch in range(10):
            total_loss = 0.
            data_iter = data_iteration(features, labels)
            for feature, label in data_iter:
                with ag.record():
                    output = net(feature)
                    loss = softmax_cross_entropy(output, label)
                loss.backward()
                trainer.step(batch_size)
                total_loss += nd.mean(loss).asscalar()
            print('Epoch %d loss: %f' % (epoch, total_loss / len(data_iter)))
        net.save_parameters(params_path)
        print('save success')


    return net


def data_iteration(features, labels):
    dataset = gluon.data.ArrayDataset(features, labels)
    data_iter = gluon.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_iter

def predict_label(predict):
    return predict.argmax(axis=1)

def accuracy(y_hat, y):
    return (y_hat == y.astype('float32')).mean().asscalar()


if __name__ == '__main__':
    from load_data import split_train_data,train_features,train_labels,test_features,test_id,evaliate
    import numpy as np
    from decision_tree import sklearn_tree

    tfs, tls, vfs, vls = split_train_data()
    i = 0
    mean_acc = 0
    for (features, labels, vfeatures, vlabels) in zip(tfs, tls, vfs, vls):
        print('net : ', i, ' train')
        features = nd.array(features)
        labels = nd.array(np.array(labels))

        vfeatures = nd.array(np.array(vfeatures))
        vlabels = nd.array(np.array(vlabels))
        net = train(features, labels, './params/nn.params')
        net = inter_output(net, 12)
        features = net(features)

        clf = sklearn_tree(features.asnumpy(), labels.asnumpy())
        vfeatures = net(vfeatures)
        predict = clf.predict(vfeatures.asnumpy())
        print(net)
        accuracy = evaliate(predict, vlabels.asnumpy())
        print('accracy :', accuracy)
        # predict = net(vfeatures)
        # predict = predict_label(predict)
        # i = i+1
        # acc = accuracy(predict, vlabels)
        # mean_acc += acc
        # print('acc : ',acc)
        break
    print('mean acc : ', mean_acc / 10)
    # train_features = nd.array(train_features,ctx=mx.cpu())
    # train_labels = nd.array(train_labels,ctx=mx.cpu())
    # net = train(train_features, train_labels)
    # test_features = nd.array(test_features)
    # print('start predict')
    # predict = net(test_features)
    # predict = predict_label(predict)
    # predict = predict+1
    # print('predict finished')
    # predict = predict.asnumpy()
    # data = {'Cover_Type':predict.astype(int)}
    # result = DataFrame(data, columns=['Cover_Type'],index=test_id.astype(int))
    # print(result)
