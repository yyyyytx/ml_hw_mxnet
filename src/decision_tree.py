from sklearn import tree

def sklearn_tree(features, labels):
    clf = tree.DecisionTreeClassifier()
    clf.fit(features, labels)
    return clf

if __name__ == '__main__':
    from load_data import split_train_data,evaliate

    tfs, tls, vfs, vls = split_train_data()
    i = 0
    mean_acc = 0
    for (tfeatures,tlabels,vfeatures,vlabels) in zip(tfs, tls, vfs, vls):
        print('tree : ', i, ' start')
        clf = sklearn_tree(tfeatures, tlabels)
        predict = clf.predict(vfeatures)
        accuracy = evaliate(predict, vlabels)
        mean_acc += accuracy
        i += 1
        print('tree ', i, ' :', accuracy)
    print('mean acc : ', mean_acc / 10)