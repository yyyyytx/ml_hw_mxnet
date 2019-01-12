from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from load_data import train_labels, train_features

# def PCA_reduction(out_number, tfeatures, vfeatures):
#     pca = PCA(n_components=out_number)
#     pca = pca.fit(tfeatures)
#     tf = pca.transform(tfeatures)
#     vf = pca.transform(vfeatures)
#     # tf = pca.fit_transform(tfeatures)        mean_acc = 0        mean_acc = 0


#     # vf = pca.fit_transform(vfeatures)
#     return tf, vf
#
#
# def TSNE_reduction(out_number, tfeatures, vfeatures):
#     tsne = TSNE(n_components=out_number)
#     tf = tsne.fit_transform(tfeatures)
#     vf = tsne.fit_transform(vfeatures)
#     return tf,vf

def PCA_features(out_number, tfeatures):
    pca = PCA(n_components=out_number)
    tf = pca.fit_transform(tfeatures)
    return tf

def TSNE_features(out_number, tfeatures):
    tsne = TSNE(n_components=out_num, init='pca')
    tf = tsne.fit_transform(tfeatures)
    return tf

def LDA_features(out_number, tfeatures, tlabels):
    lda = LinearDiscriminantAnalysis(n_components=out_number)
    lda = lda.fit(tfeatures, tlabels)
    return lda.transform(tfeatures)

def dimension_split_train_data(train_features,train_labels):
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


if __name__ == '__main__':
    from load_data import split_train_data, evaliate,train_features,train_labels
    from decision_tree import sklearn_tree
    from pandas import DataFrame
    from sklearn.model_selection import train_test_split


    for out_num in range(12):
        pca_features = PCA_features(out_number=out_num+1, tfeatures=train_features)
        features = DataFrame(pca_features)



        # train_test_split(pca_features,train_labels,test_size=0.3)

        tfs, tls, vfs, vls = dimension_split_train_data(train_features=features, train_labels=train_labels)
        i = 0
        mean_acc = 0
        for (tfeatures, tlabels, vfeatures, vlabels) in zip(tfs, tls, vfs, vls):
            clf = sklearn_tree(tfeatures, tlabels)
            predict = clf.predict(vfeatures)
            accuracy = evaliate(predict, vlabels)
            mean_acc += accuracy
            i += 1
        print('PCA : out features : %d number mean acc : %.4f'%(out_num+1, mean_acc / 10))

    for out_num in range(12):
        # tsne_features = TSNE_features(out_number=(out_num+1), tfeatures=train_features)
        lda_features = LDA_features(out_number=(out_num+1),tfeatures=train_features,tlabels=train_labels)
        features = DataFrame(lda_features)
        tfs, tls, vfs, vls = dimension_split_train_data(train_features=features, train_labels=train_labels)
        i = 0
        mean_acc = 0
        for (tfeatures, tlabels, vfeatures, vlabels) in zip(tfs, tls, vfs, vls):
            clf = sklearn_tree(tfeatures, tlabels)
            predict = clf.predict(vfeatures)
            accuracy = evaliate(predict, vlabels)
            mean_acc += accuracy
            i += 1
        print('lda : out features : %d number mean acc : %.4f'%(out_num+1, mean_acc / 10))

