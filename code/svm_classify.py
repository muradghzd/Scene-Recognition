import numpy as np
from sklearn import svm


def svm_classify(train_image_feats, train_labels, test_image_feats, kernel_type):
    """
    This function should train a linear SVM for every category (i.e., one vs all)
    and then use the learned linear classifiers to predict the category of every
    test image. Every test feature will be evaluated with all 15 SVMs and the
    most confident SVM will 'win'.

    :param train_image_feats:
        an N x d matrix, where d is the dimensionality of the feature representation.
    :param train_labels:
        an N array, where each entry is a string indicating the ground truth category
        for each training image.
    :param test_image_feats:
        an M x d matrix, where d is the dimensionality of the feature representation.
        You can assume M = N unless you've modified the starter code.
    :param kernel_type:
        the name of a kernel type. 'linear' or 'RBF'.

    :return:
        an M array, where each entry is a string indicating the predicted
        category for each test image.
    """

    categories = np.unique(train_labels)
    score_table = np.zeros((len(test_image_feats),len(categories)))

    for i, cat in enumerate(categories):
        X_labels = (train_labels == cat).astype(int)

        svm_clf = svm.SVC(kernel=kernel_type.lower(), C=2.36)
        svm_clf.fit(train_image_feats, X_labels)
        prob_scores = svm_clf.decision_function(test_image_feats)
        score_table[:,i] = prob_scores

    test_scores = categories[score_table.argmax(1)]

    return test_scores