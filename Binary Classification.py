import numpy as np

y_true = np.array([0, 1, 1, 0, 1, 0])
y_pred = np.array([1, 1, 1, 0, 0, 1])

# TP
TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))

# TN
TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))

# FP
FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))

# FN
FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))

print(TP, TN, FP, FN)


def recall(tp, tn, fp, fn):
    return tp / (tp + fn)


def precision(tp, tn, fp, fn):
    return tp / (tp + fp)


def f1(r, p):
    return 2 * r * p / (r + p)


print("recall: {}, precision: {}, f1-score: {}".format(recall(TP, TN, FP, FN), precision(TP, TN, FP, FN),
                                                       f1(recall(TP, TN, FP, FN), precision(TP, TN, FP, FN))))
