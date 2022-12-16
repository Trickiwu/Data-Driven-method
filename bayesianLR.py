import numpy as np
from sklearn.linear_model import ARDRegression


def bayesian_lr(X: np.ndarray, Y: np.ndarray, threshold: int):
    #
    X_row, X_col = X.shape
    Y_row, Y_col = Y.shape

    judge_Y = ~(Y.sum(axis=0) == np.zeros(Y_col))

    X_blr = np.zeros((Y_col, X_col + 1))
    # sigma_blr = np.zeros((Y_col, X_col))

    for i in range(0, Y_col):
        if judge_Y[i]:
            y = Y[:, i]
            clf = ARDRegression()
            #            clf.n_iter = 500
            clf.threshold_lambda = threshold

            clf.fit(X, y)
            coef = clf.coef_.T
            X_blr[i, :] = np.hstack((coef, clf.intercept_))

    return X_blr

