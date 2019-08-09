#!/usr/bin/env python
# coding: utf-8

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

class modelProducer(object):
    def randomTree(self, features, label, RANDOM_STATE=23000, n_estimator=1000):
        # 我們想要用歷史的價格去預測明天的漲跌符號，如果是上漲(標記為+1)，如果是下跌(標記為-1)。<br>
        # 所以這是個**二元分類**問題。我們有許多模型可以使用，linear models/SVM models/tree-based models/KNNs.

        # XY = pd.concat([features, label])
        XY = pd.merge(features, label, left_index=True, right_index=True)

        XY = XY.dropna()
        XY = XY[XY["Y"] != 0] #只選取return有變動的

        timestamps = XY.index

        N = 2800
        n = 100
        # Xs = [x for x in XY.columns if "X" in x] #選取feature的columns

        # X = XY[Xs]
        # y = XY["Y"]
        X = features.loc[timestamps]
        y = label.loc[timestamps]

        X_train = np.array(X[:N])
        y_train = np.array(y[:N])

        X_test = np.array(X[N+n:])
        y_test = np.array(y[N+n:])

        clf = RandomForestClassifier(max_depth=4, n_estimators=n_estimator,
                                    criterion='entropy', random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        pred = clf.predict(X_test)

        return clf, proba, pred, timestamps
