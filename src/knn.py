from pre_processing import *
from sklearn.neighbors import KNeighborsClassifier

# parameter tuned by gridsearchcv
knn_clf = KNeighborsClassifier(algorithm='auto',
                                leaf_size=10,
                                n_neighbors=2,
                                p = 1)
knn_clf.fit(x_train, y_train)
print(knn_clf.score(x_test, y_test))

from sklearn.metrics import classification_report
y_preds = knn_clf.predict(x_test)
print(classification_report(y_test, y_preds))