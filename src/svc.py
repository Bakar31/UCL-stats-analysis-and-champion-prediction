from pre_processing import *
from sklearn import svm

svc_clf = svm.SVC(random_state = 7)
svc_clf.fit(x_train, y_train)
print(svc_clf.score(x_test, y_test))

from sklearn.metrics import classification_report
y_preds = svc_clf.predict(x_test)
print(classification_report(y_test, y_preds))