from pre_processing import *
from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression(max_iter = 1000, random_state = 4)
log_clf.fit(x_sm_train, y_sm_train)
print(log_clf.score(x_test, y_test))

from sklearn.metrics import classification_report
y_preds = log_clf.predict(x_test)
print(classification_report(y_test, y_preds))

dev_preds = log_clf.predict(devx)
print(log_clf.score(devx, devy))
print(classification_report(devy, dev_preds))
