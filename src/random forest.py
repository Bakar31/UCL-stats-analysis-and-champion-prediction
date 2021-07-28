from pre_processing import *
from sklearn.ensemble import RandomForestClassifier

rand_clf = RandomForestClassifier(n_estimators=1000, random_state = 35)
rand_clf.fit(x_sm_train, y_sm_train)
print(rand_clf.score(x_test, y_test))

from sklearn.metrics import classification_report
y_preds = rand_clf.predict(x_test)
print(classification_report(y_test, y_preds))

dev_preds = rand_clf.predict(devx)
print(rand_clf.score(devx, devy))
print(classification_report(devy, dev_preds))