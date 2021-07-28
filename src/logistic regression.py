from pre_processing import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

log_clf = LogisticRegression(max_iter = 100,
                                 C = 5.428675439323859,
                                 penalty='l1',
                                 solver='liblinear',
                                random_state = 41)
log_clf.fit(x_train, y_train)
print(log_clf.score(x_test, y_test))

from sklearn.metrics import classification_report
y_preds = log_clf.predict(x_test)
print(classification_report(y_test, y_preds))

