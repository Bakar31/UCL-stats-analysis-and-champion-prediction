from pre_processing import *
from sklearn.ensemble import GradientBoostingClassifier

gbc_clf = GradientBoostingClassifier(learning_rate=0.1,
                                 loss='deviance',
                                 max_depth=2,
                                 min_samples_leaf=5,
                                 min_samples_split=2,
                                 n_estimators=500,
                                 random_state=31)

gbc_clf.fit(x_train, y_train)
print(gbc_clf.score(x_test, y_test))

from sklearn.metrics import classification_report
y_preds = gbc_clf.predict(x_test)
print(classification_report(y_test, y_preds))