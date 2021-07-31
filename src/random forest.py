from pre_processing import *
from sklearn.ensemble import RandomForestClassifier

# parameters were taken by randomizedsearchcv
rand_clf = RandomForestClassifier(n_estimators=1000,
                                 min_samples_split = 4,
                                 min_samples_leaf = 1,
                                 max_depth = None,
                                 random_state = 35)
rand_clf.fit(x_train, y_train)
print(rand_clf.score(x_test, y_test))

from sklearn.metrics import classification_report
y_preds = rand_clf.predict(x_test)
print(classification_report(y_test, y_preds))