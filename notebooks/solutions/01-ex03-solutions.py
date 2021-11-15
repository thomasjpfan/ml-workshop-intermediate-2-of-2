from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC

rf_proba = rf.predict_proba(X_test)

roc_auc_score(y_test, rf_proba[:, 1])

svc = SVC(random_state=42)
svc.fit(X_train, y_train)

svc_decision_func = svc.decision_function(X_test)

average_precision_score(y_test, svc_decision_func)
