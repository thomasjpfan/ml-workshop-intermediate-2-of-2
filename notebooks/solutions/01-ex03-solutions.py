from sklearn.metrics import roc_auc_score

rf_predict_proba = rf.predict_proba(X_test)
roc_auc_score(y_test, rf_predict_proba[:, 1])

from sklearn.svm import SVC

svc = SVC(random_state=42)
svc.fit(X_train, y_train)

svc_decision = svc.decision_function(X_test)
roc_auc_score(y_test, svc_decision)
