from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

fig, ax = plt.subplots(figsize=(12, 8))
RocCurveDisplay.from_estimator(log_reg, X_test, y_test, ax=ax, name="Logistic Regression")
RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax, name="Random Forest")

dummy = DummyClassifier(strategy='prior')
dummy.fit(X_train, y_train)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
RocCurveDisplay.from_estimator(dummy, X_test, y_test, ax=ax1)
PrecisionRecallDisplay.from_estimator(dummy, X_test, y_test, ax=ax2)

log_reg_pred = log_reg.predict(X_test)
rf_pred = rf.predict(X_test)
dummy_pred = dummy.predict(X_test)

f1_score(y_test, log_reg_pred)

f1_score(y_test, rf_pred)

f1_score(y_test, dummy_pred)
