
fig, ax = plt.subplots()
plot_roc_curve(log_reg, X_test, y_test, ax=ax, name="Logistic Regression")
plot_roc_curve(rf, X_test, y_test, ax=ax, name="Random Forest")

from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="prior")
dummy.fit(X_train, y_train)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
plot_precision_recall_curve(dummy, X_test, y_test, ax=ax1)
plot_roc_curve(dummy, X_test, y_test, ax=ax2)

from sklearn.metrics import f1_score

y_pred_rf = rf.predict(X_test)
y_pred_log_reg = log_reg.predict(X_test)
y_pred_dummy = dummy.predict(X_test)

f1_score(y_test, y_pred_rf)

f1_score(y_test, y_pred_log_reg)

f1_score(y_test, y_pred_dummy)
