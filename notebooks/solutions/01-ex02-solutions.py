fig, ax = plt.subplots()
plot_roc_curve(log_reg, X_test, y_test, ax=ax, name="LogisticRegression")
plot_roc_curve(rf, X_test, y_test, ax=ax)

from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='prior')
dummy.fit(X_train, y_train)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
plot_precision_recall_curve(dummy, X_test, y_test, ax=ax1)
plot_roc_curve(dummy, X_test, y_test, ax=ax2)

dummy_pred = dummy.predict(X_test)

from sklearn.metrics import f1_score

f1_score(y_test, dummy.predict(X_test))

f1_score(y_test, log_reg.predict(X_test))

f1_score(y_test, rf.predict(X_test))
