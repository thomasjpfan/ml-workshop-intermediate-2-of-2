from sklearn.metrics import log_loss

inner_hist = HistGradientBoostingClassifier(random_state=0)
hist_cal = CalibratedClassifierCV(inner_hist, method='isotonic', cv=3)
hist_cal.fit(X_train, y_train)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
CalibrationDisplay.from_estimator(hist, X_test, y_test, ax=ax1, n_bins=10)
ax1.set_title("Hist without calibration")
CalibrationDisplay.from_estimator(hist_cal, X_test, y_test, ax=ax2, n_bins=10)
ax2.set_title("calibrated");
