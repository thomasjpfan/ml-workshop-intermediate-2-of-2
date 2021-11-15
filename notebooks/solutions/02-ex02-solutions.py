from sklearn.metrics import log_loss

inner_hist = HistGradientBoostingClassifier(random_state=0)
hist_cal = CalibratedClassifierCV(inner_hist, method='isotonic')
hist_cal.fit(X_train, y_train)

hist_cal_proba = hist_cal.predict_proba(X_test)

hist_cal_brier = brier_score_loss(y_test, hist_proba[:, 1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
CalibrationDisplay.from_estimator(hist, X_test, y_test, ax=ax1, n_bins=10)
ax1.set_title(f"Hist without calibration: {lr_brier:0.4f}")
CalibrationDisplay.from_estimator(hist_cal, X_test, y_test, ax=ax2, n_bins=10)
ax2.set_title(f"calibrated: {cal_lr_brier:0.4f}");

log_loss(y_test, hist_cal_proba[:, 1])

log_loss(y_test, hist_proba[:, 1])
