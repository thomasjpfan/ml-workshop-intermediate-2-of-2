inner_gbc = GradientBoostingClassifier(random_state=0)
gbc_cal = CalibratedClassifierCV(inner_gbc, method='isotonic')
gbc_cal.fit(X_train, y_train)


gbc_cal_pred = hist_cal.predict_proba(X_test)

gbc_cal_brier = brier_score_loss(y_test, gbc_cal_pred[:, 1])

gbc_cal_brier

gbc_brier

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
CalibrationDisplay.from_estimator(gbc, X_test, y_test, ax=ax1, n_bins=10)
ax1.set_title(f"Hist without calibration: {gbc_brier:0.4f}")
CalibrationDisplay.from_estimator(gbc_cal, X_test, y_test, ax=ax2, n_bins=10)
ax2.set_title(f"calibrated: {gbc_cal_brier:0.4f}");
