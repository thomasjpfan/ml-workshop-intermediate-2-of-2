from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
cal_nb = CalibratedClassifierCV(nb, method="isotonic")
cal_nb.fit(X_train, y_train)

cal_nb_proba = cal_nb.predict_proba(X_test)

cal_nb_brier = brier_score_loss(y_test, cal_nb_proba[:, 1])

fig, (ax1, ax2) = plt.subplots(1, 2)
plot_calibration_curve(y_test, nb_proba[:, 1], ax=ax1, n_bins=10)
ax1.set_title(f"no calibration: {nb_brier:0.4f}")
plot_calibration_curve(y_test, cal_nb_proba[:, 1], ax=ax2, n_bins=10)
ax2.set_title(f"calibrated: {cal_nb_brier:0.4f}");
