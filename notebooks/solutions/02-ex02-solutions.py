nb = GaussianNB()
nb.fit(X_train, y_train)
nb_proba = nb.predict_proba(X_test)

nb_brier = brier_score_loss(y_test, nb_proba[:, 1])

cal_nb = CalibratedClassifierCV(GaussianNB(), method='isotonic')
cal_nb.fit(X_train, y_train)

cal_nb_prob = cal_nb.predict_proba(X_test)

cal_nb_brier = brier_score_loss(y_test, cal_nb_prob[:, 1])

fig, (ax1, ax2) = plt.subplots(1, 2)
plot_calibration_curve(y_test, nb_proba[:, 1], ax=ax1, n_bins=10)
ax1.set_title(f"no calibration: {nb_brier:0.4f}")
plot_calibration_curve(y_test, cal_lr_proba[:, 1], ax=ax2, n_bins=10)
ax2.set_title(f"isotonic: {cal_nb_brier:0.4f}");
