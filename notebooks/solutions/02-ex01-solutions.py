from sklearn.naive_bayes import GaussianNB

nb = GaussianNB().fit(X_train, y_train)

nb_proba = nb.predict_proba(X_test)

brier_score_loss(y_test, nb_proba[:, 1])

plot_calibration_curve(y_test, nb_proba[:, 1], n_bins=10)
