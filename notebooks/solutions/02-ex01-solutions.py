from sklearn.ensemble import HistGradientBoostingClassifier

hist = HistGradientBoostingClassifier(random_state=42)

hist.fit(X_train, y_train)

hist_proba = hist.predict_proba(X_test)

brier_score_loss(y_test, hist_proba[:, 1])

CalibrationDisplay.from_estimator(hist, X_test, y_test, n_bins=10)
