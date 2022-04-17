gbc = GradientBoostingClassifier(random_state=42)

gbc.fit(X_train, y_train)

gbc_proba = gbc.predict_proba(X_test)

gbc_brier = brier_score_loss(y_test, gbc_proba[:, 1])

CalibrationDisplay.from_estimator(gbc, X_test, y_test, n_bins=10)
