from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)
rf_pred = rf.predict(X_test)


r2_score(y_test, rf_pred)

mean_squared_error(y_test, rf_pred)

mean_absolute_error(y_test, rf_pred)
