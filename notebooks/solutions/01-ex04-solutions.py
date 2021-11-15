from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

r2_score(y_test, rf_pred)

mean_squared_error(y_test, rf_pred)

mean_absolute_error(y_test, rf_pred)
