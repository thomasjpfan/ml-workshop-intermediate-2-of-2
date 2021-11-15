
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

rfc.score(X_test, y_test)

y_pred = rfc.predict(X_test)

print(classification_report(y_test, y_pred))
