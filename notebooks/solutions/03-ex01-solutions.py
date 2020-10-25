from sklearn.linear_model import Lasso

lasso = Pipeline([
    ('scale', StandardScaler()),
    ('reg', Lasso(random_state=42, alpha=0.04))
])
lasso.fit(X_train, y_train)

lasso.score(X_test, y_test)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
plot_linear_coef(ridge['reg'].coef_, X_train.columns, ax=ax1)
plot_linear_coef(lasso['reg'].coef_, X_train.columns, ax=ax2)
