from sklearn.linear_model import Lasso
lasso = Pipeline([
    ('scale', StandardScaler()),
    ('reg', Lasso(alpha=0.06))
])

lasso.fit(X_train, y_train)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
plot_linear_coef(lasso['reg'].coef_, names=X_train.columns, sorted=True, ax=ax1);
plot_linear_coef(ridge['reg'].coef_, names=X_train.columns, sorted=True, ax=ax2);

lasso_cvs = cross_validate(
    lasso, X_train, y_train, return_estimator=True, cv=RepeatedKFold(n_splits=5, n_repeats=5)
)

lasso_coefs = pd.DataFrame(
   [model['reg'].coef_ for model in lasso_cvs['estimator']],
   columns=X.columns
)
fig, ax = plt.subplots()
_ = ax.boxplot(lasso_coefs, vert=False, labels=lasso_coefs.columns)