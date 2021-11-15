from sklearn.linear_model import Lasso

lasso = Pipeline([
    ("scalar", StandardScaler()),
    ("reg", Lasso(alpha=0.06)),
])

lasso.fit(X_train, y_train)

plot_linear_coef(lasso["reg"].coef_, names=X_train.columns)

lasso_results = cross_validate(
    lasso, X_train, y_train, cv=RepeatedKFold(n_repeats=5, n_splits=5),
    return_estimator=True
)

lasso_coefs = pd.DataFrame(
    [model['reg'].coef_ for model in lasso_results['estimator']],
    columns=X_train.columns
)

sorted_lasso_coefs = lasso_coefs.mean().argsort()

lasso_coefs.iloc[:, sorted_lasso_coefs].boxplot(vert=False);
