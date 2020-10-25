from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor

boston = load_boston()
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)

reg = GradientBoostingRegressor(random_state=42)
reg.fit(X_train, y_train)

reg.score(X_test, y_test)

reg_importances = reg.feature_importances_

plot_importances(reg_importances, boston.feature_names)

top_reg_sorted_indices = reg_importances.argsort()[::-1][:4]
top_names = boston.feature_names[top_reg_sorted_indices]
top_names

reg_perm_results = permutation_importance(
    reg, X_test, y_test, n_repeats=30, n_jobs=-1)

_ = plot_permutation_importance(reg_perm_results, boston.feature_names)

top_perm_sorted_indices = reg_perm_results.importances_mean.argsort()[::-1][:4]
top_perm_names = boston.feature_names[top_perm_sorted_indices]

plot_partial_dependence(
    reg, X_test, features=top_perm_names,
    feature_names=boston.feature_names, n_cols=2)

plot_partial_dependence(
    reg, X_test, features=[('LSTAT', 'RM')],
    feature_names=boston.feature_names,
)
