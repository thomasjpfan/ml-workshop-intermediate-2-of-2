from sklearn.datasets import load_boston

boston = load_boston()

X, y = boston.data, boston.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)

from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(random_state=0)

gb.fit(X_train, y_train)

gb.score(X_train, y_train)

plot_importances(gb.feature_importances_, boston.feature_names)

gb_perm_results = permutation_importance(gb, X_test, y_test, n_repeats=10, n_jobs=-1)

plot_permutation_importance(gb_perm_results, boston.feature_names)

plot_partial_dependence(gb, X_test, features=["LSTAT", "RM", "DIS", "CRIM"],
                        feature_names=boston.feature_names, n_cols=2)

plot_partial_dependence(gb, X_test, features=[('LSTAT', 'RM')],
                        feature_names=boston.feature_names)
