from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor

dataset = fetch_openml(data_id=531, as_frame=True)

X, y = dataset.data, dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

hist_reg = HistGradientBoostingRegressor(random_state=42)
hist_reg.fit(X_train, y_train)

hist_reg.score(X_test, y_test)

hist_perm_results = permutation_importance(
   hist_reg, X_test, y_test, n_repeats=5
)

plot_permutation_importance(hist_perm_results, names=X_train.columns);

hist_top_features_idx = hist_perm_results['importances_mean'].argsort()[-4:]

hist_top_features = X_train.columns[hist_top_features_idx][::-1]

fig, ax = plt.subplots(figsize=(20, 6))
PartialDependenceDisplay.from_estimator(
    hist_reg, X_test, hist_top_features, n_cols=4, ax=ax
);
