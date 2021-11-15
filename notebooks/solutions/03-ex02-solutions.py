lasso_perm_results = permutation_importance(lasso, X_test, y_test, n_repeats=5, n_jobs=-1)

_ = plot_permutation_importance(lasso_perm_results, X_test.columns)
