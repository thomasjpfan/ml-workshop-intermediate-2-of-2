lasso_result = permutation_importance(lasso, X_test, y_test,
                                      n_repeats=15, n_jobs=-1)

_ = plot_permutation_importance(lasso_result, X_test.columns)
