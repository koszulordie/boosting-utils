# Example gradient boosting with XGBoost
# https://xgboost.readthedocs.io/en/latest/parameter.html

import numpy as np
import xgboost as xgb

random_seed = 42


def custom_eval(alpha=1):
    """alpha: tolerance factor"""
    def rmse(predt, dtrain):
        """ Modified root mean squared log error metric."""
        y = dtrain.get_label()
        elements = np.power(y - predt, 2)
        return 'customrmse', (1e-3 / alpha) * np.sqrt(np.sum(elements) / len(y))
    return rmse


def train(values, XGB_PARAMS, tol_alpha=1, verbose=False):
    """
    Returns:
        XGBoost model fit to data
        learning curve dictionary
    """

    x_train, x_test, y_train, y_test = tuple(values)
    params = XGB_PARAMS.copy()
    params['n_estimators'] = 2000  # set this high enough to allow early-stopping events
    params['base_score'] = y_train.mean()
    params['n_jobs'] = 1
    params['seed'] = random_seed
    model = xgb.XGBRegressor(**params)

    # custom eval function
    rmse = custom_eval(tol_alpha)

    # train with xgboost
    best = model.fit(x_train, y_train,
                     eval_set=[(x_train, y_train), (x_test, y_test)],
                     eval_metric=rmse,
                     callbacks=[xgb.callback.EarlyStopping(rounds=15,
                                                           metric_name='customrmse',
                                                           save_best=True)],
                     verbose=verbose)

    learning_curve = best.evals_result()

    return best, learning_curve


if __name__ == '__main__':
    pass
