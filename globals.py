# XGBoost params
# https://xgboost.readthedocs.io/en/latest/parameter.html

XGB_PARAMS = {
        "objective": "reg:squarederror",
        "reg_lambda": 1,
        "random_state": 42,
        "scale_pos_weight": 1,
        "subsample": 0.7,
        "reg_alpha": 0,
        "max_delta_step": 0,
        "min_child_weight": 1,
        "learning_rate": 1e-05,
        "colsample_bylevel": 1.0,
        "gamma": 0,
        "colsample_bytree": 1.0,
        "booster": "gbtree",
        "max_depth": 4,
        "seed": 21
}
