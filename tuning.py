"""
Optuna hyperparameter tuning for CatBoost tabular models.

Usage
-----
from tuning import run_optuna_study

best_params, study = run_optuna_study(
    X, y, groups,
    seed=42,
    n_trials=30,
    n_folds=3,
    storage_path='data/optuna.db',
    study_name='my_study',
)
"""

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor, Pool


def run_optuna_study(
    X,
    y,
    groups,
    *,
    seed: int = 42,
    n_trials: int = 30,
    n_folds: int = 3,
    storage_path: str | None = None,
    study_name: str = "tabular_catboost",
    show_progress: bool = True,
):
    """
    Run an Optuna TPE study optimising CatBoost hyperparameters via GroupKFold CV.

    Parameters
    ----------
    X, y, groups : array-like
        Features, target values, and fold-group labels.
    seed : int
        Random seed for CatBoost and Optuna sampler.
    n_trials : int
        Number of Optuna trials.
    n_folds : int
        Number of CV folds inside each trial.
    storage_path : str or None
        SQLite file path for persistent study storage, e.g. 'data/optuna.db'.
        Pass None to use in-memory storage (not resumable).
    study_name : str
        Name of the Optuna study (used for storage key).
    show_progress : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    best_params : dict
        Best hyperparameters found by the study.
    study : optuna.Study
        The completed Optuna study object.
    """
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Install optuna:  pip install optuna") from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    gkf = GroupKFold(n_splits=n_folds)

    def objective(trial):
        params = dict(
            depth=trial.suggest_int("depth", 4, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            iterations=2000,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=seed,
            od_type="Iter",
            od_wait=80,
            verbose=False,
            allow_writing_files=False,
        )
        fold_r2s = []
        for tr_idx, va_idx in gkf.split(X, y, groups):
            model = CatBoostRegressor(**params)
            model.fit(
                Pool(X.iloc[tr_idx], y.iloc[tr_idx]),
                eval_set=Pool(X.iloc[va_idx], y.iloc[va_idx]),
            )
            preds = model.predict(X.iloc[va_idx])
            fold_r2s.append(r2_score(y.iloc[va_idx], preds))
        return float(np.mean(fold_r2s))

    storage_url = (
        f"sqlite:///{storage_path}" if storage_path is not None else None
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage_url,
        study_name=study_name,
        load_if_exists=True,
    )

    callbacks = []
    if show_progress:
        try:
            from tqdm.auto import tqdm as _tqdm

            pbar = _tqdm(total=n_trials, desc=f"Optuna [{study_name}]", unit="trial")

            def _cb(study, trial):
                best = study.best_value if study.best_value is not None else 0.0
                pbar.set_postfix({"best_r2": f"{best:.4f}", "trial": trial.number + 1})
                pbar.update(1)

            callbacks.append(_cb)
        except ImportError:
            pass

    study.optimize(objective, n_trials=n_trials, callbacks=callbacks)

    if show_progress and callbacks:
        pbar.close()

    return study.best_params, study
