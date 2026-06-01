import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

from sksurv.metrics import concordance_index_censored, integrated_brier_score, brier_score
from lifelines import KaplanMeierFitter, CoxPHFitter

def get_safe_eval_time_grid(y_train, y_test, max_eval_time=None, step_size=0.5):
    """
    Provide fixed time grid for evaluatin AUC and Brier at always the same time points across folds. Runs per current fold.

    Parameters
    ----------
    y_train : array-like
        Training targets (signed times: positive = event, negative = censored).
    y_test : array-like
        Test targets (signed times: positive = event, negative = censored).
    ste_size: the distance between evaluation time points. default half a year (0.5)

    Returns
    -------
    eval_times_trimmed : np.ndarray
        Safe evaluation times for the metric.
    mask : np.ndarray
        Boolean mask for test samples included.
    """
    # reconstruct Surv arrays
    event_train = (y_train > 0).astype(bool)
    time_train = np.abs(y_train)

    event_test = (y_test > 0).astype(bool)
    time_test = np.abs(y_test)

    # find the last censored patient in test set
    last_censor_time = np.min([
        time_test[~event_test].max(),
        time_train[~event_train].max()
    ])

    # restrict test set to only time points in censoring horizon
    mask = time_test <= last_censor_time
    time_test_trimmed = time_test[mask]

    # define min and max times for this fold
    t_min = np.ceil(time_test_trimmed.min()) + 0.1e-8

    if max_eval_time is not None:
        t_max = np.floor(min(time_test_trimmed.max(), max_eval_time)) - 0.1e-8
    else:
        t_max = np.floor(time_test_trimmed.max()) - 0.1e-8
    

    eval_times_trimmed = np.arange(t_min, t_max+step_size, step_size)
    eval_times_trimmed[0] = t_min
    eval_times_trimmed[-1] = t_max
    # print("in time function", t_min, t_max)
    # print(eval_times_trimmed)

    return eval_times_trimmed, mask

def get_safe_eval_times(y_train, y_test, risk_test_full=None, min_points=2, eps=1e-8):
    """
    DEPRECATED BECAUSE OF get_safe_eval_time_grid????????
    Compute fold-safe evaluation times for time-dependent metrics (AUC or Brier). !!!!! This gives different time intervals per fold. 
    I also coded a fixed time grid without np.linspace to make all the eval times the same across folds.

    Parameters
    ----------
    y_train : array-like
        Training targets (signed times: positive = event, negative = censored).
    y_test : array-like
        Test targets (signed times: positive = event, negative = censored).
    risk_test_full : array-like, optional
        Risk scores for test set. Only needed if you want to trim corresponding array.
    min_points : int
        Minimum number of evaluation times required.
    eps : float
        Small epsilon to avoid exact min/max values.

    Returns
    -------
    eval_times_trimmed : np.ndarray
        Safe evaluation times for the metric.
    mask : np.ndarray
        Boolean mask for test samples included.
    """
    # reconstruct Surv arrays
    event_train = (y_train > 0).astype(bool)
    time_train = np.abs(y_train)

    event_test = (y_test > 0).astype(bool)
    time_test = np.abs(y_test)

    # find the last censored patient in test set
    last_censor_time = np.min([
        time_test[~event_test].max(),
        time_train[~event_train].max()
    ])

    # restrict test set
    mask = time_test <= last_censor_time
    time_test_trimmed = time_test[mask]

    # define SAFE eval times
    t_min = time_test_trimmed.min()
    t_max = time_test_trimmed.max()

    eval_times_trimmed = np.unique(
        np.floor(
            time_test_trimmed[
                (time_test_trimmed > t_min + eps) &
                (time_test_trimmed < t_max - eps)
            ]
        )
    ).astype(float)

    # fallback if too few times
    if eval_times_trimmed.size < min_points:
        raise ValueError("Too few safe evaluation times in this fold")

    # final clamp (paranoia)
    eval_times_trimmed = eval_times_trimmed[
        (eval_times_trimmed > t_min + eps) &
        (eval_times_trimmed < t_max - eps)
    ]

    return eval_times_trimmed, mask


def harrell_c_index(y_true, y_pred):
    """Compute Harrell's C-index from ±time label array"""
    event = (y_true > 0).astype(bool)
    time = np.abs(y_true)
    cindex = concordance_index_censored(event, time, y_pred)
    return float(cindex[0])

def time_dependent_auc(estimator, y_train, X_test, y_test, min_points=2, eps=1e-8):
    """
    Compute time-dependent AUC using fold-safe evaluation times. 

    Parameters
    ----------
    estimator : fitted model
        Must implement .predict(X) returning risk scores.
    y_train : array-like
        Training targets (signed times: positive = event, negative = censored).
    X_test : array-like
        Test features.
    y_test : array-like
        Test targets (signed times).
    min_points : int
        Minimum number of evaluation times required.
    eps : float
        Small epsilon to avoid exact min/max values.

    Returns
    -------
    eval_times_trimmed : np.ndarray
        Safe evaluation times.
    aucs : np.ndarray
        AUC values at each evaluation time.
    mean_auc : float
        Integrated mean AUC.
    """
    # Predict risk scores for test set
    risk_test_full = estimator.predict(X_test).ravel()

    # Get safe evaluation times and mask for test samples
    eval_times_trimmed, mask = get_safe_eval_time_grid(
            y_train, y_test)

    # eval_times_trimmed, mask = get_safe_eval_times(
    #     y_train, y_test, min_points=min_points, eps=eps
    # )

    if len(eval_times_trimmed) == 0:
        print("OH NOOO. AUC FAILED!!\n\n")
        return np.array(["No valid eval times"]), np.array([np.nan]), np.nan

    # Reconstruct Surv objects
    event_train, time_train = y_train > 0, np.abs(y_train)
    surv_train = Surv.from_arrays(event_train, time_train)

    event_test, time_test = y_test > 0, np.abs(y_test)
    surv_test_trimmed = Surv.from_arrays(event_test[mask], time_test[mask])
    risk_test_trimmed = risk_test_full[mask]

    # Compute cumulative dynamic AUC
    aucs, mean_auc = cumulative_dynamic_auc(
        surv_train, surv_test_trimmed, risk_test_trimmed, eval_times_trimmed
    )

    # Handle NaN mean_auc when some eval times are invalid
    if np.isnan(mean_auc):
        valid = ~np.isnan(aucs)
        if valid.sum() >= 2:
            mean_auc = (
                np.trapz(aucs[valid], eval_times_trimmed[valid]) /
                (eval_times_trimmed[valid][-1] - eval_times_trimmed[valid][0])
            )
        else:
            mean_auc = np.nan

    return eval_times_trimmed, aucs, mean_auc

def plot_auc_over_time(results, save_path="."):
    """
    Used in FinalModel
    Plot mean AUC over time across folds.
    Expects `results` list from nested_cv_evaluate(), where each element contains
    'eval_times' and 'aucs' lists.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_style("whitegrid")

    # Collect all evaluation times
    all_times = np.unique(np.concatenate([r["eval_times"] for r in results]))
    auc_means, auc_stds = [], []

    for t in all_times:
        vals = [r["aucs"][i] for r in results for i, tt in enumerate(r["eval_times"]) if np.isclose(tt, t)]
        auc_means.append(np.mean(vals))
        auc_stds.append(np.std(vals))

    plt.figure(figsize=(7, 5))
    plt.plot(all_times, auc_means, marker="o", lw=2, label="Mean AUC")
    plt.fill_between(all_times, np.array(auc_means) - np.array(auc_stds),
                     np.array(auc_means) + np.array(auc_stds),
                     alpha=0.2, label="±1 SD")
    plt.title("Time-dependent AUC Across Folds")
    plt.xlabel("Time (years)")
    plt.ylabel("AUC")
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.tight_layout()
    if ".png" in save_path:
        plt.savefig(f'{save_path}', dpi=150)
    else:
        plt.savefig(f'{save_path}/time_dependent_AUC_plot.png', dpi=150)
    plt.close()

def plot_aggregate_stratified_risk_in_final_model(
    all_calibration_data,
    save_path=".",
    n_quantiles=5,
    show_std_band=True,
    show_folds=True,
):
    """
    Aggregate calibration plot across folds using quantile-based risk groups.

    Parameters
    ----------
    all_calibration_data : list of pd.DataFrame
        Each dataframe must have columns ['time', 'event', 'risk'].
    output_path : str
        File path to save the plot.
    n_quantiles : int
        Number of quantile-based groups to split risk scores into (default=10).
    show_std_band : bool
        Whether to show ±1 SD shading around the mean survival curve.
    show_folds : bool
        Whether to overlay individual fold survival curves (faint lines).
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    kmf = KaplanMeierFitter()

    # assign quantile bins for each fold
    all_quantile_dfs = []
    for df in all_calibration_data:
        df = df.copy()
        try:
            df["risk_quantile"] = pd.qcut(df["risk"], n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            # fallback if not enough unique risk values
            df["risk_quantile"] = pd.cut(df["risk"], n_quantiles, labels=False, duplicates="drop")
        all_quantile_dfs.append(df)

    all_quantiles = sorted(set().union(*[df["risk_quantile"].unique() for df in all_quantile_dfs]))

    for q in all_quantiles:
        fold_surv_curves = []
        color = sns.color_palette("tab10", n_quantiles)[int(q % n_quantiles)]

        for df in all_quantile_dfs:
            group = df[df["risk_quantile"] == q]
            if len(group) < 5:
                continue  # skip too-small groups
            kmf.fit(group["time"], event_observed=group["event"])
            surv_df = kmf.survival_function_
            fold_surv_curves.append(surv_df)

        if len(fold_surv_curves) > 0:
            # align on common time grid
            all_times = np.linspace(0, max(s.index.max() for s in fold_surv_curves), 200)
            aligned = []
            for surv in fold_surv_curves:
                aligned_curve = np.interp(all_times, surv.index, surv.iloc[:, 0])
                aligned.append(aligned_curve)
                if show_folds:
                    plt.plot(all_times, aligned_curve, lw=0.8, alpha=0.3, color=color)
            aligned = np.vstack(aligned)
            mean_surv = np.mean(aligned, axis=0)
            std_surv = np.std(aligned, axis=0)

            # ±1 SD band
            if show_std_band:
                plt.fill_between(all_times,
                                 mean_surv - std_surv,
                                 mean_surv + std_surv,
                                 color=color,
                                 alpha=0.15)

            # mean line
            plt.plot(all_times, mean_surv, lw=2, color=color, label=f"Quantile {int(q)+1}")

    plt.title("Aggregate Calibration Plot Across Folds (Quantile-based)")
    plt.xlabel("Time (years)")
    plt.ylabel("Estimated Survival Probability")
    plt.legend(title="Predicted risk quantile", fontsize="small")
    plt.tight_layout()
    if ".png" in save_path:
        plt.savefig(f"{save_path}", dpi=150)
    else:
        plt.savefig(f"{save_path}/calibration_aggregate.png", dpi=300)
    plt.close()
    print(f"Saved quantile-based calibration plot to {save_path}")

def plot_aggregate_stratified_risk_across_repeats_in_cv(
    calibration_data_all,
    save_path=".",
    n_quantiles=5,
    show_std_band=True,
    show_repeats=True,
):
    """
    Aggregate calibration plot across repeats using average survival curves. Used in nested_cv

    Parameters
    ----------
    calibration_data_all : dict
        {repeat: list of pd.DataFrame}, each dataframe has columns ['time','event','risk']
    """

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    kmf = KaplanMeierFitter()

    repeat_quantile_curves = {}

    # ---- Loop over repeats ----
    for repeat, fold_dfs in calibration_data_all.items():

        all_quantile_dfs = []
        for df in fold_dfs:
            df = df.copy()
            try:
                df["risk_quantile"] = pd.qcut(df["eta_test"], n_quantiles,
                                              labels=False, duplicates="drop")
            except ValueError:
                df["risk_quantile"] = pd.cut(df["eta_test"], n_quantiles,
                                             labels=False, duplicates="drop")
            all_quantile_dfs.append(df)

        quantiles = sorted(set().union(*[df["risk_quantile"].unique()
                                         for df in all_quantile_dfs]))

        for q in quantiles:
            fold_surv_curves = []

            for df in all_quantile_dfs:
                group = df[df["risk_quantile"] == q]
                if len(group) < 5:
                    continue

                kmf.fit(group["time"], event_observed=group["event"])
                fold_surv_curves.append(kmf.survival_function_)

            if len(fold_surv_curves) == 0:
                continue

            # align time grid within repeat
            all_times = np.linspace(
                0, max(s.index.max() for s in fold_surv_curves), 200
            )

            aligned = []
            for surv in fold_surv_curves:
                aligned_curve = np.interp(all_times,
                                          surv.index,
                                          surv.iloc[:, 0])
                aligned.append(aligned_curve)

            aligned = np.vstack(aligned)
            mean_curve = np.mean(aligned, axis=0)

            repeat_quantile_curves.setdefault(q, []).append((all_times, mean_curve))

    # ---- Aggregate across repeats ----
    for q, curves in repeat_quantile_curves.items():

        color = sns.color_palette("tab10", n_quantiles)[int(q % n_quantiles)]

        min_max_time = min(c[0].max() for c in curves)
        common_time = np.linspace(0, min_max_time, 200)

        aligned = []
        for times, surv in curves:
            interp_surv = np.interp(common_time, times, surv)
            aligned.append(interp_surv)

            if show_repeats:
                plt.plot(common_time, interp_surv, lw=0.8,
                         alpha=0.3, color=color)

        aligned = np.vstack(aligned)
        mean_surv = aligned.mean(axis=0)
        std_surv = aligned.std(axis=0)

        if show_std_band:
            plt.fill_between(common_time,
                             mean_surv - std_surv,
                             mean_surv + std_surv,
                             color=color,
                             alpha=0.15)

        plt.plot(common_time, mean_surv, lw=2,
                 color=color, label=f"Quantile {int(q)+1}")

    plt.title("Aggregate Calibration Plot Across Repeats (Quantile-based)")
    plt.xlabel("Time (years)")
    plt.ylabel("Estimated Survival Probability")
    plt.legend(title="Predicted risk quantile", fontsize="small")
    plt.tight_layout()

    if ".png" in save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.savefig(f"{save_path}/calibration_aggregate_repeats.png", dpi=300)

    plt.close()
    print(f"Saved calibration plot to {save_path}")


def compute_ibs_and_brier_curve(cox_model, y_train, y_test, risk_scores, max_eval_time=None):
    """
    Robust IBS and Brier curve computation using fold-safe evaluation times.

    Parameters
    ----------
    cox_model : fitted survival model
        Must implement .predict_survival_function(X, times=eval_times).
    y_train : array-like or pd.Series
        Training targets (signed times: positive=event, negative=censored).
    y_test : array-like or pd.Series
        Test targets (signed times).
    risk_scores : array-like
        Risk scores corresponding to test set.
    min_points : int
        Minimum number of evaluation times required.
    eps : float
        Small epsilon to avoid exact min/max values.

    Returns
    -------
    eval_times_trimmed : np.ndarray
        Safe evaluation times.
    brier_scores : np.ndarray
        Brier scores at each evaluation time.
    ibs : float
        Integrated Brier Score.
    """
    # Ensure numpy arrays
    y_train_vals = y_train.values if hasattr(y_train, "values") else np.array(y_train)
    y_test_vals = y_test.values if hasattr(y_test, "values") else np.array(y_test)
    risk_scores = np.array(risk_scores)

    # Reconstruct Surv arrays
    event_train, time_train = y_train_vals > 0, np.abs(y_train_vals)
    event_test, time_test = y_test_vals > 0, np.abs(y_test_vals)
    surv_train = Surv.from_arrays(event_train, time_train)

    # -------------------------------
    # Use fold-safe evaluation times
    # -------------------------------

    eval_times_trimmed, mask = get_safe_eval_time_grid(
            y_train, y_test, max_eval_time=max_eval_time)
    
    # print(eval_times_trimmed.min(), eval_times_trimmed.max())
    # print(np.absolute(y_test).max())
    # print(np.absolute(y_test).min())

    if len(eval_times_trimmed) == 0:
        print("OH NOOO. Brier FAILED!!\n\n")
        return np.array(["No valid eval times"]), np.array([np.nan]), np.nan

    # Trim test set accordingly
    surv_test_trimmed = Surv.from_arrays(event_test[mask], time_test[mask])
    risk_test_trimmed = risk_scores[mask]

    # -------------------------------
    # Predict survival function at safe eval times
    # -------------------------------
    surv_df = cox_model.predict_survival_function(
        pd.DataFrame({"eta": risk_test_trimmed}),
        times=eval_times_trimmed
    )
    surv_preds = surv_df.T.values

    # -------------------------------
    # Compute Brier scores + IBS
    # -------------------------------
    times_bs, brier_scores = brier_score(
        surv_train,
        surv_test_trimmed,
        surv_preds,
        eval_times_trimmed
    )

    ibs = integrated_brier_score(
        surv_train,
        surv_test_trimmed,
        surv_preds,
        eval_times_trimmed
    )

    return eval_times_trimmed, brier_scores, ibs


def plot_brier_curve_cv(all_brier_curves, eval_times, save_path=None):
    """
    Plot aggregated Brier curves.
    
    all_brier_curves: list of arrays, each array is fold-wise OOF Brier scores
    eval_times: array of times corresponding to the curves (all curves must match this length)
    """
    # Ensure curves are arrays of same length
    curves = np.array([np.array(curve) for curve in all_brier_curves])
    
    # Compute mean ± std
    mean_brier_score = curves.mean(axis=0)
    std_brier_score = curves.std(axis=0)

    plt.figure(figsize=(7,5))
    print()
    plt.plot(eval_times, mean_brier_score, lw=2, label="Brier score (OOF)")
    plt.fill_between(
        eval_times,
        mean_brier_score - std_brier_score,
        mean_brier_score + std_brier_score,
        alpha=0.2,
        label="±1 SD (folds × repeats)"
    )

    plt.xlabel("Time")
    plt.ylabel("Brier score")
    plt.title("Aggregated Prediction Error Curve (OOF, folds × repeats)")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.tight_layout()
    plt.close()

def get_max_eval_time_from_censoring(y_train):
    from sksurv.nonparametric import kaplan_meier_estimator

    event_train, time_train = y_train > 0, np.abs(y_train)
    censor_indicator = ~event_train

    times, G = kaplan_meier_estimator(censor_indicator, time_train)

    max_time = times[G > 0.05].max()
    return max_time
