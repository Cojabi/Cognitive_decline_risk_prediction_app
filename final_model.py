import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap 
from lifelines import CoxPHFitter
from evaluation_functions import plot_auc_over_time, get_max_eval_time_from_censoring,\
    harrell_c_index, time_dependent_auc, plot_aggregate_stratified_risk_in_final_model, compute_ibs_and_brier_curve

from xgboost import XGBRegressor

RANDOM_STATE = None
PARAM_TYPES = {'continuous_params':
                                    ["learning_rate", "gamma", "reg_lambda",
                                    "subsample", "colsample_bytree"],
                'discrete_params':["n_estimators", "max_depth"],
                'categorical_params':[] } # if any (booster, grow_policy, etc)





class FinalModel():

    def __init__(self, save_path=".", final_model=None, mean_predicted_eta=None, coxph_model=None):
        self.save_path = save_path # folder in which stuff gets saved
        self.model = final_model
        self.coxph = coxph_model
        self.mean_predicted_eta = mean_predicted_eta

    def fit(self, X, y, fit_coxph=True, cv_results=None):
        """Fit a final model on all available data."""
        def _get_best_hyperparams(cv_results=None, param_types=PARAM_TYPES):
            """Get best hyperparameters from result dictionary returned by the nested_cv function."""
            best_params_all = pd.DataFrame(
                                    [r["best_params"] for repeat in cv_results.values() for r in repeat]
                                )

            final_params = {}

            # continuous goes median
            for p in param_types['continuous_params']:
                if p in best_params_all.columns:
                    final_params[p] = best_params_all[p].median()

            # discrete goes mode
            for p in param_types['discrete_params'] + param_types['categorical_params']:
                if p in best_params_all.columns:
                    final_params[p] = best_params_all[p].mode().iloc[0]

            # cast ints
            for p in param_types['discrete_params']:
                final_params[p] = int(round(final_params[p]))

            print("Using parameters:", final_params, "to fit the final model.")
            return final_params

        mean_params = _get_best_hyperparams(cv_results=cv_results)

        self.model = XGBRegressor(
            objective="survival:cox",
            enable_categorical=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **mean_params)
        
        self.model.fit(X, y)

        if fit_coxph:
            eta_train = self.model.predict(X).ravel()
            self.mean_predicted_eta = eta_train.mean()

            ## Fit breslow estimator with centered etas
            df_train_eta = pd.DataFrame({
                                            "time": np.abs(y),
                                            "event": (y > 0).astype(bool),
                                            "eta": eta_train - self.mean_predicted_eta # centered
                                        })
            self.coxph = CoxPHFitter()
            self.coxph.fit(df_train_eta, 
                           duration_col="time", 
                           event_col="event", 
                           formula="eta")
    



    #### STUFF BELOW IS NOT ADAPTED FOR CENTERED ETAS YET. OPERATES WITHOUT CENTERING
    def shap_analysis(self, X, save_path=None):
        if save_path is None:
            save_path = self.save_path

        ### Compute SHAP values for the final model ###
        X_float = X.astype(float)
        explainer = shap.TreeExplainer(self.model, X_float)
        shap_values = explainer(X_float)

        print("Computed SHAP values for final model on all data.")

        # --- SHAP Summary Plot ---
        rename_dict = {
                        'ptau217_Age': "Age",
                        'ptau217_harm': "pTau217",
                        'e4_carrier': "APOEε4 status",
                        'nfl_harm': "NFL",
                        'gfap_harm': "GFAP",
                    }

        # Update feature names inside SHAP object
        shap_values.feature_names = [
            rename_dict.get(f, f) for f in shap_values.feature_names
        ]
        plt.figure()
        shap.summary_plot(shap_values, X_float, show=False)

        ax = plt.gca()
        # Increase axis label font size
        ax.set_xlabel(ax.get_xlabel(), fontsize=17)
        ax.set_ylabel(ax.get_ylabel(), fontsize=17)
        # Increase tick label font size
        ax.tick_params(axis='both', which='major', labelsize=16)
        # Increase annotation (feature name) font size
        for text in ax.texts:
            text.set_fontsize(16)
        plt.title("Final Model - SHAP Summary (All Data)", fontsize=16)
        plt.savefig(f"{save_path}/final_shap_summary.png", bbox_inches='tight')
        plt.close()

        # --- SHAP Bar Plot (Mean(|SHAP|)) ---
        plt.figure()
        shap.summary_plot(shap_values, X_float, plot_type="bar", show=False)
        plt.title("Final Model - SHAP Feature Importance (All Data)")
        plt.savefig(f"{save_path}/final_shap_bar.png", bbox_inches='tight')
        plt.close()

        # Save mean absolute SHAP importance as CSV
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        feature_importance_final = pd.DataFrame({
            "feature": X.columns,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)

        feature_importance_final.to_csv(f"{save_path}/final_shap_importance.csv", index=False)

    def external_validation(self,
                            X, 
                            y,
                            y_train,
                            output_prefix="external_validation",
                            max_eval_time_brier=None,
                            n_quantiles=5):
        """
        External validation for a fitted final survival model.
        """

        # --- 3. Predict risk scores ---
        try:
            eta_test = self.model.predict(X).ravel()
        except Exception as e:
            raise RuntimeError(f"Prediction failed on external data: {e}")

        event = (y > 0).astype(bool)
        time = np.abs(y)

        prediction_df = pd.DataFrame(index=X.index)
        prediction_df["Risk_scores"] = eta_test
        prediction_df['Label'] = y

        ### Calibration Stuff
        # Compute Brier + IBS for this fold (make fully supported eval times inside helper)'
        if max_eval_time_brier is None:
            max_eval_time = get_max_eval_time_from_censoring(y_train)
        else:
            max_eval_time = max_eval_time_brier

        brier_eval_times, brier_scores, ibs = compute_ibs_and_brier_curve(
            cox_model=self.coxph,
            y_train=y_train,
            y_test=y,
            risk_scores=eta_test,
            max_eval_time=max_eval_time                        
        )

        # --- Performance metrics ---
        cindex = harrell_c_index(y, eta_test)
        print(f"Harrell's C-index (external): {cindex:.4f}")

        eval_times, aucs, mean_auc = time_dependent_auc(self.model, y_train, X, y)
        print(f"Mean AUC (external): {mean_auc:.4f}")

        auc_results = {
            "eval_times": eval_times.tolist(),
            "aucs": aucs.tolist(),
            "mean_auc": float(mean_auc),
        }

        # --- Stratificaton Plot (using quantiles) ---
        df_ext = pd.DataFrame({"risk": eta_test, "time": time, "event": event})
        if "ptau217_harm" in X.columns:
            df_ext["ptau217"] = X["ptau217_harm"]

        plot_aggregate_stratified_risk_in_final_model(
            [df_ext],
            save_path=f"{self.save_path}/{output_prefix}_calibration.png",
            n_quantiles=n_quantiles,
            show_std_band=False,
            show_folds=False)
        
        if "ptau217" in df_ext.columns:
            plot_ptau_stratified_aggregated(
                [df_ext],
                save_path=f"{self.save_path}/{output_prefix}_ptau_stratification.png",
                n_quantiles=n_quantiles,
                show_std_band=False,
                show_folds=False)

        # --- 7. AUC-over-time Plot ---
        if auc_results["eval_times"] is not None:
            single_results_like = [{"eval_times": auc_results["eval_times"],
                                    "aucs": auc_results["aucs"]}]
            plot_auc_over_time(single_results_like,
                                save_path=f"{self.save_path}/{output_prefix}_auc_over_time.png")

        out = {"Harell's C-Index": float(cindex),
               "IBS":ibs,\
                "Brier_curve": list(brier_scores),
                "Brier_curve_eval_times": list(brier_eval_times),
                "Mean AUC": auc_results.get("mean_auc"),
                "Eval_times": auc_results["eval_times"],
                "AUCs": auc_results["aucs"]}
        

        return out, prediction_df

    def fit_bootstrap_shap(self, X, y, bt_samples=300, 
                           plot_bt_shap=True, shap_sample_size=None, reference_shap_table=True,
                           save_path=None):
        """
        Fit bootstrapped models and compute SHAP values per feature across bootstraps.
        Also computes 95% confidence intervals (2.5% and 97.5% percentiles).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : array-like
            Survival labels
        bt_samples : int
            Number of bootstrap models
        shap_sample_size : int or None
            Optional subsample size for SHAP calculation to speed up computation
        """

        if self.model is None:
            raise ValueError("Final model must be fit before running bootstrap SHAP analysis.")

        X_float = X.astype(float)
        feature_names = X.columns.tolist()
        
        if reference_shap_table:
            if shap_sample_size is None:
                shap_sample_size = X_float.shape[0]
            X_shap_reference = shap.sample(X_float, shap_sample_size, random_state=0) # to estiamte SHAP on the same reference dataset
            X_shap_reference.to_csv(f"{save_path}/X_shap_reference_sample.csv", index=False)

        shap_values_all = []
        interaction_values_all = []
        shap_values_store = []

        print(f"Running {bt_samples} bootstrap models with SHAP analysis...")

        for b in range(bt_samples):
            print(f"Bootstrap {b+1}/{bt_samples}")

            # --- Bootstrap sample ---
            idx = np.random.choice(len(X_float), size=len(X_float), replace=True)
            X_bt = X_float.iloc[idx]
            y_bt = y.iloc[idx]

            # --- Fit model with same hyperparameters ---
            bt_model = XGBRegressor(**self.model.get_params())
            bt_model.fit(X_bt, y_bt)

            # --- SHAP sample selection ---
            if reference_shap_table:
                X_shap = X_shap_reference
            elif shap_sample_size is not None and shap_sample_size < len(X_bt):
                shap_idx = np.random.choice(len(X_bt), shap_sample_size, replace=False)
                X_shap = X_bt.iloc[shap_idx]
            else:
                X_shap = X_bt

            explainer = shap.TreeExplainer(bt_model) #Explainer
            shap_vals = explainer(X_shap).values

            # --- SHAP interaction values ---
            interaction_vals = explainer.shap_interaction_values(X_shap)
            # mean interaction strength per feature pair
            mean_interactions = np.abs(interaction_vals).mean(axis=0)
            interaction_values_all.append(mean_interactions)

            # mean(|SHAP|) per feature for this bootstrap
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            shap_values_all.append(mean_abs_shap)
            shap_values_store.append(shap_vals)

        np.save(
            f"{save_path}/bootstrap_shap_interactions.npy",
            np.array(interaction_values_all)
        )

        np.save(
            f"{save_path}/bootstrap_shap_values.npy",
            np.array(shap_values_store)
        )

        # Save all bootstrapped shap values
        shap_values_all = np.array(shap_values_all)
        pd.DataFrame(shap_values_all, columns=X.columns).to_csv(f"{save_path}/all_bootstrap_shaps.csv")

        # --- Aggregate across bootstraps ---
        shap_mean = shap_values_all.mean(axis=0)
        shap_std = shap_values_all.std(axis=0)
        shap_lower = np.percentile(shap_values_all, 2.5, axis=0)
        shap_upper = np.percentile(shap_values_all, 97.5, axis=0)

        shap_summary_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": shap_mean,
            "std_abs_shap": shap_std,
            "lower_95_ci": shap_lower,
            "upper_95_ci": shap_upper
        }).sort_values("mean_abs_shap", ascending=False)
        # Convert to percent contribution
        total_importance = shap_summary_df["mean_abs_shap"].sum()

        shap_summary_df["percent_contribution"] = (
            shap_summary_df["mean_abs_shap"] / total_importance * 100
        )

        shap_summary_df["percent_lower_95_ci"] = (
            shap_summary_df["lower_95_ci"] / total_importance * 100
        )

        shap_summary_df["percent_upper_95_ci"] = (
            shap_summary_df["upper_95_ci"] / total_importance * 100
        )

        shap_summary_df.to_csv(
            f"{save_path}/bootstrap_shap_importance_with_ci.csv",
            index=False
        )

        print("Saved bootstrap SHAP importance with 95% CI.")

        if plot_bt_shap:
            def _plot_bootstrapped_SHAP(shap_summary_df, save_path=save_path):    
                # --- Bar plot with 95% CI error bars ---
                plt.figure(figsize=(8, 6))

                left_err = shap_summary_df["mean_abs_shap"] - shap_summary_df["lower_95_ci"]
                right_err = shap_summary_df["upper_95_ci"] - shap_summary_df["mean_abs_shap"]

                xerr = np.vstack([
                    np.clip(left_err, 0, None),
                    np.clip(right_err, 0, None)
                ])

                plt.barh(
                    shap_summary_df["feature"],
                    shap_summary_df["mean_abs_shap"],
                    xerr=xerr
                )

                plt.gca().invert_yaxis()
                plt.xlabel("Mean |SHAP value|")
                plt.title("Bootstrap SHAP Feature Importance (95% CI)")
                plt.tight_layout()
                plt.savefig(f"{save_path}/bootstrap_shap_bar_ci.png", dpi=300)
                plt.close()


            try:
                _plot_bootstrapped_SHAP(shap_summary_df, save_path)
            except:
                #DEBUGGING
                bad = shap_summary_df[
                    (shap_summary_df["mean_abs_shap"] < shap_summary_df["lower_95_ci"]) |
                    (shap_summary_df["mean_abs_shap"] > shap_summary_df["upper_95_ci"])
                ]

                print(bad)
                raise 

        return shap_summary_df
