import numpy as np
import pandas as pd
import joblib

class XGB_Model_Backend():

    def __init__(self, model_path, breslow_path=None, mean_predicted_eta_path=None, bootstrap_dir=None):
        
        self.load_model(model_path)

        # Only used when paths are provided. Defaults to the breslow estimator stored in object under model_path
        if breslow_path is not None:
            self.load_breslow_estimator(breslow_path)
        if mean_predicted_eta_path is not None:
            self.load_mean_predicted_eta(mean_predicted_eta_path)

        # optionally load bootstrap models for CI
        self.bootstrap_models = []
        if bootstrap_dir is not None:
            import os
            for f in sorted(os.listdir(bootstrap_dir)):
                if f.endswith(".joblib"):
                    self.bootstrap_models.append(
                        joblib.load(os.path.join(bootstrap_dir, f))
                    )

    def load_mean_predicted_eta(self, path):
        self.mean_predicted_eta = joblib.load(path)

    def load_model(self, path):
        final_model_obj = joblib.load(path)
        self.model = final_model_obj.model

        if final_model_obj.coxph is not None:
            self.breslow_estimator = final_model_obj.coxph
        if final_model_obj.mean_predicted_eta is not None:
            self.mean_predicted_eta = final_model_obj.mean_predicted_eta

    
    def load_breslow_estimator(self, path):
        # load fitted breslow estimator for retrieving the S0 function
        self.breslow_estimator = joblib.load(path)
        # get baseline survival function to be scaled later
        self.baseline_survival = self.breslow_estimator.baseline_survival_

    def predict_for_single_patient(self, x_new, timeline, return_CI=True, CI_alpha=0.05):
        """Predict the cumulative event probability at time t 
        for a new covariate vector x (patient data)"""

        # predict event probability
        eta_new_centered = self.model.predict(pd.DataFrame(x_new))[0] - self.mean_predicted_eta
        survival_prob = self.breslow_estimator.predict_survival_function(
                                            pd.DataFrame({"eta": [eta_new_centered]}),
                                            times=timeline
                                        )
        
        event_prob_new = (1 - survival_prob)[0] # output is DF by default. index with 0 for pat vector

        df = pd.DataFrame(
            {"event_probability": event_prob_new},
            index=timeline
        )

        if return_CI and self.bootstrap_models:
            boot_probs = []

            for b in self.bootstrap_models:

                eta_b_centered = b["xgb_model"].predict(pd.DataFrame(x_new))[0] - b["mean_predicted_eta"] # centered prediction
                survival_prob_b = b["breslow_estimator"].predict_survival_function(
                                                            pd.DataFrame({"eta": [eta_b_centered]}),
                                                            times=timeline
                                                        )
                boot_probs.append(1 - survival_prob_b.iloc[:, 0].values)
            
            boot_probs = np.vstack(boot_probs)
            df["lower_95"] = np.quantile(boot_probs, CI_alpha / 2, axis=0)
            df["upper_95"] = np.quantile(boot_probs, 1 - CI_alpha / 2, axis=0)
        else:
            df["lower_95"] = np.zeros(len(df))
            df["upper_95"] = np.zeros(len(df))

        return df
    


    


