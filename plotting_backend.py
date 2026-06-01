import matplotlib.pyplot as plt

def plot_event_probability_with_ci(
    event_df,
    title="Probability of Event Over Time",
    xlabel="Time",
    ylabel="Event Probability",
    plot_population_average=True,
    plot_bt_ci=False,
    breslow_estimator=None,
    vertical_lines_at_x = None,
    legend_label='Event probability',
    plot_reduced_ptau=()
):
    """
    Plot event probability over time with 95% confidence interval.

    Parameters
    ----------
    event_df : pd.DataFrame
        Index: time
        Columns: ['event_probability', 'lower_95', 'upper_95']
    """
    time = event_df.index.values
    prob = event_df["event_probability"].values
    

    fig = plt.figure()
    plt.plot(time, prob, label=legend_label)

    ## Plot CI
    if plot_bt_ci:
        lower = event_df["lower_95"].values
        upper = event_df["upper_95"].values
        plt.fill_between(time, lower, upper, alpha=0.3, label="95% bootstrapped CI")

    # add curves for reduced ptau
    if len(plot_reduced_ptau) > 0:
        for label, risk_df in plot_reduced_ptau.items():
            plt.plot(time, risk_df["event_probability"].values, label=label)


    # plot vertical lines at clinically important times
    if vertical_lines_at_x is not None:
        for t in vertical_lines_at_x:
            plt.axvline(t, color='grey', linestyle=":", alpha=0.4)
    
    if plot_population_average:
        assert type(breslow_estimator) is not None, "Please provide breslow estimator to plot average"
        baseline_survival = breslow_estimator.baseline_survival_
        S0 = baseline_survival.iloc[:, 0]
        population_event_prob = 1 - S0

        plt.plot(
            population_event_prob.index.values,
            population_event_prob.values,
            color="grey",
            linestyle="--",
            linewidth=2,
            label="Average risk in population"
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend()

    return fig