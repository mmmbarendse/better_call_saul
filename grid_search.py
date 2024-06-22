class GridSearch:
    """
    This class performs a grid search over the given model parameters and provides functionality to plot the results.

    Attributes:
        model: The model to perform the grid search on.
        parameters: A dictionary of model parameters for the grid search. Parameters are given as lists. Parameters that
        need to be specified are "N", "deterrence", "gamma_alpha", "gamma_beta", "steps", "distribution", and
        "fraction_stolen".
        results: A list to store the results of the grid search.
        df_results: A DataFrame to store the results of the grid search in a structured format.

    Methods:
        run_search(): Performs the grid search.
        create_boxplot(param, ax, target): Creates a boxplot for a given parameter.
        plot_results(target): Creates a grid of boxplots for all parameters.
    """
    def __init__(self, model, parameters):
        self.model = model
        self.parameters = parameters
        self.results = []

    def run_search(self):
        import itertools
        from distribution import gamma
        import pandas as pd
        from crime_model import CrimeModel

        param_names = list(self.parameters.keys())
        param_combinations = list(itertools.product(*[self.parameters[param] for param in param_names]))

        # loop over all combinations
        for combination in param_combinations:
            print(f"Combination {len(self.results) + 1}/{len(param_combinations)} \n"
                  f"Parameters: {dict(zip(param_names, combination))}")
            # initialize parameters
            param_dict = dict(zip(param_names, combination))
            N = param_dict["N"]
            deterrence = param_dict["deterrence"]
            gamma_alpha = param_dict["gamma_alpha"]
            gamma_beta = param_dict["gamma_beta"]
            steps = param_dict["steps"]
            distribution = param_dict["distribution"]
            fraction_stolen = param_dict["fraction_stolen"]

            # create distribution
            wealth_arr = gamma(gamma_alpha, gamma_beta, N)

            # create model
            model = CrimeModel(N, deterrence, wealth_arr, fraction_stolen)
            while model.schedule.steps < steps:
                model.step()

            # get data
            model_df = model.datacollector.get_model_vars_dataframe()
            agent_df = model.datacollector.get_agent_vars_dataframe()

            # add data to results df
            self.results.append({
                "N": N,
                "deterrence": deterrence,
                "gamma_alpha": gamma_alpha,
                "gamma_beta": gamma_beta,
                "steps": steps,
                "distribution": distribution,
                "fraction_stolen": fraction_stolen,
                "gini_start": model_df['Gini Coefficient'].values[0],
                "gini_end": model_df['Gini Coefficient'].values[-1],
                "crime_rate": model_df['Crime rate'].values[-1],
                "wealth_arr_start": wealth_arr,
                "wealth_arr_end": model.wealth_arr,
                "gini_over_time": model_df['Gini Coefficient'].values.tolist(),
                "crime_rate_over_time": model_df['Crime rate'].values.tolist()
            })

        self.df_results = pd.DataFrame(self.results)

    def create_boxplot(self, target='crime_rate'):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(nrows=len(self.parameters), figsize=(10, len(self.parameters)*5))

        for idx, param in enumerate(self.parameters):
            self.df_results.boxplot(column=target, by=param, ax=axs[idx])
            axs[idx].set_title('')
            axs[idx].set_xlabel(param)
            axs[idx].set_ylabel(target)

        plt.suptitle('')
        plt.tight_layout()
        plt.show()

    def model_plots(self, indices):
        import numpy as np
        import matplotlib.pyplot as plt

        for idx in indices:
            if idx not in self.df_results.index:
                raise KeyError(f"Specified index {idx} does not exist in the DataFrame.")

            # Convert lists back to arrays
            gini_over_time = np.array(self.df_results.loc[idx, 'gini_over_time'])
            crime_rate_over_time = np.array(self.df_results.loc[idx, 'crime_rate_over_time'])
            wealth_arr_start = np.array(self.df_results.loc[idx, 'wealth_arr_start'])
            wealth_arr_end = np.array(self.df_results.loc[idx, 'wealth_arr_end'])

            N = self.df_results.loc[idx, 'N']
            deterrence = self.df_results.loc[idx, 'deterrence']
            gamma_alpha = self.df_results.loc[idx, 'gamma_alpha']
            gamma_beta = self.df_results.loc[idx, 'gamma_beta']
            steps = self.df_results.loc[idx, 'steps']
            distribution = self.df_results.loc[idx, 'distribution']
            fraction_stolen = self.df_results.loc[idx, 'fraction_stolen']

            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs[0, 0].plot(gini_over_time)
            axs[0, 0].set_title("Gini Coefficient over time")

            axs[0, 1].plot(crime_rate_over_time)
            axs[0, 1].set_title("Crime rate over time")

            axs[1, 0].hist(wealth_arr_start, bins=50)
            axs[1, 0].set_title(
                f"Wealth distribution start, Gini coefficient = {self.df_results.loc[idx, 'gini_start']:.2f}")

            axs[1, 1].hist(wealth_arr_end, bins=50)
            axs[1, 1].set_title(
                f"Wealth distribution end, Gini coefficient = {self.df_results.loc[idx, 'gini_end']:.2f}")

            fig.suptitle(f"N: {N} \n"
                         f"deterrence = {deterrence} \n"
                         f"gamma_alpha = {gamma_alpha}\n"
                         f"gamma_beta = {gamma_beta}\n"
                         f"steps = {steps}\n"
                         f"distribution = {distribution}\n"
                         f"fraction_stolen = {fraction_stolen}")

            plt.tight_layout()
            plt.show()

    def view_results(self):
        return self.df_results
