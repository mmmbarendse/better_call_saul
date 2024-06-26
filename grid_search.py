from crime_model import CrimeModel
import pandas as pd
import numpy as np
from tqdm import tqdm

def is_fraction(num) -> bool:
    return (num >= 0) & (num <= 1)

def run_model(params: dict) -> tuple:
    """
    @params needs to contain the following variables:
    - num_steps: number of steps
    - wealth_arr: array of wealth values
    - fraction_stolen: fraction of wealth stolen
    - deterrence: deterrence factor 

    Returns a tuple of two dataframes:
    - df_model: model variables
    - df_agent: agent variables

    WARNING: This function will infer the number of agents from the length of wealth_arr.
    """

    num_steps = params['num_steps']
    assert num_steps > 0, 'num_steps must be a positive integer'

    wealth_arr = params['wealth_arr']
    assert all(isinstance(i, float) or isinstance(i, int) for i in wealth_arr), 'all elements in wealth_arr must be integers or floats'
    assert all(is_fraction(wealth_arr)), 'all elements in wealth_arr must be between 0 and 1'
    N = len(wealth_arr)

    fraction_stolen = params['fraction_stolen']
    assert is_fraction(fraction_stolen), 'fraction_stolen must be between 0 and 1'

    deterrence = params['deterrence']
    assert is_fraction(deterrence), 'deterrence must be between 0 and 1'

    model = CrimeModel(N, deterrence, wealth_arr, fraction_stolen)

    for _ in tqdm(range(num_steps)):
        model.step()

    df_model = model.datacollector.get_model_vars_dataframe()
    df_agent = model.datacollector.get_agent_vars_dataframe()

    return df_model, df_agent 

def run_grid_search(params: dict, verbose = True) -> dict:
    import itertools
    """
    @params_dict needs to contain the following variables:
    - num_steps: list with all number of steps
    - wealth_arr: list with all arrays of wealth values
    - fraction_stolen: list with all fractions of wealth stolen
    - deterrence: list with all deterrence factors
    """

    for key in params.keys():
        assert isinstance(params[key], list) or isinstance(params[key], np.ndarray), f'{key} must be a list'
    
    param_names = list(params.keys())
    param_combinations = list(itertools.product(*params.values()))

    results = []

    for i, combination in enumerate(param_combinations):
        param = {param_names[i]: combination[i] for i in range(len(param_names))}
        
        if verbose:
            # Print parameters for this run
            print(f'Run {i+1}/{len(param_combinations)}')
            for key in param.keys():
                if isinstance(param[key], list) or isinstance(param[key], np.ndarray):
                    print(f'{key}: {len(param[key])} values')
                else:
                    print(f'{key}: {param[key]}')

        # Run model
        df_model, df_agent = run_model(param)

        if verbose:
            print(f'Done.\n')

        results.append({
            "run_id":   i, 
            "param":    param, 
            "df_model": df_model, 
            "df_agent": df_agent
        })

    return results

def parse_results(results : list) -> pd.DataFrame:

    def get_end_gini_coef(df_model):
        return df_model.loc[df_model.index[-1], 'gini_coef']

    def get_end_crime_rate(df_model):
        return df_model.loc[df_model.index[-1], 'crime_rate']

    def parse_param(row):
        for key, value in row['param'].items():
            row[key] = value
        return row

    df_run = pd.DataFrame(results).set_index('run_id')
    
    df_run.loc[:, 'end_gini_coef'] = df_run['df_model'].apply(get_end_gini_coef)
    df_run.loc[:, 'end_crime_rate'] = df_run['df_model'].apply(get_end_crime_rate)
    df_run = df_run.apply(parse_param, axis=1)

    df_run.drop(columns=['df_agent', 'df_model', 'wealth_arr', 'param'], inplace=True)

    return df_run