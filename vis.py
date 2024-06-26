import matplotlib.pyplot as plt
import pandas as pd

def plot_run(results : list):
    fig, ax = plt.subplots(ncols = 2, figsize=[16, 8])

    for run_data in results:
        run_id = run_data['run_id']
        df_model = run_data['df_model']
        ax[0].plot(df_model.index, df_model['gini_coef'], label=f'Run {run_id}')
        ax[1].plot(df_model.index, df_model['crime_rate'], label=f'Run {run_id}')

    ax[0].set_title('Gini Coefficient')
    ax[1].set_title('Crime Rate')
    ax[0].set_xlabel('Iteration')
    ax[1].set_xlabel('Iteration')
    fig.tight_layout()

    if len(results) < 10:
        ax[0].legend()
        ax[1].legend()

def plot_end_crime_rate(df_run : pd.DataFrame):
    df_run.plot(kind='scatter', x='fraction_stolen', y='deterrence', c='end_crime_rate', colormap='Blues')

def plot_end_gini(df_run : pd.DataFrame):
    df_run.plot(kind='scatter', x='fraction_stolen', y='deterrence', c='end_gini', colormap='Blues')
