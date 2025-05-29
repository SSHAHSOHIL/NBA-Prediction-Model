import os
import papermill as pm
from joblib import load

# Filepaths for your existing notebooks (update paths if needed)
NOTEBOOK_DIR = os.path.dirname(__file__)
NB = {
    'data_collection': os.path.join(NOTEBOOK_DIR, 'data_collection.ipynb'),
    'dump_joblib':     os.path.join(NOTEBOOK_DIR, 'dump_joblib.ipynb'),
    'model_creation':  os.path.join(NOTEBOOK_DIR, 'model_creation.ipynb'),
    'model_spread':    os.path.join(NOTEBOOK_DIR, 'model_creation_spread.ipynb'),
    'player_props':    os.path.join(NOTEBOOK_DIR, 'player_props.ipynb'),
}

# Helper: execute a notebook and delete the output immediately
def execute_and_cleanup(input_path, output_path):
    pm.execute_notebook(input_path, output_path)
    try:
        os.remove(output_path)
    except OSError:
        pass

# 1) Full pipeline to prepare raw data and dump ELOs & momentum
def retrain_data_and_state():
    # runs data_collection and dump_joblib notebooks in sequence
    dc_tmp = os.path.join(NOTEBOOK_DIR, '.tmp_data_collection_out.ipynb')
    execute_and_cleanup(NB['data_collection'], dc_tmp)

    dj_tmp = os.path.join(NOTEBOOK_DIR, '.tmp_dump_joblib_out.ipynb')
    execute_and_cleanup(NB['dump_joblib'], dj_tmp)

# 2) Retrain moneyline win/loss model
def retrain_moneyline_model():
    retrain_data_and_state()
    mc_tmp = os.path.join(NOTEBOOK_DIR, '.tmp_model_creation_out.ipynb')
    execute_and_cleanup(NB['model_creation'], mc_tmp)
    return load(os.path.join(NOTEBOOK_DIR, 'moneyline_model.joblib'))

# 3) Retrain home-team spread model
def retrain_spread_model():
    retrain_data_and_state()
    ms_tmp = os.path.join(NOTEBOOK_DIR, '.tmp_model_spread_out.ipynb')
    execute_and_cleanup(NB['model_spread'], ms_tmp)
    return load(os.path.join(NOTEBOOK_DIR, 'spread_model.joblib'))

# 4) Retrain player props model
def retrain_player_props_model():
    pp_tmp = os.path.join(NOTEBOOK_DIR, '.tmp_player_props_out.ipynb')
    execute_and_cleanup(NB['player_props'], pp_tmp)
    return load(os.path.join(NOTEBOOK_DIR, 'player_props_model.joblib'))

# 5) Helpers to load precalculated state (elos & momentum)
def load_team_elos():
    return load(os.path.join(NOTEBOOK_DIR, 'team_elos.joblib'))

def load_last_mom5():
    return load(os.path.join(NOTEBOOK_DIR, 'last_mom5.joblib'))
