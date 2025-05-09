# NBA Prediction Model

This repository contains code for an NBA game outcome prediction model and a moneyline spread prediction model. Both models leverage historical NBA data to provide insights into game results and betting opportunities. A Streamlit frontend allows users to explore past game outcomes, view model-generated win probabilities for hypothetical matchups, and assess the value of moneyline spreads.

## Overview

This project consists of the following key components:

* **Nonlinear Classifier Model:** Predicts the winner of NBA games using a gradient boosted trees algorithm (XGBoost) and TimeSeriesSplit for robust cross-validation.
* **Moneyline Spread Outcome Model:** Predicts the point differential (AWAY\_TEAM score - HOME\_TEAM score) using an XGBoost regressor within a pipeline.
* **Streamlit Frontend:** Provides an interactive web application for users to:
    * View outcomes of past games (last 5 seasons, regular season and playoffs).
    * Obtain win probabilities for user-defined hypothetical games.
    * Analyze the value of home team moneyline spreads based on the model's spread prediction.

Both models are trained using the same set of carefully engineered features.

## Setup and Usage

Follow these steps to set up and run the project:

### 1. Data Collection

* Run the `data_collection.ipynb` notebook. This notebook will process historical NBA data to create the `train_data.csv` file.
* The `train_data.csv` file will contain the features used for training, including:
    * `W_PCT_diff`: Difference in winning percentages between the two teams (based on the previous season).
    * `PTS_RANK_diff`: Difference in points per game rank (based on the previous season).
    * `AST_RANK_diff`: Difference in assists per game rank (based on the previous season).
    * `REB_RANK_diff`: Difference in rebounds per game rank (based on the previous season).
    * `is_home`: Binary indicator (1 if the first team is the home team, 0 otherwise).
    * `elo_diff`: Difference in Elo ratings between the two teams (updated).
    * `momentum_5`: Difference in recent momentum scores (based on the last 5 games, updated).
* **Move `train_data.csv` to the `models` folder.**

### 2. Model Creation

* Run the `model_creation.ipynb` notebook. This notebook trains the nonlinear classifier model (using XGBoost and TimeSeriesSplit for cross-validation) to predict game outcomes.
    * The model achieves a mean cross-validation AUC score of approximately 72.43%.
    * The calibration curve demonstrates good alignment with a perfectly calibrated model.
    * The trained model is saved as `moneyline_model.joblib` in the `models` folder.
* Run the `model_creation_spread.ipynb` notebook. This notebook trains the moneyline spread outcome model (using XGBoost regression within a pipeline) to predict the point differential.
    * The residuals of this model exhibit a roughly normal distribution.
    * The trained spread model is saved as `spread_model.joblib` in the `models` folder.
* Both model pipelines utilize `StandardScaler` for feature scaling.
* Move both joblib files outside of the models folder and in the same directory as `streamlit_app.py`.


### 3. Load Updated Data

* Run the `dump_joblib.ipynb` notebook. This notebook loads the pre-calculated and updated team Elo ratings and momentum scores from `team_elos.joblib` and `last_mom5.joblib`.
* Move both joblib files outside of the models folder and in the same directory as `streamlit_app.py`.

### 4. Run the Streamlit Frontend

* Ensure that `train_data.csv`, `team_elos.joblib`, `last_mom5.joblib`, `spread_model.joblib`, and `moneyline_model.joblib` are located in the same directory as `streamlit_app.py` (i.e., outside the `models` folder).
* Open your terminal or command prompt, navigate to the directory containing `streamlit_app.py`, and run the following command:

    ```bash
    streamlit run streamlit_app.py
    ```

* The Streamlit application will open in your web browser.

## Streamlit Frontend Features

The Streamlit application provides the following functionalities:

* **Past Game Outcomes:** Explore the results of NBA games from the past 5 seasons, with the ability to filter by regular season or playoff games.
* **Model Probabilities:** Input two hypothetical NBA teams, and the model will output the predicted win probabilities for each team.
* **Moneyline Spread Value:** Enter the home team's moneyline spread, and the model will:
    * Predict the home team's spread based on the `model_spread.joblib`.
    * Indicate whether the provided moneyline spread offers potential value based on the model's prediction.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Joblib
* Streamlit

