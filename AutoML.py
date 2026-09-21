from autogluon.tabular import TabularPredictor
from read_data         import read_wq_taiwan
from sklearn.metrics   import r2_score, mean_squared_error, mean_absolute_percentage_error

import numpy as np
import pandas as pd
import os
import json
import random


random.seed(42)
time     = 20 * 60  # Maximum training time
runs     = 100      # Number of executions
seeds    = [random.randint(1, 1000) for _ in range(runs)]
num_cpus = 8        # Number of CPUs


for run in range(len(seeds)):

    # Read the data
    data = read_wq_taiwan(0.2, seeds[run])

    # Create the training and test DataFrames
    train_data                     = pd.DataFrame(data['X_train'], columns=data['feature_names'])
    train_data[data['targets'][0]] = data['y_train']
    X_test                         = pd.DataFrame(data['X_test'], columns=data['feature_names'])
    y_test                         = data['y_test']

    # Define the target label
    label = data['targets'][0]

    # Create the predictor
    predictor = TabularPredictor(
        label        = label,
        problem_type = data['task'],
        eval_metric  = 'r2',
    )

    predictor.fit(
        train_data,
        time_limit = time,       # Maximum execution time
        num_cpus   = num_cpus,   # Specify the number of CPUs
    )

    # Training summary
    leaderboard = predictor.leaderboard(silent=True)
    summary     = predictor.fit_summary()
    hyperparams = summary['model_hyperparams']

    for model in leaderboard['model']:

        model_path = os.path.join("results", model)
        os.makedirs(model_path, exist_ok=True)

        # Make predictions
        y_pred = predictor.predict(X_test, model=model)

        # Evaluate the model
        r2   = r2_score(y_test, y_pred)                           # Coefficient of determination
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))        # Root mean squared error
        mape = mean_absolute_percentage_error(y_test, y_pred)     # Mean absolute percentage error

        # Calculate feature importance
        feature_importance = predictor.feature_importance(
            data=train_data,
            model=model
        )
        feature_importance_sum = feature_importance['importance'].sum()

        # Normalize feature importance
        feature_importance['normalized_importance'] = (
            feature_importance['importance'] / feature_importance_sum
        )
        feature_importance_dict = feature_importance.to_dict(orient="index")

        # Dictionary containing the execution results
        result = {
            'run': run + 1,
            'seed': seeds[run],
            "Model": model,
            "R2": r2,
            "RMSE": rmse,
            'MAPE': mape,
            "Best Config": hyperparams[model],
            "Feature importance": feature_importance_dict,
            "X_test": X_test.values.tolist(),
            "y_test": list(y_test),
            "y_pred": list(y_pred),
        }

        # Save the results JSON file for each model
        json_path = os.path.join(
            model_path,
            f"result_run_{run + 1}.json"
        )

        with open(json_path, "w") as json_file:
            json.dump(result, json_file, indent=4)

        if model == summary['model_best']:

            model_path = os.path.join("results_model_best")
            os.makedirs(model_path, exist_ok=True)

            # Dictionary containing the execution results
            result = {
                'run': run + 1,
                'seed': seeds[run],
                "Model": model,
                "R2": r2,
                "RMSE": rmse,
                'MAPE': mape,
                "Best Config": hyperparams[model],
                "Feature importance": feature_importance_dict,
                "X_test": X_test.values.tolist(),
                "y_test": list(y_test),
                "y_pred": list(y_pred),
            }

            # Save the results JSON file for the best model
            json_path = os.path.join(
                model_path,
                f"result_run_{run + 1}.json"
            )

            with open(json_path, "w") as json_file:
                json.dump(result, json_file, indent=4)
