# Titanic Survival Analysis

This repository contains a minimal Titanic dataset, exploratory analysis, and a logistic regression model that forecasts each passenger's survival probability.

## Repository structure

- `analysis/titanic_analysis.py` – exploratory statistics and markdown summary generation.
- `analysis/titanic_model.py` – end-to-end survival model training and prediction script.
- `analysis/titanic_predictions.csv` – sample model output generated from the bundled dataset.
- `notebooks/titanic_survival_model.ipynb` – Jupyter Notebook version of the modeling workflow.
- `data/titanic.csv` – sample Titanic manifest used by the scripts.

## Running the model locally

1. Ensure you have **Python 3.9+** installed.
2. Install (or upgrade) `pip` and run the model:

   ```bash
   python -m pip install --upgrade pip
   python analysis/titanic_model.py
   ```

The script will print accuracy/log-loss metrics and regenerate `analysis/titanic_predictions.csv` with the latest predictions.

## Using the Jupyter Notebook

1. Launch Jupyter (or VS Code / another notebook environment) from the repository root:

   ```bash
   jupyter notebook
   ```

2. Open `notebooks/titanic_survival_model.ipynb`.
3. Execute the cells from top to bottom to walk through loading the dataset, engineering features, training the logistic regression model, and exporting predictions to `analysis/titanic_predictions.csv`.

The notebook mirrors the standalone script so you can tweak hyperparameters or inspect intermediate variables interactively.

## Running the model on GitHub

A GitHub Actions workflow is included to execute the model whenever you push to `main`, open a pull request, or trigger it manually.

1. Commit and push this repository to GitHub.
2. Navigate to **Actions → Run Titanic survival model** in your GitHub project.
3. Click **Run workflow** (or open a pull request / push to `main`).
4. After the workflow finishes, download the `titanic-predictions` artifact to retrieve the generated `analysis/titanic_predictions.csv` file.

The workflow definition lives in `.github/workflows/run-titanic-model.yml`. Feel free to adjust the Python version or extend the steps (for example, to persist additional artifacts or publish results elsewhere).
