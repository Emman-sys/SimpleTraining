# SimpleTraining

A GUI-based machine learning training app built with Gradio.

The app lets you:

- Upload a CSV dataset
- Select a target column
- Choose multiple models in one run
- Train and evaluate all selected models
- Detect overfitting automatically
- Pick the best model using configurable ranking metrics
- Visualize model comparison and task-specific plots
- Export the best model as a .pkl file
- Download a full results table as CSV

## Supported Models

Regression:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors Regressor

Classification:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors Classifier
- Naive Bayes

## Metrics

Regression:

- R2
- MSE

Classification:

- Accuracy
- Precision / Recall / F1 (reported in results)

## Overfitting Detection

A model is flagged as overfitting when:

- Train Score - Test Score > Overfitting Threshold

You can choose to exclude overfitting models from best-model selection.

## Run Locally

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
python app.py
```

4. Open the Gradio URL shown in your terminal.