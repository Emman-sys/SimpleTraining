import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


REGRESSION_MODELS: Dict[str, Any] = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(random_state=42, n_estimators=200),
    "Support Vector Regressor (SVR)": SVR(),
    "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
}

CLASSIFICATION_MODELS: Dict[str, Any] = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "Random Forest Classifier": RandomForestClassifier(random_state=42, n_estimators=200),
    "Support Vector Machine (SVM)": SVC(probability=False),
    "K-Nearest Neighbors Classifier": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
}

TMP_DIR = tempfile.mkdtemp(prefix="simpletraining_")


@dataclass
class ModelRunResult:
    model_name: str
    model: Pipeline
    train_score: float
    test_score: float
    mse: Optional[float]
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    cv_score: Optional[float]
    overfit_gap: float
    overfitting: bool
    y_test: np.ndarray
    y_pred: np.ndarray


def infer_task_type(y: pd.Series) -> str:
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        return "classification"

    non_null = y.dropna()
    unique_count = non_null.nunique()
    if unique_count <= 20:
        return "classification"

    unique_ratio = unique_count / max(len(non_null), 1)
    return "regression" if unique_ratio > 0.05 else "classification"


def get_model_choices(task_type: str) -> List[str]:
    if task_type == "regression":
        return list(REGRESSION_MODELS.keys())
    return list(CLASSIFICATION_MODELS.keys())


def get_metric_choices(task_type: str) -> List[str]:
    if task_type == "regression":
        return ["R2", "MSE"]
    return ["Accuracy", "F1"]


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in x.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def maybe_set_n_jobs(estimator: Any, n_jobs: int) -> Any:
    if "n_jobs" in estimator.get_params():
        estimator.set_params(n_jobs=n_jobs)
    return estimator


def train_one_model(
    model_name: str,
    estimator: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    task_type: str,
    overfit_threshold: float,
    cv_folds: int,
    n_jobs: int,
) -> ModelRunResult:
    preprocessor = make_preprocessor(x_train)
    estimator = maybe_set_n_jobs(clone(estimator), n_jobs)
    model = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])

    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    if task_type == "regression":
        train_score = r2_score(y_train, y_train_pred)
        test_score = r2_score(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        accuracy = precision = recall = f1 = None
        cv_metric = "r2"
    else:
        train_score = accuracy_score(y_train, y_train_pred)
        test_score = accuracy_score(y_test, y_test_pred)
        mse = None
        accuracy = test_score
        precision = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
        cv_metric = "accuracy"

    cv_score = None
    if cv_folds >= 2 and len(x_train) >= cv_folds:
        try:
            scores = cross_val_score(model, x_train, y_train, cv=cv_folds, scoring=cv_metric)
            cv_score = float(np.mean(scores))
        except Exception:
            cv_score = None

    gap = train_score - test_score
    overfitting = gap > overfit_threshold

    return ModelRunResult(
        model_name=model_name,
        model=model,
        train_score=float(train_score),
        test_score=float(test_score),
        mse=None if mse is None else float(mse),
        accuracy=None if accuracy is None else float(accuracy),
        precision=None if precision is None else float(precision),
        recall=None if recall is None else float(recall),
        f1=None if f1 is None else float(f1),
        cv_score=cv_score,
        overfit_gap=float(gap),
        overfitting=overfitting,
        y_test=np.asarray(y_test),
        y_pred=np.asarray(y_test_pred),
    )


def build_comparison_plot(results_df: pd.DataFrame, score_col: str):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sorted_df = results_df.sort_values(score_col, ascending=False)
    colors = ["#d9534f" if val else "#2ca02c" for val in sorted_df["Overfitting"]]
    ax.bar(sorted_df["Model"], sorted_df[score_col], color=colors)
    ax.set_title(f"Model Comparison ({score_col})")
    ax.set_ylabel(score_col)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return fig


def build_problem_specific_plot(best_result: ModelRunResult, task_type: str):
    if task_type == "classification":
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            best_result.y_test,
            best_result.y_pred,
            normalize=None,
            cmap="Blues",
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix - {best_result.model_name}")
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(best_result.y_test, best_result.y_pred, alpha=0.65)
    min_v = min(np.min(best_result.y_test), np.min(best_result.y_pred))
    max_v = max(np.max(best_result.y_test), np.max(best_result.y_pred))
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="red")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs Actual - {best_result.model_name}")
    fig.tight_layout()
    return fig


def choose_best_model(
    results: List[ModelRunResult],
    task_type: str,
    selection_metric: str,
    exclude_overfitting: bool,
) -> Tuple[ModelRunResult, str]:
    filtered = [r for r in results if not r.overfitting] if exclude_overfitting else results
    note = ""
    if not filtered:
        filtered = results
        note = "All models were flagged as overfitting, so selection includes them."

    if task_type == "regression":
        if selection_metric == "MSE":
            key_fn = lambda r: -r.mse if r.mse is not None else -np.inf
            best = max(filtered, key=key_fn)
            score_text = f"MSE={best.mse:.5f}"
        else:
            key_fn = lambda r: r.test_score
            best = max(filtered, key=key_fn)
            score_text = f"R2={best.test_score:.5f}"
    else:
        if selection_metric == "F1":
            key_fn = lambda r: r.f1 if r.f1 is not None else -np.inf
            best = max(filtered, key=key_fn)
            score_text = f"F1={best.f1:.5f}"
        else:
            key_fn = lambda r: r.test_score
            best = max(filtered, key=key_fn)
            score_text = f"Accuracy={best.test_score:.5f}"

    return best, score_text + (" | " + note if note else "")


def run_training(
    file_path: str,
    target_col: str,
    task_mode: str,
    selected_models: List[str],
    test_size: float,
    cv_folds: int,
    overfit_threshold: float,
    selection_metric: str,
    exclude_overfitting: bool,
    parallel_training: bool,
    n_jobs: int,
):
    if not file_path:
        raise gr.Error("Please upload a CSV file.")

    if not target_col:
        raise gr.Error("Please select a target column.")

    if not selected_models:
        raise gr.Error("Please select at least one model.")

    df = pd.read_csv(file_path)
    if target_col not in df.columns:
        raise gr.Error("Target column does not exist in the uploaded dataset.")

    df = df.dropna(subset=[target_col]).copy()
    x = df.drop(columns=[target_col])
    y = df[target_col]

    inferred_task = infer_task_type(y)
    task_type = inferred_task if task_mode == "Auto" else task_mode.lower()

    if task_type == "regression":
        models_pool = REGRESSION_MODELS
        stratify = None
    else:
        models_pool = CLASSIFICATION_MODELS
        stratify = y if y.nunique() > 1 else None

    invalid = [m for m in selected_models if m not in models_pool]
    if invalid:
        raise gr.Error(
            "Selected model(s) do not match current task type: " + ", ".join(invalid)
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    jobs = [
        (name, models_pool[name])
        for name in selected_models
    ]

    if parallel_training and len(jobs) > 1:
        max_jobs = min(len(jobs), max(1, n_jobs))
        results = Parallel(n_jobs=max_jobs)(
            delayed(train_one_model)(
                model_name=name,
                estimator=est,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                task_type=task_type,
                overfit_threshold=overfit_threshold,
                cv_folds=cv_folds,
                n_jobs=max(1, n_jobs),
            )
            for name, est in jobs
        )
    else:
        results = [
            train_one_model(
                model_name=name,
                estimator=est,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                task_type=task_type,
                overfit_threshold=overfit_threshold,
                cv_folds=cv_folds,
                n_jobs=max(1, n_jobs),
            )
            for name, est in jobs
        ]

    best, best_score_text = choose_best_model(
        results=results,
        task_type=task_type,
        selection_metric=selection_metric,
        exclude_overfitting=exclude_overfitting,
    )

    rows = []
    for r in results:
        row = {
            "Model": r.model_name,
            "Train Score": round(r.train_score, 5),
            "Test Score": round(r.test_score, 5),
            "MSE": None if r.mse is None else round(r.mse, 5),
            "Accuracy": None if r.accuracy is None else round(r.accuracy, 5),
            "Precision": None if r.precision is None else round(r.precision, 5),
            "Recall": None if r.recall is None else round(r.recall, 5),
            "F1": None if r.f1 is None else round(r.f1, 5),
            "CV Score": None if r.cv_score is None else round(r.cv_score, 5),
            "Overfit Gap": round(r.overfit_gap, 5),
            "Overfitting": r.overfitting,
        }
        rows.append(row)

    results_df = pd.DataFrame(rows)

    if task_type == "regression" and selection_metric == "MSE":
        results_df = results_df.sort_values(["Overfitting", "MSE"], ascending=[True, True])
        compare_col = "MSE"
    elif task_type == "classification" and selection_metric == "F1":
        results_df = results_df.sort_values(["Overfitting", "F1"], ascending=[True, False])
        compare_col = "F1"
    else:
        results_df = results_df.sort_values(["Overfitting", "Test Score"], ascending=[True, False])
        compare_col = "Test Score"

    results_df.insert(0, "Rank", range(1, len(results_df) + 1))

    model_file = os.path.join(TMP_DIR, "best_model.pkl")
    result_file = os.path.join(TMP_DIR, "training_results.csv")
    joblib.dump(best.model, model_file)
    results_df.to_csv(result_file, index=False)

    summary = (
        f"Task Type: {task_type.capitalize()}\n"
        f"Best Model: {best.model_name}\n"
        f"Best Score: {best_score_text}\n"
        f"Overfitting: {'Yes' if best.overfitting else 'No'}"
    )

    comparison_plot = build_comparison_plot(results_df, compare_col)
    specific_plot = build_problem_specific_plot(best, task_type)

    return summary, results_df, comparison_plot, specific_plot, model_file, result_file


def on_file_upload(file_path: str):
    if not file_path:
        return pd.DataFrame(), gr.update(choices=[], value=None), gr.update(choices=[]), gr.update(choices=[])

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise gr.Error(f"Unable to read CSV: {exc}") from exc

    preview = df.head(12)
    target_choices = list(df.columns)
    inferred = infer_task_type(df[df.columns[-1]]) if len(df.columns) > 0 else "classification"
    model_choices = get_model_choices(inferred)
    metric_choices = get_metric_choices(inferred)

    return (
        preview,
        gr.update(choices=target_choices, value=target_choices[-1] if target_choices else None),
        gr.update(choices=model_choices, value=model_choices[:2]),
        gr.update(choices=metric_choices, value=metric_choices[0]),
    )


def on_mode_change(task_mode: str):
    if task_mode == "Auto":
        return gr.update(), gr.update()

    task_type = task_mode.lower()
    model_choices = get_model_choices(task_type)
    metric_choices = get_metric_choices(task_type)
    return (
        gr.update(choices=model_choices, value=model_choices[:2]),
        gr.update(choices=metric_choices, value=metric_choices[0]),
    )


def build_app():
    with gr.Blocks(title="SimpleTraining - Multi-Model Trainer") as app:
        gr.Markdown(
            "## Multi-Model Training and Selection\n"
            "Upload a dataset, pick multiple models, train in one run, detect overfitting, and export the best model."
        )

        with gr.Row():
            file_input = gr.File(label="Upload CSV Dataset", file_types=[".csv"], type="filepath")
            target_col = gr.Dropdown(label="Target Column", choices=[], interactive=True)
            task_mode = gr.Radio(
                label="Task Type",
                choices=["Auto", "Regression", "Classification"],
                value="Auto",
                interactive=True,
            )

        preview_df = gr.Dataframe(label="Dataset Preview", interactive=False)

        with gr.Row():
            model_selector = gr.CheckboxGroup(
                label="Select One or More Models",
                choices=[],
                interactive=True,
            )
            selection_metric = gr.Dropdown(label="Best Model Metric", choices=[], interactive=True)

        with gr.Row():
            test_size = gr.Slider(0.1, 0.5, value=0.2, step=0.05, label="Test Split Ratio")
            cv_folds = gr.Slider(0, 10, value=5, step=1, label="Cross-Validation Folds (0 to disable)")
            overfit_threshold = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="Overfitting Threshold")

        with gr.Row():
            exclude_overfitting = gr.Checkbox(value=True, label="Exclude Overfitting Models from Best Selection")
            parallel_training = gr.Checkbox(value=True, label="Parallel Model Training")
            n_jobs = gr.Slider(1, 16, value=4, step=1, label="Parallel Jobs")

        train_button = gr.Button("Train Selected Models", variant="primary")

        summary_out = gr.Textbox(label="Best Model Summary", lines=4)
        results_table = gr.Dataframe(label="Results Dashboard", interactive=False)

        with gr.Row():
            perf_plot = gr.Plot(label="Model Performance Comparison")
            detail_plot = gr.Plot(label="Task-Specific Visualization")

        with gr.Row():
            best_model_file = gr.File(label="Download Best Model (.pkl)")
            result_csv_file = gr.File(label="Download Results (.csv)")

        file_input.upload(
            on_file_upload,
            inputs=[file_input],
            outputs=[preview_df, target_col, model_selector, selection_metric],
        )

        task_mode.change(
            on_mode_change,
            inputs=[task_mode],
            outputs=[model_selector, selection_metric],
        )

        train_button.click(
            run_training,
            inputs=[
                file_input,
                target_col,
                task_mode,
                model_selector,
                test_size,
                cv_folds,
                overfit_threshold,
                selection_metric,
                exclude_overfitting,
                parallel_training,
                n_jobs,
            ],
            outputs=[
                summary_out,
                results_table,
                perf_plot,
                detail_plot,
                best_model_file,
                result_csv_file,
            ],
        )

    return app


if __name__ == "__main__":
    demo = build_app()
    demo.launch()
