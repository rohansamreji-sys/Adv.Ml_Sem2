import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SequentialFeatureSelector
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

st.set_page_config(
    page_title="Lab 3 - Feature Selection",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Advanced Machine Learning - Lab 3")
st.subheader("Feature Selection and Feature Reduction")
st.caption("MAI511-2 | I MSc AIML")

# ------------------------------------------------------------

# DATA LOADING
# ------------------------------------------------------------

@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Diabetes - Linear Regression":
        data = load_diabetes(as_frame=True)
        X = data.data.copy()
        y = data.target.copy()
        problem = "Regression"
    else:
        data = load_breast_cancer(as_frame=True)
        X = data.data.copy()
        y = data.target.copy()
        problem = "Classification"

    return X, y, problem


# ------------------------------------------------------------
# FEATURE SELECTION FUNCTIONS
# ------------------------------------------------------------

def low_variance_features(X_train, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_train)
    return X_train.columns[selector.get_support()].tolist()


def high_correlation_features(X_train, threshold=0.90):
    corr = X_train.corr().abs()
    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )

    remove = [
        column
        for column in upper.columns
        if any(upper[column] > threshold)
    ]

    return [column for column in X_train.columns if column not in remove]


def forward_features(X_train, y_train, problem, n_features):
    if problem == "Regression":
        estimator = LinearRegression()
        scoring = "r2"
    else:
        estimator = LogisticRegression(max_iter=5000)
        scoring = "accuracy"

    selector = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_features,
        direction="forward",
        scoring=scoring,
        cv=5,
        n_jobs=-1
    )

    selector.fit(X_train, y_train)

    return X_train.columns[selector.get_support()].tolist()


def backward_features(X_train, y_train, problem, n_features):
    if problem == "Regression":
        estimator = LinearRegression()
        scoring = "r2"
    else:
        estimator = LogisticRegression(max_iter=5000)
        scoring = "accuracy"

    selector = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_features,
        direction="backward",
        scoring=scoring,
        cv=5,
        n_jobs=-1
    )

    selector.fit(X_train, y_train)

    return X_train.columns[selector.get_support()].tolist()


# ------------------------------------------------------------
# GRADIENT DESCENT
# ------------------------------------------------------------

def linear_gradient_descent(X, y, learning_rate=0.005, iterations=2000):
    X_array = np.asarray(X)
    y_array = np.asarray(y)

    X_bias = np.c_[
        np.ones(X_array.shape[0]),
        X_array
    ]

    weights = np.zeros(X_bias.shape[1])
    loss_history = []

    for _ in range(iterations):
        predictions = X_bias @ weights
        error = predictions - y_array

        gradient = (
            X_bias.T @ error
        ) / len(y_array)

        weights -= learning_rate * gradient

        mse = np.mean(error ** 2)
        loss_history.append(mse)

    return weights, loss_history


def sigmoid(z):
    return 1 / (
        1 + np.exp(-np.clip(z, -500, 500))
    )


def logistic_gradient_descent(
    X,
    y,
    learning_rate=0.01,
    iterations=1000
):
    X_array = np.asarray(X)
    y_array = np.asarray(y)

    X_bias = np.c_[
        np.ones(X_array.shape[0]),
        X_array
    ]

    weights = np.zeros(X_bias.shape[1])
    loss_history = []

    for _ in range(iterations):
        linear_output = X_bias @ weights
        probabilities = sigmoid(linear_output)

        error = probabilities - y_array

        gradient = (
            X_bias.T @ error
        ) / len(y_array)

        weights -= learning_rate * gradient

        probabilities_safe = np.clip(
            probabilities,
            1e-10,
            1 - 1e-10
        )

        loss = -np.mean(
            y_array * np.log(probabilities_safe)
            + (1 - y_array)
            * np.log(1 - probabilities_safe)
        )

        loss_history.append(loss)

    return weights, loss_history


def gd_predict(X, weights, classification=False):
    X_array = np.asarray(X)
    X_bias = np.c_[
        np.ones(X_array.shape[0]),
        X_array
    ]

    output = X_bias @ weights

    if classification:
        return (sigmoid(output) >= 0.5).astype(int)

    return output


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    [
        "Diabetes - Linear Regression",
        "Breast Cancer - Logistic Regression"
    ]
)

X, y, problem = load_data(dataset_name)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y if problem == "Classification" else None
)

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


# ------------------------------------------------------------
# DATASET OVERVIEW
# ------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Dataset",
    "🔎 Feature Selection",
    "📈 Model Evaluation",
    "⚙️ Optimization",
    "🧠 Self-Learning"
])

with tab1:

    st.header("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Instances", X.shape[0])
    c2.metric("Original Features", X.shape[1])
    c3.metric("Training Instances", X_train.shape[0])
    c4.metric("Testing Instances", X_test.shape[0])

    st.write("### Dataset Information")

    info_df = pd.DataFrame({
        "Property": [
            "Problem Type",
            "Instances",
            "Features",
            "Missing Values"
        ],
        "Value": [
            problem,
            X.shape[0],
            X.shape[1],
            int(X.isnull().sum().sum())
        ]
    })

    st.dataframe(
        info_df,
        use_container_width=True,
        hide_index=True
    )

    st.write("### First Five Records")
    st.dataframe(
        X.head(),
        use_container_width=True
    )

    st.write("### Target Distribution")

    fig, ax = plt.subplots()

    if problem == "Classification":
        y.value_counts().sort_index().plot(
            kind="bar",
            ax=ax
        )
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title("Target Class Distribution")
    else:
        y.plot(
            kind="hist",
            bins=20,
            ax=ax
        )
        ax.set_xlabel("Target Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Target Distribution")

    st.pyplot(fig)


# ------------------------------------------------------------
# FEATURE SELECTION
# ------------------------------------------------------------

with tab2:

    st.header("Feature Selection and Feature Reduction")

    if problem == "Regression":
        forward_n = 5
        backward_n = 5
        factor_n = 5
    else:
        forward_n = 10
        backward_n = 10
        factor_n = 10

    low_var = low_variance_features(
        X_train_scaled
    )

    high_corr = high_correlation_features(
        X_train_scaled
    )

    forward = forward_features(
        X_train_scaled,
        y_train,
        problem,
        forward_n
    )

    backward = backward_features(
        X_train_scaled,
        y_train,
        problem,
        backward_n
    )

    factor_analysis = FactorAnalysis(
        n_components=factor_n,
        random_state=42
    )

    X_train_factor = factor_analysis.fit_transform(
        X_train_scaled
    )

    X_test_factor = factor_analysis.transform(
        X_test_scaled
    )

    selection_summary = pd.DataFrame({
        "Method": [
            "Original",
            "Low Variance",
            "High Correlation",
            "Forward Selection",
            "Backward Elimination",
            "Factor Analysis"
        ],
        "Features / Factors": [
            X.shape[1],
            len(low_var),
            len(high_corr),
            len(forward),
            len(backward),
            factor_n
        ]
    })

    st.dataframe(
        selection_summary,
        use_container_width=True,
        hide_index=True
    )

    st.write("### Retained Features")

    method = st.selectbox(
        "Select technique",
        [
            "Original",
            "Low Variance",
            "High Correlation",
            "Forward Selection",
            "Backward Elimination",
            "Factor Analysis"
        ]
    )

    feature_map = {
        "Original": list(X.columns),
        "Low Variance": low_var,
        "High Correlation": high_corr,
        "Forward Selection": forward,
        "Backward Elimination": backward
    }

    if method == "Factor Analysis":
        st.info(
            f"Factor Analysis creates {factor_n} latent factors "
            "instead of retaining the original feature names."
        )
    else:
        st.write(
            feature_map[method]
        )


# ------------------------------------------------------------
# MODEL EVALUATION
# ------------------------------------------------------------

with tab3:

    st.header("Model Evaluation")

    results = []

    feature_sets = {
        "Original": list(X.columns),
        "Low Variance": low_var,
        "High Correlation": high_corr,
        "Forward Selection": forward,
        "Backward Elimination": backward
    }

    for method_name, features in feature_sets.items():

        Xtr = X_train_scaled[features]
        Xte = X_test_scaled[features]

        if problem == "Regression":

            model = LinearRegression()
            model.fit(Xtr, y_train)

            pred = model.predict(Xte)

            results.append({
                "Method": method_name,
                "Features": len(features),
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(
                    mean_squared_error(y_test, pred)
                ),
                "R2": r2_score(y_test, pred)
            })

        else:

            model = LogisticRegression(
                max_iter=5000
            )

            model.fit(Xtr, y_train)

            pred = model.predict(Xte)

            results.append({
                "Method": method_name,
                "Features": len(features),
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred),
                "Recall": recall_score(y_test, pred),
                "F1 Score": f1_score(y_test, pred)
            })

    # Factor Analysis model
    if problem == "Regression":

        factor_model = LinearRegression()
        factor_model.fit(
            X_train_factor,
            y_train
        )

        factor_pred = factor_model.predict(
            X_test_factor
        )

        results.append({
            "Method": "Factor Analysis",
            "Features": factor_n,
            "MAE": mean_absolute_error(
                y_test,
                factor_pred
            ),
            "RMSE": np.sqrt(
                mean_squared_error(
                    y_test,
                    factor_pred
                )
            ),
            "R2": r2_score(
                y_test,
                factor_pred
            )
        })

        result_df = pd.DataFrame(results)

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        st.write("### R² Comparison")

        fig, ax = plt.subplots()

        ax.bar(
            result_df["Method"],
            result_df["R2"]
        )

        ax.set_ylabel("R² Score")
        ax.set_xlabel("Feature Selection Method")
        ax.tick_params(axis="x", rotation=30)

        st.pyplot(fig)

    else:

        factor_model = LogisticRegression(
            max_iter=5000
        )

        factor_model.fit(
            X_train_factor,
            y_train
        )

        factor_pred = factor_model.predict(
            X_test_factor
        )

        results.append({
            "Method": "Factor Analysis",
            "Features": factor_n,
            "Accuracy": accuracy_score(
                y_test,
                factor_pred
            ),
            "Precision": precision_score(
                y_test,
                factor_pred
            ),
            "Recall": recall_score(
                y_test,
                factor_pred
            ),
            "F1 Score": f1_score(
                y_test,
                factor_pred
            )
        })

        result_df = pd.DataFrame(results)

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        st.write("### Accuracy Comparison")

        fig, ax = plt.subplots()

        ax.bar(
            result_df["Method"],
            result_df["Accuracy"]
        )

        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Feature Selection Method")
        ax.tick_params(axis="x", rotation=30)

        st.pyplot(fig)


# ------------------------------------------------------------
# OPTIMIZATION
# ------------------------------------------------------------

with tab4:

    st.header("Gradient Descent and Hyperparameter Optimization")

    st.subheader("Gradient Descent")

    if problem == "Regression":

        gd_features = forward

        gd_lr = st.select_slider(
            "Learning Rate",
            options=[
                0.0001,
                0.0005,
                0.001,
                0.005
            ],
            value=0.005
        )

        gd_iterations = st.slider(
            "Iterations",
            min_value=100,
            max_value=5000,
            value=2000,
            step=100
        )

        gd_weights, gd_loss = linear_gradient_descent(
            X_train_scaled[gd_features],
            y_train,
            learning_rate=gd_lr,
            iterations=gd_iterations
        )

        gd_pred = gd_predict(
            X_test_scaled[gd_features],
            gd_weights
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "MAE",
            f"{mean_absolute_error(y_test, gd_pred):.4f}"
        )

        c2.metric(
            "RMSE",
            f"{np.sqrt(mean_squared_error(y_test, gd_pred)):.4f}"
        )

        c3.metric(
            "R²",
            f"{r2_score(y_test, gd_pred):.4f}"
        )

        fig, ax = plt.subplots()

        ax.plot(
            range(1, len(gd_loss) + 1),
            gd_loss
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Training MSE")
        ax.set_title("Linear Regression Gradient Descent Convergence")

        st.pyplot(fig)

        st.info(
            "Changing the learning rate and number of iterations "
            "shows how quickly Gradient Descent converges."
        )

    else:

        gd_features = forward

        gd_lr = st.select_slider(
            "Learning Rate",
            options=[
                0.001,
                0.005,
                0.01,
                0.05
            ],
            value=0.01
        )

        gd_iterations = st.slider(
            "Iterations",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100
        )

        gd_weights, gd_loss = logistic_gradient_descent(
            X_train_scaled[gd_features],
            y_train,
            learning_rate=gd_lr,
            iterations=gd_iterations
        )

        gd_pred = gd_predict(
            X_test_scaled[gd_features],
            gd_weights,
            classification=True
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Accuracy",
            f"{accuracy_score(y_test, gd_pred):.4f}"
        )

        c2.metric(
            "Precision",
            f"{precision_score(y_test, gd_pred):.4f}"
        )

        c3.metric(
            "F1 Score",
            f"{f1_score(y_test, gd_pred):.4f}"
        )

        fig, ax = plt.subplots()

        ax.plot(
            range(1, len(gd_loss) + 1),
            gd_loss
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log Loss")
        ax.set_title("Logistic Regression Gradient Descent Convergence")

        st.pyplot(fig)

    # --------------------------------------------------------
    # GRIDSEARCHCV
    # --------------------------------------------------------

    st.subheader("GridSearchCV")

    if problem == "Regression":

        grid_model = LinearRegression()

        param_grid = {
            "fit_intercept": [
                True,
                False
            ]
        }

        grid = GridSearchCV(
            grid_model,
            param_grid,
            cv=5,
            scoring="r2",
            n_jobs=-1
        )

        grid.fit(
            X_train_scaled[forward],
            y_train
        )

        pred = grid.predict(
            X_test_scaled[forward]
        )

        st.write(
            "Best Parameters:",
            grid.best_params_
        )

        st.write(
            "Best CV R²:",
            round(grid.best_score_, 4)
        )

        grid_results = pd.DataFrame({
            "Metric": [
                "MAE",
                "RMSE",
                "R²"
            ],
            "Value": [
                mean_absolute_error(y_test, pred),
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        pred
                    )
                ),
                r2_score(y_test, pred)
            ]
        })

        st.dataframe(
            grid_results,
            use_container_width=True,
            hide_index=True
        )

    else:

        grid_model = LogisticRegression(
            max_iter=5000
        )

        param_grid = {
            "C": [
                0.001,
                0.01,
                0.1,
                1,
                10,
                100
            ]
        }

        grid = GridSearchCV(
            grid_model,
            param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        grid.fit(
            X_train_scaled[forward],
            y_train
        )

        pred = grid.predict(
            X_test_scaled[forward]
        )

        st.write(
            "Best Parameters:",
            grid.best_params_
        )

        st.write(
            "Best CV Accuracy:",
            round(grid.best_score_, 4)
        )

        grid_results = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ],
            "Value": [
                accuracy_score(y_test, pred),
                precision_score(y_test, pred),
                recall_score(y_test, pred),
                f1_score(y_test, pred)
            ]
        })

        st.dataframe(
            grid_results,
            use_container_width=True,
            hide_index=True
        )


# ------------------------------------------------------------
# SELF-LEARNING
# ------------------------------------------------------------

with tab5:

    st.header("Self-Learning Experiment")

    if problem == "Regression":

        feature_counts = [
            3,
            5,
            7,
            9
        ]

        scores = []

        for n in feature_counts:

            selector = SequentialFeatureSelector(
                LinearRegression(),
                n_features_to_select=n,
                direction="forward",
                scoring="r2",
                cv=5,
                n_jobs=-1
            )

            selector.fit(
                X_train_scaled,
                y_train
            )

            selected = X_train_scaled.columns[
                selector.get_support()
            ]

            cv_scores = cross_val_score(
                LinearRegression(),
                X_train_scaled[selected],
                y_train,
                cv=5,
                scoring="r2"
            )

            scores.append({
                "Features": n,
                "CV Mean R²": cv_scores.mean(),
                "CV Std": cv_scores.std()
            })

        self_df = pd.DataFrame(scores)

        st.dataframe(
            self_df,
            use_container_width=True,
            hide_index=True
        )

        fig, ax = plt.subplots()

        ax.plot(
            self_df["Features"],
            self_df["CV Mean R²"],
            marker="o"
        )

        ax.set_xlabel("Number of Selected Features")
        ax.set_ylabel("Mean CV R²")
        ax.set_title("Self-Learning: Feature Count vs R²")

        st.pyplot(fig)

    else:

        feature_counts = [
            5,
            10,
            15,
            20
        ]

        scores = []

        for n in feature_counts:

            selector = SequentialFeatureSelector(
                LogisticRegression(
                    max_iter=5000
                ),
                n_features_to_select=n,
                direction="forward",
                scoring="accuracy",
                cv=5,
                n_jobs=-1
            )

            selector.fit(
                X_train_scaled,
                y_train
            )

            selected = X_train_scaled.columns[
                selector.get_support()
            ]

            cv_scores = cross_val_score(
                LogisticRegression(
                    max_iter=5000
                ),
                X_train_scaled[selected],
                y_train,
                cv=5,
                scoring="accuracy"
            )

            scores.append({
                "Features": n,
                "CV Mean Accuracy": cv_scores.mean(),
                "CV Std": cv_scores.std()
            })

        self_df = pd.DataFrame(scores)

        st.dataframe(
            self_df,
            use_container_width=True,
            hide_index=True
        )

        fig, ax = plt.subplots()

        ax.plot(
            self_df["Features"],
            self_df["CV Mean Accuracy"],
            marker="o"
        )

        ax.set_xlabel("Number of Selected Features")
        ax.set_ylabel("Mean CV Accuracy")
        ax.set_title("Self-Learning: Feature Count vs Accuracy")

        st.pyplot(fig)


# ------------------------------------------------------------
# CONCLUSION
# ------------------------------------------------------------

st.divider()

st.header("📝 Conclusion and What I Understood")

st.write("""
This lab demonstrated how different feature selection and feature
reduction techniques can be applied to Linear Regression and
Logistic Regression. Low Variance Filtering, High Correlation
Filtering, Forward Selection, Backward Elimination and Factor
Analysis were studied and compared using model evaluation and
cross-validation. Gradient Descent helped in understanding the
effect of learning rate and iterations on convergence, while
GridSearchCV demonstrated how suitable hyperparameters can be
identified automatically. The self-learning experiment showed
that increasing the number of features does not always improve
model performance. Through this lab, I understood that effective
feature selection should preserve useful information while
reducing unnecessary or redundant variables, and that model
performance should be evaluated using both test metrics and
cross-validation rather than relying on a single score.
""")
