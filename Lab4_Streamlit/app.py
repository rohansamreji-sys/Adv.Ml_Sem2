
# Comparative Analysis of PCA, LDA and ICA
# MAI511-2 - Advanced Machine Learning

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PCA, LDA and ICA Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("Comparative Analysis of PCA, LDA and ICA")
st.write(
    "An interactive application to explore dimensionality "
    "reduction techniques using the Breast Cancer Wisconsin dataset."
)

st.divider()


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_dataset():

    data = load_breast_cancer()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    y = pd.Series(data.target)

    return X, y


X, y = load_dataset()

class_names = np.array(["Malignant", "Benign"])
target_labels = class_names[y.to_numpy()]


# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Samples", X.shape[0])
col2.metric("Input Features", X.shape[1])
col3.metric("Target Classes", len(np.unique(y)))

st.subheader("First Five Records")
st.dataframe(X.head(), use_container_width=True)

st.subheader("Class Distribution")

class_counts = pd.Series(target_labels).value_counts()

fig, ax = plt.subplots()
class_counts.plot(kind="bar", ax=ax)

ax.set_title("Distribution of Breast Cancer Classes")
ax.set_xlabel("Class")
ax.set_ylabel("Number of Samples")
ax.tick_params(axis="x", rotation=0)

st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# STANDARDISATION
# --------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# TECHNIQUE SELECTION
# --------------------------------------------------

st.divider()
st.header("2. Dimensionality Reduction")

technique = st.sidebar.selectbox(
    "Select a Technique",
    ["PCA", "LDA", "ICA"]
)

st.sidebar.header("Component Settings")

if technique == "PCA":

    n_components = st.sidebar.slider(
        "Number of Components",
        min_value=2,
        max_value=10,
        value=2
    )

elif technique == "ICA":

    n_components = st.sidebar.slider(
        "Number of Components",
        min_value=2,
        max_value=10,
        value=2
    )

else:

    n_components = 1

    st.sidebar.info(
        "LDA uses 1 component because the dataset has 2 classes."
    )


# --------------------------------------------------
# APPLY SELECTED TECHNIQUE
# --------------------------------------------------

if technique == "PCA":

    model = PCA(n_components=n_components)
    X_transformed = model.fit_transform(X_scaled)

    st.subheader("Principal Component Analysis (PCA)")

    st.write(
        "PCA is an unsupervised technique that reduces "
        "dimensionality by retaining maximum variance."
    )

    variance = model.explained_variance_ratio_

    total_variance = np.sum(variance)

    st.metric(
        "Total Explained Variance",
        f"{total_variance * 100:.2f}%"
    )

    variance_df = pd.DataFrame({
        "Component": [
            f"PC{i+1}" for i in range(n_components)
        ],
        "Explained Variance": variance
    })

    st.subheader("Explained Variance")
    st.dataframe(variance_df, use_container_width=True)

    fig, ax = plt.subplots()

    ax.bar(
        variance_df["Component"],
        variance_df["Explained Variance"]
    )

    ax.set_title("PCA Explained Variance")
    ax.set_xlabel("Principal Components")
    ax.set_ylabel("Explained Variance Ratio")

    st.pyplot(fig)
    plt.close(fig)


elif technique == "LDA":

    model = LinearDiscriminantAnalysis(n_components=1)

    X_transformed = model.fit_transform(X_scaled, y)

    st.subheader("Linear Discriminant Analysis (LDA)")

    st.write(
        "LDA is a supervised technique that uses class labels "
        "to maximise separation between different classes."
    )

    lda_df = pd.DataFrame({
        "LD1": X_transformed[:, 0],
        "Class": target_labels
    })

    fig, ax = plt.subplots()

    for class_name in class_names:

        class_data = lda_df[
            lda_df["Class"] == class_name
        ]

        ax.hist(
            class_data["LD1"],
            bins=25,
            alpha=0.5,
            label=class_name
        )

    ax.set_title("LDA Class Separation")
    ax.set_xlabel("Linear Discriminant 1")
    ax.set_ylabel("Frequency")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)


else:

    model = FastICA(
        n_components=n_components,
        max_iter=2000,
        tol=0.001,
        random_state=42,
        whiten="unit-variance"
    )

    X_transformed = model.fit_transform(X_scaled)

    st.subheader("Independent Component Analysis (ICA)")

    st.write(
        "ICA is an unsupervised technique that attempts to "
        "identify statistically independent components."
    )

    correlation = np.corrcoef(
        X_transformed,
        rowvar=False
    )

    correlation_df = pd.DataFrame(
        correlation,
        columns=[
            f"IC{i+1}" for i in range(n_components)
        ],
        index=[
            f"IC{i+1}" for i in range(n_components)
        ]
    )

    st.subheader("Independent Component Correlation")
    st.dataframe(correlation_df, use_container_width=True)

    fig, ax = plt.subplots()

    image = ax.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    ax.set_title("ICA Component Correlation")
    ax.set_xlabel("Independent Components")
    ax.set_ylabel("Independent Components")

    fig.colorbar(image, ax=ax)

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# TRANSFORMED DATA VISUALISATION
# --------------------------------------------------

st.divider()
st.header("3. Transformed Data Visualisation")

if technique in ["PCA", "ICA"]:

    fig, ax = plt.subplots()

    for class_name in class_names:

        mask = target_labels == class_name

        ax.scatter(
            X_transformed[mask, 0],
            X_transformed[mask, 1],
            label=class_name,
            alpha=0.7
        )

    ax.set_xlabel(f"{technique} Component 1")
    ax.set_ylabel(f"{technique} Component 2")
    ax.set_title(f"{technique} - Two Component Visualisation")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

else:

    fig, ax = plt.subplots()

    for class_name in class_names:

        mask = target_labels == class_name

        ax.scatter(
            X_transformed[mask, 0],
            np.zeros(mask.sum()),
            label=class_name,
            alpha=0.7
        )

    ax.set_xlabel("Linear Discriminant 1")
    ax.set_yticks([])
    ax.set_title("LDA - One Component Visualisation")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# TRANSFORMED DATA TABLE
# --------------------------------------------------

st.subheader("Transformed Dataset")

component_names = [
    f"{technique}{i+1}"
    for i in range(X_transformed.shape[1])
]

result_df = pd.DataFrame(
    X_transformed,
    columns=component_names
)

result_df["Class"] = target_labels

st.dataframe(
    result_df.head(10),
    use_container_width=True
)

st.write(
    "Original dimensions:",
    X.shape
)

st.write(
    "Reduced dimensions:",
    X_transformed.shape
)


# --------------------------------------------------
# COMPARISON OF TECHNIQUES
# --------------------------------------------------

st.divider()
st.header("4. Comparison of PCA, LDA and ICA")

comparison_df = pd.DataFrame({

    "Technique": ["PCA", "LDA", "ICA"],

    "Learning Type": [
        "Unsupervised",
        "Supervised",
        "Unsupervised"
    ],

    "Uses Class Labels": [
        "No",
        "Yes",
        "No"
    ],

    "Main Objective": [
        "Maximum Variance",
        "Maximum Class Separation",
        "Statistical Independence"
    ],

    "Output Components": [
        "User selected",
        "1",
        "User selected"
    ]

})

st.dataframe(
    comparison_df,
    use_container_width=True
)


# --------------------------------------------------
# CONCLUSION
# --------------------------------------------------

st.divider()
st.header("5. Conclusion")

st.write(
    "PCA reduces dimensionality by retaining maximum variance, "
    "LDA uses class labels to improve class separation, and "
    "ICA attempts to identify statistically independent components. "
    "Each technique has a different objective and is useful "
    "for different machine learning applications."
)

st.success("Analysis completed successfully!")
