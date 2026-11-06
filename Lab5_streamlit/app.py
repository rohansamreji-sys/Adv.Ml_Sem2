"""
Streamlit Dashboard – Single Layer Perceptron Lab Exercise 5
MAI511-2 Advanced Machine Learning

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import os

# Ensure local module is importable
sys.path.insert(0, os.path.dirname(__file__))
from perceptron import (
    SingleLayerPerceptron,
    get_logic_gate_data,
    get_scholarship_dataset,
    get_candidate_selection_dataset,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SLP Lab 5 – Advanced ML",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 Single Layer Perceptron – Lab Exercise 5")
st.caption("MAI511-2 Advanced Machine Learning | I MSc AIML")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Training Controls")

gate = st.sidebar.selectbox(
    "Select Logic Gate / Dataset",
    ["AND", "OR", "NAND", "XOR", "Candidate Selection", "Scholarship Eligibility"],
)

lr = st.sidebar.slider("Learning Rate (η)", 0.01, 1.0, 0.1, 0.01)
epochs = st.sidebar.slider("Max Epochs", 10, 200, 50, 10)
seed = st.sidebar.number_input("Random Seed", 0, 999, 42)
show_updates = st.sidebar.checkbox("Show weight updates (verbose)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**Theory Reminder**  
- AND / OR / NAND → linearly separable → converges  
- XOR → **not** linearly separable → never reaches 100 %  
"""
)

# ---------------------------------------------------------------------------
# Data selection
# ---------------------------------------------------------------------------
if gate in ["AND", "OR", "NAND", "XOR"]:
    X, y = get_logic_gate_data(gate)
    feature_names = ["x1", "x2"]
    title = f"{gate} Gate"
elif gate == "Candidate Selection":
    df, X, y = get_candidate_selection_dataset()
    feature_names = ["Technical Test", "Interview"]
    title = "Candidate Selection"
else:  # Scholarship
    df, X_raw, y = get_scholarship_dataset()
    X_min, X_max = X_raw.min(axis=0), X_raw.max(axis=0)
    X = (X_raw - X_min) / (X_max - X_min + 1e-8)
    feature_names = ["Internal Assessment (norm)", "Attendance (norm)"]
    title = "Scholarship Eligibility (normalized)"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def train_model(_X, _y, lr, epochs, seed):
    model = SingleLayerPerceptron(
        learning_rate=lr,
        epochs=epochs,
        random_state=seed,
        verbose=False,
    )
    model.fit(_X, _y)
    return model

with st.spinner("Training perceptron..."):
    model = train_model(X, y, lr, epochs, seed)

# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
acc = model.score(X, y) * 100
col1.metric("Accuracy", f"{acc:.1f} %")
col2.metric("Converged", "Yes ✅" if model.converged else "No ❌")
col3.metric("Epochs Used", model.final_epoch)
col4.metric("Final Bias", f"{model.bias:.4f}")

st.markdown(f"**Final Weights:** `{np.round(model.weights, 4).tolist()}`")

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Decision Boundary", "📈 Training Curve", "📋 Truth / Predictions", "📚 Analysis"]
)

# ---- Tab 1: Decision Boundary ----
with tab1:
    st.subheader(f"Decision Boundary – {title}")

    fig, ax = plt.subplots(figsize=(7, 6))

    # Background mesh
    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, levels=[-0.1, 0.5, 1.1], colors=["#ff9999", "#99ff99"])

    # Points
    for label, marker, color, name in [
        (0, "o", "#d62728", "Class 0"),
        (1, "s", "#2ca02c", "Class 1"),
    ]:
        mask = y == label
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=color,
            marker=marker,
            s=140,
            edgecolors="k",
            label=name,
            zorder=5,
        )

    # Decision line
    xs, ys = model.get_decision_boundary_2d(x_range=(x_min, x_max))
    if xs is not None:
        ax.plot(xs, ys, "k--", lw=2.5, label="Decision boundary")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(feature_names[0], fontsize=12)
    ax.set_ylabel(feature_names[1], fontsize=12)
    ax.set_title(f"{title}  |  Acc = {acc:.0f}%", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    st.pyplot(fig)
    plt.close()

# ---- Tab 2: Training Curve ----
with tab2:
    st.subheader("Misclassifications per Epoch")

    hist = model.history
    if hist:
        epochs_list = [h["epoch"] for h in hist]
        errors = [h["errors"] for h in hist]

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(epochs_list, errors, "b-o", lw=2, markersize=5)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Number of Misclassifications")
        ax2.set_title("Training Dynamics")
        ax2.grid(True, alpha=0.3)
        if model.converged:
            ax2.axvline(model.final_epoch, color="green", ls="--", label=f"Converged @ {model.final_epoch}")
            ax2.legend()
        st.pyplot(fig2)
        plt.close()

        # Optional weight history table
        if show_updates:
            st.markdown("#### Weight & Bias Trajectory")
            traj = pd.DataFrame(
                [
                    {
                        "Epoch": h["epoch"],
                        "Errors": h["errors"],
                        "w1": round(h["weights"][0], 4),
                        "w2": round(h["weights"][1], 4) if len(h["weights"]) > 1 else None,
                        "bias": round(h["bias"], 4),
                    }
                    for h in hist
                ]
            )
            st.dataframe(traj, use_container_width=True)
    else:
        st.info("No training history available.")

# ---- Tab 3: Predictions ----
with tab3:
    st.subheader("Predictions vs Ground Truth")

    preds = model.predict(X)
    table = pd.DataFrame(
        {
            feature_names[0]: X[:, 0],
            feature_names[1]: X[:, 1],
            "Actual": y.astype(int),
            "Predicted": preds.astype(int),
            "Correct": ["✅" if p == a else "❌" for p, a in zip(preds, y)],
        }
    )
    st.dataframe(table.style.apply(
        lambda row: ["background-color: #d4edda" if row["Correct"] == "✅" else "background-color: #f8d7da"] * len(row),
        axis=1
    ), use_container_width=True)

    st.metric("Classification Accuracy", f"{acc:.1f} %")

# ---- Tab 4: Analysis ----
with tab4:
    st.subheader("Analysis & Comparison")

    st.markdown(
        """
### Linear Separability Summary

| Logic Gate / Dataset | Linearly Separable | SLP Learning Outcome |
|----------------------|--------------------|----------------------|
| **AND**              | Yes                | Converges            |
| **OR**               | Yes                | Converges            |
| **NAND**             | Yes                | Converges            |
| **XOR**              | **No**             | Does **not** converge to a perfect solution |
| Candidate Selection  | Yes                | Converges            |
| Scholarship (norm.)  | Yes (approx.)      | Converges            |

### Why XOR fails
The four points of XOR form two interlocking diagonals.  
No single straight line can separate the two classes.  
This is the classic demonstration that a **single-layer** architecture is insufficient for non-linearly separable problems.

### Learning Rule Used
$$
\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta (y - \\hat{y})\\mathbf{x}, \\quad
b \\leftarrow b + \\eta (y - \\hat{y})
$$
Activation: Step function (threshold at 0).
"""
    )

    if gate == "XOR":
        st.warning(
            "⚠️ You selected **XOR**. Observe that accuracy never reaches 100 % "
            "and the decision boundary cannot separate all four points correctly. "
            "This is expected and is the key learning outcome of Question 2."
        )
    elif model.converged:
        st.success(
            f"✅ The model converged in {model.final_epoch} epoch(s) with 100 % accuracy "
            "on this linearly separable problem."
        )
    else:
        st.info("Model stopped after max epochs. Try increasing epochs or adjusting learning rate.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Lab Exercise 5 · Single Layer Perceptron · "
    "Correctness + Visualization + Streamlit · Advanced Machine Learning"
)
