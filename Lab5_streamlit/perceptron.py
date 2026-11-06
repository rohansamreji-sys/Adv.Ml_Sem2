"""
Single Layer Perceptron (SLP) Implementation
MAI511-2 Advanced Machine Learning - Lab Exercise 5
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class SingleLayerPerceptron:
    """
    Single Layer Perceptron for binary classification.
    
    Activation: Step function (threshold at 0)
    Learning rule: Rosenblatt's perceptron rule
    w <- w + lr * (target - prediction) * x
    b <- b + lr * (target - prediction)
    """

    def __init__(
        self,
        n_features: int = 2,
        learning_rate: float = 0.1,
        epochs: int = 100,
        random_state: Optional[int] = 42,
        verbose: bool = False,
    ):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose

        # Will be initialized on fit
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.history: List[Dict] = []  # track weight updates, errors per epoch
        self.converged: bool = False
        self.final_epoch: int = 0

    def _initialize_weights(self):
        rng = np.random.default_rng(self.random_state)
        # Small random weights help avoid symmetry issues
        self.weights = rng.uniform(-0.5, 0.5, size=self.n_features)
        self.bias = rng.uniform(-0.5, 0.5)

    def _activation(self, z: float) -> int:
        """Step / Heaviside activation function."""
        return 1 if z >= 0 else 0

    def net_input(self, x: np.ndarray) -> float:
        """Weighted sum + bias."""
        return float(np.dot(x, self.weights) + self.bias)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        X = np.atleast_2d(X)
        return np.array([self._activation(self.net_input(x)) for x in X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SingleLayerPerceptron":
        """
        Train the perceptron.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)  (binary 0/1)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features = X.shape[1]
        self._initialize_weights()
        self.history = []
        self.converged = False
        self.final_epoch = 0

        n_samples = X.shape[0]

        for epoch in range(1, self.epochs + 1):
            errors = 0
            epoch_updates = []

            # Shuffle for stochastic-ish updates (optional but good practice)
            indices = np.arange(n_samples)
            if self.random_state is not None:
                rng = np.random.default_rng(self.random_state + epoch)
                rng.shuffle(indices)

            for idx in indices:
                xi = X[idx]
                target = y[idx]
                prediction = self._activation(self.net_input(xi))
                error = target - prediction

                if error != 0:
                    # Weight & bias update
                    update = self.learning_rate * error
                    self.weights += update * xi
                    self.bias += update
                    errors += 1
                    epoch_updates.append(
                        {
                            "sample": idx,
                            "x": xi.tolist(),
                            "target": int(target),
                            "pred_before": int(prediction),
                            "error": int(error),
                            "weights_after": self.weights.copy().tolist(),
                            "bias_after": float(self.bias),
                        }
                    )

            # Record epoch summary
            self.history.append(
                {
                    "epoch": epoch,
                    "errors": errors,
                    "weights": self.weights.copy().tolist(),
                    "bias": float(self.bias),
                    "updates": epoch_updates,
                }
            )

            if self.verbose and (epoch % 10 == 0 or errors == 0):
                print(
                    f"Epoch {epoch:3d} | Misclassifications: {errors:2d} | "
                    f"Weights: {np.round(self.weights, 4)} | Bias: {self.bias:.4f}"
                )

            if errors == 0:
                self.converged = True
                self.final_epoch = epoch
                if self.verbose:
                    print(f"\n>>> Converged at epoch {epoch}")
                break
        else:
            self.final_epoch = self.epochs
            if self.verbose:
                print(f"\n>>> Did not fully converge after {self.epochs} epochs")

        return self

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Classification accuracy."""
        y_pred = self.predict(X)
        y = np.asarray(y).ravel()
        return float(np.mean(y_pred == y))

    def get_decision_boundary_2d(self, x_range=(-0.5, 1.5), n_points=100):
        """
        For 2-feature models: return x and y points of the decision boundary
        w1*x + w2*y + b = 0  =>  y = -(w1*x + b)/w2
        """
        if self.n_features != 2 or self.weights is None:
            return None, None
        w1, w2 = self.weights
        if abs(w2) < 1e-10:
            # Vertical line
            x = -self.bias / w1 if abs(w1) > 1e-10 else 0
            return np.array([x, x]), np.array([x_range[0], x_range[1]])
        xs = np.linspace(x_range[0], x_range[1], n_points)
        ys = -(w1 * xs + self.bias) / w2
        return xs, ys


def get_logic_gate_data(gate: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return X, y for classic Boolean gates."""
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=float,
    )

    gate = gate.upper()
    if gate == "AND":
        y = np.array([0, 0, 0, 1])
    elif gate == "OR":
        y = np.array([0, 1, 1, 1])
    elif gate == "NAND":
        y = np.array([1, 1, 1, 0])
    elif gate == "XOR":
        y = np.array([0, 1, 1, 0])
    elif gate == "NOR":
        y = np.array([1, 0, 0, 0])
    elif gate == "XNOR":
        y = np.array([1, 0, 0, 1])
    else:
        raise ValueError(f"Unknown gate: {gate}")

    return X, y


def print_truth_table(gate: str, model: SingleLayerPerceptron, X: np.ndarray, y: np.ndarray):
    """Pretty print predicted vs actual."""
    preds = model.predict(X)
    print(f"\n{'='*50}")
    print(f"  {gate} Gate - Truth Table Comparison")
    print(f"{'='*50}")
    print(f"{'x1':>4} {'x2':>4} | {'Actual':>6} {'Predicted':>9} | {'Match'}")
    print("-" * 50)
    correct = 0
    for i in range(len(X)):
        match = "✓" if preds[i] == y[i] else "✗"
        if preds[i] == y[i]:
            correct += 1
        print(
            f"{int(X[i,0]):>4} {int(X[i,1]):>4} | {int(y[i]):>6} {int(preds[i]):>9} | {match}"
        )
    acc = correct / len(y) * 100
    print("-" * 50)
    print(f"Accuracy: {acc:.1f}%  ({correct}/{len(y)})")
    print(f"Final Weights: {np.round(model.weights, 4)}")
    print(f"Final Bias   : {model.bias:.4f}")
    print(f"Converged    : {model.converged} (epoch {model.final_epoch})")
    print(f"{'='*50}\n")
    return acc


# ---------------------------------------------------------------------------
# Sample datasets for Question 3
# ---------------------------------------------------------------------------

def get_scholarship_dataset():
    """
    Student scholarship eligibility.
    Features: Internal Assessment, Attendance (%)
    Target: 1 = Eligible, 0 = Not Eligible
    """
    data = {
        "Student": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"],
        "Internal_Assessment": [85, 75, 65, 45, 50, 30, 90, 40],
        "Attendance": [90, 85, 80, 60, 55, 45, 95, 50],
        "Eligibility": [1, 1, 1, 0, 0, 0, 1, 0],
    }
    import pandas as pd
    df = pd.DataFrame(data)
    X = df[["Internal_Assessment", "Attendance"]].values.astype(float)
    y = df["Eligibility"].values
    return df, X, y


def get_candidate_selection_dataset():
    """
    Candidate selection (binary features).
    Technical Test & Interview → Selection
    """
    data = {
        "Candidate": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"],
        "Technical_Test": [1, 1, 0, 0, 1, 0, 1, 0],
        "Interview": [1, 0, 1, 0, 1, 0, 0, 1],
        "Selection": [1, 1, 1, 0, 1, 0, 1, 1],
    }
    import pandas as pd
    df = pd.DataFrame(data)
    X = df[["Technical_Test", "Interview"]].values.astype(float)
    y = df["Selection"].values
    return df, X, y


if __name__ == "__main__":
    # Quick smoke test
    print("=== Smoke Test: AND Gate ===")
    X, y = get_logic_gate_data("AND")
    model = SingleLayerPerceptron(learning_rate=0.1, epochs=50, verbose=True)
    model.fit(X, y)
    print_truth_table("AND", model, X, y)
