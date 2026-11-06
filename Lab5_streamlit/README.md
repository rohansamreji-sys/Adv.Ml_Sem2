# Lab Exercise 5 – Single Layer Perceptron (SLP)

**Course:** MAI511-2 Advanced Machine Learning  
**Program:** I MSc AIML  
**Date:** 03-11-2026

## Contents

| File | Description |
|------|-------------|
| `perceptron.py` | Clean, reusable Single Layer Perceptron class + helper functions |
| `Lab5_Single_Layer_Perceptron.ipynb` | Full experimental notebook (Questions 1–3 + analysis + plots) |
| `app.py` | Interactive Streamlit dashboard |
| `requirements.txt` | Python dependencies |

---

## How to run in Google Colab

1. Upload the whole `lab5_slp` folder **or** just the two files:
   - `perceptron.py`
   - `Lab5_Single_Layer_Perceptron.ipynb`
2. Open the notebook.
3. Make sure `perceptron.py` is in the same directory (or in `/content/`).
4. Run all cells.

Alternatively you can copy-paste the class definition from `perceptron.py` into the first code cell of the notebook if you prefer a single-file submission.

---

## How to run the Streamlit dashboard

```bash
cd lab5_slp
pip install -r requirements.txt
streamlit run app.py
```

Features of the dashboard:
- Select any gate (AND / OR / NAND / XOR) or the sample datasets
- Adjust learning rate, epochs, random seed
- Live decision-boundary plot
- Training error curve
- Prediction table with correctness highlighting
- Analysis panel explaining linear separability

---

## Lab Requirements Coverage

### Question 1 – AND, OR, NAND
- Input patterns & targets defined
- Perceptron learning algorithm implemented (weights, bias, lr, step activation)
- Separate training for each gate
- Final weights & bias displayed
- Predictions for all binary combinations
- Accuracy comparison with truth table

### Question 2 – XOR
- Same algorithm applied to XOR
- Weight updates / error curve observed across epochs
- Predictions & accuracy calculated
- Clear analysis why SLP cannot learn XOR (non-linearly separable)
- Comparison table with AND/OR/NAND

### Question 3 – Custom dataset
- Candidate Selection sample (binary features) implemented
- Scholarship Eligibility sample (continuous features, normalized) also implemented
- Full training + accuracy + decision boundary

### Extra (Rubrics)
- **Visualization** – decision boundaries + training curves in notebook and Streamlit
- **Streamlit** – fully interactive dashboard (`app.py`)
- Concept clarity ready for viva (see analysis cells)

---

## Expected Results (typical run)

| Gate / Dataset        | Linearly Separable | Outcome                          | Accuracy |
|-----------------------|--------------------|----------------------------------|----------|
| AND                   | Yes                | Converges                        | 100 %    |
| OR                    | Yes                | Converges                        | 100 %    |
| NAND                  | Yes                | Converges                        | 100 %    |
| XOR                   | No                 | Does not converge to perfect sol.| 50 %     |
| Candidate Selection   | Yes                | Converges                        | 100 %    |
| Scholarship (norm.)   | Yes                | Converges                        | 100 %    |

---

## File naming for submission (as per instructions)

Rename your final notebook / zip according to the pattern:

```
YourName_Last3DigitsOfRegNo_Lab5
```

Example: `Aaron_101_Lab5`

Upload the notebook (and optionally the Streamlit app) to Google Classroom within the scheduled time.
