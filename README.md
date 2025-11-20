> # 🏦 LoanSense.AI - Loan Default Prediction – Bank Indessa

### *Machine Learning solution to identify potential loan defaulters*

> **Hackathon Project • AUC-ROC Evaluation • December 2016 Lending Dataset**

> ## ⚠️ Important Note  
> Due to hackathon time shortage and a few quick mistakes, this repo may not be completely polished.

---

## 📌 **Problem Overview**

Bank Indessa has been suffering from a sharp rise in **Non-Performing Assets (NPAs)** over recent quarters.
A major cause: **loan defaults**.

To control the risk and regain investor confidence, the bank wants to build a **predictive machine learning model** to estimate the likelihood that a member will default on their loan.

This project involves:

* Cleaning messy loan data
* Feature engineering
* Training multiple ML models
* Selecting the best performing model
* Predicting **probability of default** on unseen data

📄 *Problem statement PDF is included in the repo under `docs/problem-statement/`.*

---

## 🎯 **Objective**

Predict the **probability that a loan applicant will default**, using structured customer and loan data.

**Evaluation Metric:**
🔹 **AUC-ROC Score** (as defined in the problem statement)

---

## 📁 **Project Structure**

```
Hackathon/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── docs/
│   ├── images/
│   ├── ppt/
│   ├── problem-statement/
│   └── report/
│
├── models/
│   ├── preprocessor.pkl
│   ├── svm_sgd_pipeline.joblib
│   └──  xgb_booster.json
│
├── notebooks/
│   └── 01-Main.ipynb
│
├── submissions/
│   ├── submission.xlsx
│   └── submission_XGB.xlsx
│
├── spider/               # Spyder IDE environment (ignored)
│
├── .gitignore
├── README.md
├── requirements.txt
└── requirements_installed.txt
```

> ✔ All large files (models, data, Spyder env) are ignored via `.gitignore`.

---

## 📊 **Dataset Information**

* **Train**, **Test**, and **Submission** CSV files were provided.
* The data includes **loan details**, **employment info**, **credit history**, **repayment records**, and **member demographics**.

### 🔑 **Key Variables**

| Variable              | Description                                    |
| --------------------- | ---------------------------------------------- |
| `member_id`           | Unique ID assigned to each member              |
| `loan_amnt`           | Applied loan amount                            |
| `funded_amnt`         | Loan amount funded by bank                     |
| `funded_amnt_inv`     | Amount funded by investors                     |
| `term`                | Loan term (months)                             |
| `int_rate`            | Interest rate (%)                              |
| `grade`, `sub_grade`  | Lending grades                                 |
| `emp_length`          | Years of employment                            |
| `home_ownership`      | Home ownership status                          |
| `annual_inc`          | Annual income                                  |
| `verification_status` | Whether income verified                        |
| `purpose`             | Purpose of loan                                |
| `dti`                 | Debt-to-Income ratio                           |
| `delinq_2yrs`         | Past delinquencies                             |
| `open_acc`            | Active credit lines                            |
| `revol_bal`           | Revolving balance                              |
| `total_acc`           | Total credit lines                             |
| `loan_status`         | Target variable → 1 = Default, 0 = Non-default |

(Full description available in the PDF.)

---

## 🛠️ **Approach & Methodology**

### ✔ 1. Exploratory Data Analysis (EDA)

Performed using the notebook:
📓 **`notebooks/01-Main.ipynb`**

* Missing value analysis
* Outlier inspection
* Distribution plots
* Correlation heatmaps
* Target imbalance check

### ✔ 2. Data Preprocessing

Includes:

* Handling missing values
* Label encoding & one-hot encoding
* Scaling numerical features
* Feature selection
* Creating final training matrix

Saved as: **`models/preprocessor.pkl`**

### ✔ 3. Model Training

Multiple machine learning models were trained:

* **XGBoost**
* **SVM (SGD Pipeline)**
* **Random Forest (Archived)**
* **Baseline Logistic Regression**

Final saved models:

* `svm_sgd_pipeline.joblib`
* `xgb_booster.json`

### ✔ 4. Model Evaluation

Evaluated using:

* AUC-ROC Curve
* Precision-Recall
* Cross-validation performance
* Feature importance (for tree models)

**Best Model: XGBoost**

### ✔ 5. Submission Generation

Two submissions generated:

* `submission.xlsx`
* `submission_XGB.xlsx`

---


# 📝 Hackathon Project – Environment Setup & Structure Guide

This README documents every step taken to set up the development environment for the hackathon, including Python installation, virtual environment creation, dependency installation, GPU considerations, and the recommended project folder structure.

---

## 📌 1. Python Installation

The system originally had:
`Python 3.11`

The hackathon required Python 3.12, so we installed it separately from the official Python website.

After installation:
`py -0`

showed:

```
-V:3.12 *   Python 3.12 (64-bit)
-V:3.11     Python 3.11 (64-bit)
```

✔ Python 3.12 is installed
✔ Python 3.11 is still available
✔ Existing projects remain unaffected

---

## 📌 2. Creating a Virtual Environment (venv)

Navigate to the hackathon folder:
`cd "D:\Omnie Solutions\Hackathon"`

Create the venv using Python 3.12:
`py -3.12 -m venv spider`

Activate it:
`spider\Scripts\activate`

Upgrade pip & tools:
`pip install --upgrade pip setuptools wheel`

---

## 📌 3. Installing PyTorch (CPU Version)

We chose to keep CPU-only PyTorch for simplicity and stability in the hackathon.
`pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu`

GPU support was optional and discussed, but we intentionally stayed on CPU.

---

## 📌 4. Installing All Project Dependencies

We installed the full dependency list provided in `requirements.txt`:
`pip install -r requirements.txt`

This installed libraries such as:

- `accelerate`
- `catboost`
- `fastapi`
- `lightgbm`
- `pandas`
- `scikit-learn`
- `scipy`
- `statsmodels`
- `transformers`
- `sentence-transformers`
- `xgboost`
- ...and many more

All packages installed successfully.

---

## 📌 5. GPU Mode (Optional Discussion)

Initially, `nvidia-smi` failed due to GPU mode being off.

After switching to NVIDIA GPU mode, the GPU was detected correctly:

- CUDA Version: 12.5
- GPU: NVIDIA GeForce RTX 2050
- Driver Version: 555.97

We concluded:
✔ Installing GPU PyTorch is safe
✔ But for simplicity we kept CPU-only torch
✔ The environment remains stable and ready for CPU-based ML/NLP work

---

## 📌 6. Folder Structure for the Project

Hackathon-folder structure:

```
Hackathon/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── docs/
│   ├── images/
│   ├── ppt/
│   ├── problem-statement/
│   └── report/
│
├── models/
│   ├── preprocessor.pkl
│   ├── svm_sgd_pipeline.joblib
│   └──  xgb_booster.json
│   
│
├── notebooks/
│   └── 01-Main.ipynb
│
├── submissions/
│   ├── submission.xlsx
│   └── submission_XGB.xlsx
│
├── spider/                # sSpider IDE environment (ignored)
│
├── .gitignore
├── README.md
├── requirements.txt
└── requirements_installed.txt

```


---

## 📌 7. Environment Validation Tests

To verify core libraries:
```python
import torch, sklearn, pandas, numpy, transformers
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("pandas:", pandas.__version__)
print("numpy:", numpy.__version__)
print("sklearn:", sklearn.__version__)
print("transformers:", transformers.__version__)
```

Expected:
- All imports successful
- `cuda_available = False` (CPU mode)
---
## 📌 8. Freezing the Environment

After all packages were installed:
`pip freeze > requirements_installed.txt`

---

## 🚀 **How to Run the Project (No need for set up)**

### **Step 1: Clone the repository**

```bash
git clone https://github.com/HarshitWaldia/LoanSenseAI.git
cd Hackathon
```

### **Step 2: Install dependencies**

Create the venv :
`python -m venv spider`

Activate it:
`spider\Scripts\activate`

```bash
pip install -r requirements.txt
```

### **Step 3: Open the Jupyter Notebook**

```bash
jupyter notebook notebooks/01-Main.ipynb
```

OR run manually:

```python
# Load preprocessor
import joblib
pre = joblib.load("models/preprocessor.pkl")

# Load model
import xgboost as xgb
model = xgb.Booster()
model.load_model("models/xgb_booster.json")

# Make predictions...
```

---

## 🧪 **Tech Stack**

* **Python 3.x**
* **NumPy / Pandas**
* **Matplotlib / Seaborn**
* **Scikit-learn**
* **XGBoost**
* **Joblib**
* **Jupyter Notebook**

---

## 📈 Results

* Best AUC-ROC Score (Cross-validation): **XGBoost**
* Best Public Submission: `submission_XGB.xlsx`

---

## 🏁 Conclusion

This project successfully builds a machine learning pipeline to predict loan defaults for Bank Indessa.
The model helps the bank:

* identify high-risk borrowers
* reduce NPAs
* safeguard investor confidence
* optimize loan approvals

---

## 📩 Contact

For questions or clarifications, feel free to reach out!
**Harshit Waldia**

-   **Email**: `harshitwaldia112@gmail.com`
-   **LinkedIn**: [linkedin.com/in/harshit-waldia](https://www.linkedin.com/in/harshit-waldia/)
-   **GitHub**: [@HarshitWaldia](https://github.com/HarshitWaldia)

---
