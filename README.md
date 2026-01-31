# Loan Prediction (Analytics Vidhya) — Rank 20

A machine-learning solution for the **Loan Prediction** problem (Analytics Vidhya), where the goal is to predict whether a loan application will be **approved (Y)** or **rejected (N)** based on applicant and loan attributes.

**Result:** Achieved a **public leaderboard score of ~0.819444** and reached **Rank 20** (as captured in the included screenshot/PDF).

---

## Highlights (for resume)

- Built an end-to-end classification pipeline for loan approval prediction: **data cleaning → feature engineering → model training → evaluation → submission generation**.
- Implemented and compared multiple baseline ML models (LogReg/KNN/Decision Tree/Random Forest), then improved performance using **XGBoost**.
- Generated final competition submission file (**bestCheck.csv**) containing `Loan_ID` and predicted `Loan_Status`.

---

## What I did (in order)

1. **Loaded the dataset** (`train2.csv`) and performed basic exploration (shape, describe, null counts, unique counts).
2. **Handled missing values** using domain-driven rules and statistical imputations:
   - Filled categorical fields (e.g., `Gender`, `Self_Employed`, `Dependents`, `Credit_History`) using **mode**.
   - Filled numeric fields (e.g., `LoanAmount`, `Loan_Amount_Term`) using **mean**.
   - Filled `Married` using a **Gender-based rule** for rows with null values (Male → Yes, Female → No).
3. **Feature engineered** additional income-related features:
   - `HouseIncome = ApplicantIncome + CoapplicantIncome`
   - `AverageHouseIncome = (ApplicantIncome + CoapplicantIncome)/2`
   - `Ratio_Inc_Loan = HouseIncome / LoanAmount`
4. **Converted target labels** `Loan_Status` from `Y/N` to `1/0`.
5. **Prepared model-ready features** using **one-hot encoding** (`pd.get_dummies`) and dropping non-feature ID columns (e.g., `Loan_ID`).
6. **Trained baseline ML models** and compared performance using a validation split (`train_test_split`).
7. **Trained XGBoost** as the best-performing model and used it to generate predictions for the test set (`test2.csv`).
8. **Exported submission CSV** with predictions (stored as **bestCheck.csv**).
9. (Optional exploration) Attempted a **simple voting ensemble** using predictions from Logistic Regression, KNN, Decision Tree, and XGBoost, but **XGBoost alone performed best**, so the final submission is based on XGBoost.

---

## ML models used / trained

### Baseline models (explored in `loan att1 .ipynb`)
- **Logistic Regression** (scikit-learn)
- **K-Nearest Neighbors (KNN)** (scikit-learn)
- **Decision Tree Classifier** (scikit-learn)  
  - Included a small loop-based sweep over depth-related settings to observe train vs validation behavior.
- **Random Forest Classifier** (scikit-learn)

### Best-performing model (final submission)
- **XGBoost** (xgboost)
  - Trained using `xgb.train(...)` on one-hot encoded features.
  - Used to generate the predictions stored in `bestCheck.csv`.

### Ensemble attempt (optional)
- **Mode / majority-vote ensemble** combining predictions from:
  - Logistic Regression, KNN, Decision Tree, and XGBoost  
  - Implemented as a per-row `mode([...])` vote (see `besttONE_0.819444.ipynb`).

---

## Data preprocessing & feature engineering

**Missing value handling**
- `Gender`: filled using mode
- `Married`: filled via a rule based on `Gender` (for null rows)
- `Dependents`: filled using mode
- `Self_Employed`: filled using mode
- `LoanAmount`: filled using mean
- `Loan_Amount_Term`: filled using mean
- `Credit_History`: filled using mode (then treated as categorical for encoding)

**Feature engineering**
- Total household income (`HouseIncome`)
- Average household income (`AverageHouseIncome`)
- Income-to-loan ratio (`Ratio_Inc_Loan`)

**Encoding**
- Used `pd.get_dummies(...)` to one-hot encode categorical features.

---

## How to run / reproduce

> Note: The notebooks expect the competition datasets to be available locally as:
> - `train2.csv`
> - `test2.csv`

1. Put the dataset files in the same folder as the notebooks.
2. Open and run:
   - `loan att1 .ipynb` (baseline models + preprocessing exploration), or
   - `besttONE_0.819444.ipynb` (XGBoost solution + submission generation)
3. The notebook will generate:
   - `bestCheck.csv` (final submission)

---

## Output

- **bestCheck.csv**  
  Format:
  - `Loan_ID`
  - `Loan_Status` (Y or N)

This file is ready to upload directly as a competition submission.

---

## Tech stack

- Python
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib (basic plots)

---
