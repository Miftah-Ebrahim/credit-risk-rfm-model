# Credit Risk Probability Model for Alternative Data

## 1. Project Overview

This repository contains an end-to-end Machine Learning solution designed to assess creditworthiness for unbanked populations using alternative transactional data. The system addresses the "Cold Start" problem in Buy-Now-Pay-Later (BNPL) services by leveraging behavioral analytics rather than traditional credit bureau histories.

The pipeline ingests raw transaction logs, constructs a behavioral profile using Recency, Frequency, and Monetary (RFM) analysis, and employs unsupervised learning to generate proxy risk labels. These labels are then used to train supervised classification models that predict the probability of default or high-risk behavior.

The final deliverables include a reproducible training pipeline, a ModelOps-compliant tracking system using MLflow, and a containerized REST API for real-time inference.

### Repository Structure

```
.
├── api
│   ├── main.py             # FastAPI entry point
│   └── schemas.py          # Pydantic data models
├── dashboard               # Analytical visualizations
│   ├── daily_transaction_volume.png
│   ├── feature_correlation_matrix.png
│   └── ...
├── data
│   ├── raw                 # Input transaction logs
│   └── processed           # Feature-engineered datasets
├── docker
│   ├── Dockerfile          # API container definition
│   └── docker-compose.yml  # Service orchestration
├── models
│   └── production_model    # MLflow-native model artifact
├── notebooks
│   └── eda.ipynb           # Exploratory Data Analysis
├── src
│   ├── data_processing.py  # ETL and pipeline orchestration
│   ├── features.py         # Feature engineering logic
│   ├── predict.py          # Inference engine
│   └── train.py            # Training and evaluation script
├── tests
│   └── test_data_processing.py
└── README.md
```

---

## 2. Credit Scoring Business Understanding

### a) Basel II Accord Discussion

The Basel II Capital Accord establishes international standards for banking regulators to control how much capital banks need to set aside to guard against financial and operational risks. A core component of Basel II is the **Internal Ratings-Based (IRB)** approach, which allows institutions to use their own estimated risk parameters for capital calculation.

This project aligns with Basel II principles by emphasizing **Risk Measurement** through rigorous statistical modeling. The regulatory framework mandates that credit risk models must be transparency, auditable, and interpretable.

*   **Risk Measurement**: We utilize **Weight of Evidence (WoE)** and **Information Value (IV)** analysis to assess the predictive power of each variable, ensuring that only statistically significant behavioral drivers are included in the model.
*   **Auditability**: The use of **Logistic Regression** provides clear feature coefficients, allowing risk officers to explain exactly why a specific applicant was rejected or approved. This transparency is critical for regulatory compliance and fair lending practices.
*   **Pipeline Documentation**: The structured code, MLflow tracking, and comprehensive documentation ensure that every step of the model development lifecycle is reproducible and auditable.

### b) Proxy Variable Justification

In emerging markets and BNPL sectors, a significant portion of the customer base lacks a formal credit history or a recorded "Default" event. Without a direct target variable, supervised learning is impossible. Therefore, a **Proxy Target** is required.

We employ a behavioral proxy derived from **RFM clustering**:
*   **Recency (R)**: Time since the last transaction. Long gaps may indicate dormancy or churn.
*   **Frequency (F)**: Number of transactions. High frequency often correlates with engagement and reliability.
*   **Monetary (M)**: Total spend. Higher value transactions can indicate stronger financial capacity.

**Risks and Implications**:
*   **Misclassification Risk**: A low-frequency user might be financially stable but simply inactive. Grouping them with "High Risk" users is a necessary conservative assumption in the absence of external data.
*   **Bias Risk**: The model optimizes for transaction velocity. This may bias against casual users who transact infrequently but reliably.
*   **Regulatory Implications**: Using unsupervised proxies requires careful validation. The resulting risk tiers must be monitored to ensure they do not systematically discriminate against specific demographic groups.

### c) Model Trade-offs in Regulated Finance

We implemented and compared two distinct modeling approaches: **Logistic Regression** and **Gradient Boosting**.

| Feature | Logistic Regression | Gradient Boosting (GBM) |
| :--- | :--- | :--- |
| **Interpretability** | Global and Local. Coefficients are directly translatable to scorecards (Points-based systems). | Complex. Requires SHAP/LIME for post-hoc explanation. "Black Box" nature. |
| **Performance** | Captures linear relationships well. May underfit complex non-linear patterns. | Captures complex, non-linear interactions. Often achieves higher predictive accuracy. |
| **Regulatory Acceptance** | **High**. Standard in banking for decades. Easy to validate. | **Moderate**. Requires rigorous documentation and explainability layers to be accepted. |

**Conclusion**: While Gradient Boosting offers marginal performance gains, **Logistic Regression** (or simple linear models with WoE) is often preferred in highly regulated environments. It minimizes the risk of "black swan" failures and simplifies the conversation with internal auditors and external regulators. In this project, we retained both to demonstrate the trade-off, with the production pipeline capable of serving either.

---

## 3. Technical Architecture

### Feature Engineering
The system extracts temporal features (Transaction Hour, Day, Month) and aggregates transactional metrics at the customer level. A `ColumnTransformer` pipeline applies **StandardScaler** to numerical features and **OneHotEncoder** to categorical variables such as `ChannelId`, ensuring robust handling of data distributions.

### Proxy Target Creation
We utilize **K-Means Clustering (k=3)** to segment customers. The cluster exhibiting the highest average Recency (indicating inactivity) is heuristically labeled as "High Risk". This creating a binary target variable `Risk_Label` for supervised training.

### Model Training & MLflow
The training script utilizes `sklearn.pipeline.Pipeline` to encapsulate preprocessing and modeling steps, preventing data leakage. Experiments are tracked using **MLflow**, which logs:
*   Hyperparameters (e.g., regularization strength C, tree depth).
*   Metrics (AUC, Accuracy, F1-Score, Precision, Recall).
*   Serialized Model Artifacts (for reproducibility).

### Deployment
The best-performing model is deployed as a REST API using **FastAPI**. The application loads the model artifact directly from the MLflow registry location and serves predictions via the `/predict` endpoint. Input data validation is enforced using **Pydantic** schemas.

### CI/CD
A GitHub Actions workflow (`ci.yml`) is implemented to enforce code quality. It triggers on every push to the `main` branch, running:
1.  **Linters**: `flake8` to ensure PEP-8 compliance.
2.  **Unit Tests**: `pytest` to validate feature engineering and pipeline logic.

---

## 4. Dashboard & Visual Outputs

### Transaction Analysis
The following visualizations provide insights into the underlying data distributions and transaction patterns.

#### Daily Transaction Volume
![Daily Transaction Volume](dashboard/daily_transaction_volume.png)
*Fig 1: Analysis of transaction frequency over time to identify seasonality and trends.*

#### Feature Correlations
![Correlation Matrix](dashboard/feature_correlation_matrix.png)
*Fig 2: Heatmap showing the correlation between derived RFM features and the transaction attributes.*

#### Fraud Distribution
![Fraud Distribution](dashboard/fraud_distribution_summary.png)
*Fig 3: Distribution of reported fraud cases across different categories, highlighting potential risk areas.*

---

## 5. Author

*   **Name:** Mifta Y
*   **LinkedIn:** [https://www.linkedin.com/in/miftah-ebrahim-b422b3364/](https://www.linkedin.com/in/miftah-ebrahim-b422b3364/)
*   **Telegram:** Miftah_deva
