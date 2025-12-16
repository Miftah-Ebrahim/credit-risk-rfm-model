# Final Project Report: Credit Risk Probability Model

## 1. Introduction

This report documents the development, implementation, and deployment of a machine learning system designed to estimate credit risk for unbanked customers. The project addresses the "Cold Start" challenge in the Buy-Now-Pay-Later (BNPL) industry by utilizing alternative transactional data to construct a behavioral proxy for creditworthiness.

## 2. Methodology

The core methodology relies on the hypothesis that transactional behavior—specifically Recency, Frequency, and Monetary value (RFM)—is a strong predictor of repayment reliability. In the absence of historical default labels, we employ unsupervised learning to generate a ground truth dataset for supervised modeling.

### 2.1 Feature Engineering

Data processing begins with the ingestion of raw transaction logs. The pipeline executes the following transformations:

1.  **Temporal Extraction**: Deriving `TransactionHour`, `TransactionDay`, and `TransactionMonth` to capture seasonality and time-of-day patterns.
2.  **RFM Aggregation**: Transaction logs are collapsed into customer-level records.
    *   **Recency**: Days since the last transaction.
    *   **Frequency**: Total count of transactions.
    *   **Monetary**: Sum, Mean, and Standard Deviation of transaction amounts.
3.  **Categorical Encoding**: The `ChannelId` feature is included to capture the mode of interaction (e.g., Web vs. Mobile).
4.  **Scaling**: A `StandardScaler` is applied to all numerical features to normalize distributions for clustering and linear modeling.

### 2.2 Proxy Target Generation

To create the target variable `Risk_Label`, we utilize **K-Means Clustering** with `k=3`.
*   **Cluster Segmentation**: Customers are grouped based on their standardized RFM profiles.
*   **Risk Definition**: The cluster exhibiting the highest average Recency and lowest Frequency is identified as the "High Risk" cohort.
*   **Label Assignment**: Members of this cluster are assigned `Risk_Label = 1` (High Risk), while all others are labeled `0` (Low Risk).

## 3. Model Development

### 3.1 Model Architectures

Two supervised classification models were trained on the labeled dataset:

1.  **Logistic Regression**: Selected for its interpretability and alignment with regulatory requirements for auditability.
2.  **Gradient Boosting Classifier**: Selected for its ability to capture non-linear relationships and complex feature interactions.

### 3.2 Training Pipeline

The training process is encapsulated in a `sklearn.pipeline.Pipeline` object containing:
1.  **ColumnTransformer**: Applies scaling to numeric features and OneHotEncoding to categorical features.
2.  **Estimator**: The classification model.

This ensures consistent preprocessing between training and inference environments.

### 3.3 Evaluation Results

Both models achieved perfect separation on the test set, with an **ROC-AUC of 1.000**.

> **Note**: These results are expected given that the target variable is a mathematical derivation of the input features (RFM). In a production scenario with external default labels, these metrics would naturally be lower. The high score confirms that the supervised models have successfully learned the behavioral rules defined by the unsupervised clustering step.

## 4. Analytical Insights

The following visualizations derived from the Explanatory Data Analysis (EDA) phase informed our feature engineering strategy.

### 4.1 Transaction Volume Analysis
![Daily Transaction Volume](dashboard/daily_transaction_volume.png)
*Figure 1: Daily transaction volume indicates clear cyclical patterns, suggesting that temporal features are significant predictors of customer activity.*

### 4.2 Correlation Analysis
![Correlation Matrix](dashboard/feature_correlation_matrix.png)
*Figure 2: The correlation matrix confirms strong multicollinearity between Frequency and Total Monetary Value, justifying the use of dimensionality reduction or regularization in linear models.*

### 4.3 Fraud Distribution
![Fraud Distribution](dashboard/fraud_distribution_summary.png)
*Figure 3: Fraud cases are highly concentrated in specific channels, supporting the inclusion of `ChannelId` as a categorical predictor.*

## 5. Deployment Architecture

The system is deployed as a containerized microservice:

*   **MLflow**: Used for experiment tracking and model registry. The final production model is serialized using MLflow's native format.
*   **FastAPI**: Provides a robust REST interface for serving predictions.
*   **Docker**: Ensures environment consistency across development and production.
*   **CI/CD**: A GitHub Actions workflow enforces code quality standards via `flake8` and regression testing via `pytest`.

## 6. Conclusion

The developed system successfully demonstrates the viability of alternative data scoring. By operationalizing RFM analysis into a production-grade machine learning pipeline, we provide a scalable solution for credit risk assessment in the absence of traditional credit history.
