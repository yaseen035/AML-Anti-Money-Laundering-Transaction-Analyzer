# AML - Anti-Money Laundering Transaction Analyzer

A university project that uses the Elliptic Bitcoin transaction dataset to demonstrate transaction data management, SQL analysis, statistical analysis, machine learning, risk scoring, and an interactive Streamlit dashboard.

**Repository:** https://github.com/yaseen035/AML-Anti-Money-Laundering-Transaction-Analyzer  
**Database:** SQLite  
**Language:** Python  
**Dashboard:** Streamlit + Plotly

> **Research prototype:** This project analyses an anonymized Bitcoin transaction graph. It is not a production banking AML system, does not contain real bank-customer records, and does not determine whether a person has committed a crime. Model-generated alerts are for human review only.

## 1. Project features

- SQLite database with transaction, edge, prediction, risk, alert, and investigation tables.
- Referential integrity, `CHECK` constraints, unique constraints, and indexes.
- SQL examples for filtering, grouping, `HAVING`, subqueries, CTEs, window functions, views, triggers, and `EXPLAIN QUERY PLAN`.
- Statistical analysis of the anonymized numerical features.
- Three machine-learning approaches: Logistic Regression, Random Forest, and Isolation Forest.
- A 0-100 demonstration risk score and high-risk alert generation.
- A Streamlit dashboard with Overview, Transaction Explorer, Risk Monitoring, AML Alerts, Statistical Analysis, and Model Evaluation pages.

## 2. Dataset

The project uses the **Elliptic Bitcoin Transaction Dataset**:

- Kaggle: <https://www.kaggle.com/datasets/ellipticco/elliptic-data-set>
- Research paper: Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics* (2019), <https://arxiv.org/abs/1908.02591>.

The commonly reported dataset contains 203,769 transactions, 234,355 directed transaction edges, and 49 time steps. Each transaction has a time-step value and 165 numerical features. The class labels are `1` = illicit, `2` = licit, and `unknown` = unlabelled. The commonly reported class counts are 4,545 illicit, 42,019 licit, and 157,205 unknown transactions.

The features are anonymized. Do not assign business meanings such as “account balance” or “transaction amount” to feature columns unless a reliable source defines them.

### Dataset licence and redistribution

The Kaggle dataset page lists **CC BY-NC-ND 4.0**. Read the current licence and dataset terms before reuse. This repository intentionally excludes the raw dataset, processed copies, local database, trained model files, and generated results. Do not upload the CSV dataset files or publish derived materials unless the licence permits it.

## 3. Repository structure

```text
AML-Anti-Money-Laundering-Transaction-Analyzer/
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── database/
│   ├── schema.sql              # SQLite table definitions and indexes
│   ├── queries.sql             # SQL examples (also written by the pipeline)
│   ├── views.sql               # Generated SQLite views
│   └── triggers.sql            # Generated SQLite triggers
├── data/
│   ├── raw/elliptic/            # Put the three source CSV files here (not committed)
│   └── database/                # Local SQLite database (not committed)
├── models/                      # Trained .joblib models (not committed)
├── notebooks/                   # Jupyter exploration, if used
├── outputs/                     # Metrics and analysis output (not committed)
├── src/
│   ├── run_pipeline.py          # Database import, training, scoring, alerts
│   └── statistical_analysis.py   # Feature statistics and correlations
├── requirements.txt
└── README.md
```

Some folders and SQL files are created automatically when the pipeline runs. The `.gitignore` file excludes local data, database files, model artefacts, generated outputs, credentials, and Python cache files.

## 4. Requirements

- Python 3.10 or later is recommended.
- Internet access to obtain the source dataset and Python packages.
- Enough free storage for the dataset, SQLite database, model files, and temporary processing files. Keep at least 2 GB free where possible.
- A system with limited RAM should run the pipeline without other heavy applications open. The script reads feature CSV files in chunks and caps the training sample at 30,000 rows.

The project can run on Linux or Debian in Termux. The optional `sqlite3` command-line program is useful for database inspection; Python's built-in `sqlite3` module is used by the application.

## 5. Installation

### 5.1 Clone the repository

```bash
git clone https://github.com/yaseen035/AML-Anti-Money-Laundering-Transaction-Analyzer.git
cd AML-Anti-Money-Laundering-Transaction-Analyzer
```

### 5.2 Create and activate a virtual environment

Linux / Debian / Termux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If `python3 -m venv` is not available on Debian, install the venv package first:

```bash
sudo apt update
sudo apt install -y python3-venv
```

### 5.3 Install dependencies

```bash
pip install -r requirements.txt
```

Optional Jupyter support:

```bash
pip install jupyterlab ipykernel
```

Optional SQLite command-line client on Debian:

```bash
sudo apt install -y sqlite3
```

## 6. Download and place the dataset

Download the Elliptic dataset from the Kaggle page linked above. Extract the three CSV files into:

```text
data/raw/elliptic/
```

The required filenames are:

```text
elliptic_txs_features.csv
elliptic_txs_classes.csv
elliptic_txs_edgelist.csv
```

Create the folder if it does not exist:

```bash
mkdir -p data/raw/elliptic
```

Verify that all three files are available:

```bash
find data/raw/elliptic -maxdepth 1 -type f -name 'elliptic_txs_*.csv' -print
```

The pipeline searches recursively inside the project folder for these exact filenames, but placing them in `data/raw/elliptic/` keeps the project organized.

**Do not put `kaggle.json` inside the repository.** If you use the Kaggle CLI, keep its credential file in your home directory and protect it with restrictive permissions. Manual download from Kaggle is also supported.

## 7. Run the complete pipeline

Run these commands from the repository root, with the virtual environment active:

### 7.1 Import data, create database objects, train models, and generate alerts

```bash
python src/run_pipeline.py
```

The script performs the following operations:

1. Locates the three Elliptic CSV files.
2. Creates the SQLite database at `data/database/aml_analyzer.db`.
3. Creates the seven database tables and indexes.
4. Imports transaction metadata and valid transaction edges in chunks.
5. Excludes `unknown` labels from supervised model training and evaluation.
6. Uses a chronological split where possible: time steps 1-34 for training and 35-49 for testing.
7. Uses a stratified random fallback split if the chronological partitions do not each contain both labelled classes. If this fallback is used, the evaluation is no longer strictly chronological; check the terminal warning and report it.
8. Caps the training set at 30,000 records to limit memory use.
9. Trains Logistic Regression, Random Forest, and Isolation Forest models.
10. Evaluates the models and writes metrics to `outputs/model_metrics.json`.
11. Scores all transactions, creates risk levels, and records high-risk alerts.
12. Writes SQL schema, view, trigger, and query files into `database/`.

### 7.2 Run statistical analysis

```bash
python src/statistical_analysis.py
```

This performs chunk-based calculations over the numerical features and writes:

- `outputs/feature_statistics.csv`
- `outputs/feature_correlation.csv`

The statistics include mean, population variance, standard deviation, minimum, maximum, and pairwise feature correlations. Feature names remain anonymous.

### 7.3 Start the dashboard

```bash
streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

Open the dashboard in a browser at:

```text
http://127.0.0.1:8501
```

On a different device on the same network, use the host device's LAN IP address with port `8501`. Keep the Streamlit process running while using the dashboard.

## 8. Generated files

After a successful run, the main outputs include:

```text
models/
├── logistic_regression.joblib
├── random_forest.joblib
└── isolation_forest.joblib

data/database/
└── aml_analyzer.db

outputs/
├── model_metrics.json
├── class_distribution.csv
├── time_step_summary.csv
├── risk_distribution.csv
├── top_risk_transactions.csv
├── aml_alerts.csv
├── feature_statistics.csv
└── feature_correlation.csv
```

The `.gitignore` rules keep these local files out of Git. Generate them again on the machine where you run the project.

## 9. Database inspection and SQL examples

If the SQLite command-line client is installed, inspect the database:

```bash
sqlite3 data/database/aml_analyzer.db
```

Inside the SQLite prompt:

```sql
.tables

SELECT class_label, COUNT(*) AS total_transactions
FROM transactions
GROUP BY class_label;

SELECT risk_level, COUNT(*) AS total
FROM risk_scores
GROUP BY risk_level;

PRAGMA integrity_check;

.exit
```

Run the saved SQL examples:

```bash
sqlite3 -header -column data/database/aml_analyzer.db < database/queries.sql
```

The SQL examples demonstrate `SELECT`, `WHERE`, `ORDER BY`, `GROUP BY`, `HAVING`, subqueries, CTEs, window functions, views, and `EXPLAIN QUERY PLAN`. **No SQL JOIN statements are used**, in accordance with the project requirement.

## 10. Database design

The SQLite database contains these tables:

1. `transactions` - transaction ID, time step, and known/unknown class label.
2. `transaction_edges` - directed relationships between transactions.
3. `model_predictions` - model name, predicted class, and illicit-class probability where available.
4. `risk_scores` - Random Forest risk score and risk band.
5. `aml_alerts` - high-risk alerts and review status.
6. `investigation_cases` - investigation case records associated with alerts.
7. `investigation_history` - audit-style history of case creation and status changes.

The pipeline also creates three views (`v_class_summary`, `v_time_step_summary`, `v_high_risk_transactions`) and two triggers for investigation case history. Indexes are created for frequently filtered or referenced columns.

### SQLite feature note

SQLite does not provide server-style `CREATE PROCEDURE`, `CREATE FUNCTION`, or `GRANT`/`REVOKE` user-role statements. The project's Python functions perform application-level operations; SQLite triggers provide automated database actions. File-system permissions and application controls are used for local access protection. Do not describe Python functions as SQL stored procedures.

## 11. Machine-learning methodology

- **Logistic Regression:** baseline supervised classifier with median imputation, standardization, and class weighting.
- **Random Forest:** supervised tree ensemble with class-weighted training; its predicted illicit-class probability is used for the demonstration risk score.
- **Isolation Forest:** unsupervised anomaly detector fitted on the licit portion of the training data; its output is an anomaly flag, not an illicit probability.

Evaluation includes accuracy, precision, recall, F1-score, and confusion matrices. ROC-AUC and PR-AUC are calculated for the supervised probability models when both classes are present in the test set. Because the labelled data is imbalanced, interpret precision, recall, F1, PR-AUC, and the confusion matrix rather than relying on accuracy alone.

The Isolation Forest classification measures are based on its anomaly flag and the configured contamination setting. They are not directly equivalent to calibrated supervised probabilities.

## 12. Risk score and responsible use

The demonstration risk score is:

```text
risk_score = Random Forest P(illicit) * 100
```

The configured bands are:

- Low: below 40
- Medium: 40 to below 70
- High: 70 or above

A high band creates an alert. These thresholds are project demonstration settings, not official AML rules, regulatory thresholds, or legal conclusions. A high score is not evidence that a transaction or person is criminal. Human review and independent evidence are required.

## 13. Important rerun warning

`src/run_pipeline.py` rebuilds the contents of its project tables at the start of a run. It deletes existing rows from the transaction, edge, prediction, risk, alert, case, and case-history tables before importing the dataset again. **Back up the SQLite database before rerunning the pipeline if you need to keep investigation records.**

Example backup:

```bash
cp data/database/aml_analyzer.db data/database/aml_analyzer_backup.db
```

## 14. Troubleshooting

**Dataset file not found**

- Check the exact three filenames.
- Confirm they are extracted, not still inside a ZIP file.
- Put them in `data/raw/elliptic/`.
- Run `find data/raw/elliptic -type f` to inspect the folder.

**ModuleNotFoundError**

- Activate the virtual environment.
- Run `pip install -r requirements.txt` again.

**Dashboard says database not found**

- Run `python src/run_pipeline.py` successfully before starting Streamlit.
- Start Streamlit from the repository root.

**Memory pressure on a phone**

- Close other applications.
- Keep the chunk size and training limit unchanged initially.
- Do not run model training and other large applications simultaneously.
- Ensure enough free storage is available for the database and model files.

**SQLite database or investigation rows disappeared after rerunning**

- This is expected from the current pipeline reset logic. Restore the backup if one was made.

## 15. Limitations and future work

This is an academic proof of concept. The dataset contains anonymized Bitcoin transaction data, not bank customer histories. Most records are unlabelled, the labelled classes are imbalanced, and the feature names are anonymized. The current system does not implement production user authentication, a regulatory case-management workflow, a calibrated threshold policy, graph neural networks, or a live blockchain feed. Future work could add temporal cross-validation, graph-aware models, explainability, analyst feedback, and additional legally usable datasets.

## References

1. Elliptic. *Elliptic Data Set*. Kaggle. <https://www.kaggle.com/datasets/ellipticco/elliptic-data-set>.
2. M. Weber, G. Domeniconi, J. Chen, D. K. I. Weidele, C. Bellei, T. Robinson, and C. E. Leiserson. “Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.” 2nd KDD Workshop on Anomaly Detection in Finance, 2019. <https://arxiv.org/abs/1908.02591>.
3. Python documentation: <https://docs.python.org/3/>.
4. SQLite documentation: <https://www.sqlite.org/docs.html>.
5. scikit-learn documentation: <https://scikit-learn.org/stable/>.
6. Streamlit documentation: <https://docs.streamlit.io/>.

## Academic project note

Prepared for the Introduction to Database course, Department of Computational Mathematics, University of Karachi.
