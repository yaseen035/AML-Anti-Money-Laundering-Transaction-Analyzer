from pathlib import Path
import sqlite3
import json
import warnings
import sys

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "elliptic"
DB_DIR = ROOT / "data" / "database"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

DB_PATH = DB_DIR / "aml_analyzer.db"

CHUNK_SIZE = 15000
MAX_TRAIN_ROWS = 30000
RANDOM_STATE = 42

for folder in [DB_DIR, PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def find_dataset_files():
    print("\n[1/8] Searching for Elliptic dataset files...")

    files = list(ROOT.rglob("elliptic_txs_*.csv"))

    feature = next(
        (p for p in files if p.name == "elliptic_txs_features.csv"),
        None
    )
    classes = next(
        (p for p in files if p.name == "elliptic_txs_classes.csv"),
        None
    )
    edges = next(
        (p for p in files if p.name == "elliptic_txs_edgelist.csv"),
        None
    )

    missing = []
    if feature is None:
        missing.append("elliptic_txs_features.csv")
    if classes is None:
        missing.append("elliptic_txs_classes.csv")
    if edges is None:
        missing.append("elliptic_txs_edgelist.csv")

    if missing:
        print("\nERROR: Missing dataset files:")
        for name in missing:
            print(" -", name)
        print("\nPut all three CSV files somewhere inside:")
        print(ROOT)
        sys.exit(1)

    print("Features:", feature)
    print("Classes: ", classes)
    print("Edges:   ", edges)

    return feature, classes, edges


def normalize_label(value):
    value = str(value).strip().lower()

    if value in ("1", "1.0", "illicit"):
        return "illicit"

    if value in ("2", "2.0", "licit"):
        return "licit"

    return "unknown"


def read_two_column_csv(path, expected_columns):
    df = pd.read_csv(path, dtype=str)
    lower_names = {str(c).strip().lower(): c for c in df.columns}

    required = [name.lower() for name in expected_columns]

    if all(name in lower_names for name in required):
        df = df[
            [lower_names[name] for name in required]
        ].copy()
        df.columns = expected_columns
        return df

    df = pd.read_csv(
        path,
        header=None,
        names=expected_columns,
        dtype=str
    )

    return df


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    time_step INTEGER NOT NULL,
    class_label TEXT NOT NULL
        CHECK (class_label IN ('licit', 'illicit', 'unknown')),
    feature_file TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transaction_edges (
    edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_transaction TEXT NOT NULL,
    target_transaction TEXT NOT NULL,
    UNIQUE(source_transaction, target_transaction),
    FOREIGN KEY(source_transaction)
        REFERENCES transactions(transaction_id),
    FOREIGN KEY(target_transaction)
        REFERENCES transactions(transaction_id)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,
    predicted_class TEXT NOT NULL,
    illicit_probability REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(transaction_id)
        REFERENCES transactions(transaction_id)
);

CREATE TABLE IF NOT EXISTS risk_scores (
    risk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL UNIQUE,
    risk_score REAL NOT NULL
        CHECK(risk_score BETWEEN 0 AND 100),
    risk_level TEXT NOT NULL
        CHECK(risk_level IN ('low', 'medium', 'high')),
    calculated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(transaction_id)
        REFERENCES transactions(transaction_id)
);

CREATE TABLE IF NOT EXISTS aml_alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'open'
        CHECK(status IN ('open', 'under_review', 'closed')),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(transaction_id)
        REFERENCES transactions(transaction_id)
);

CREATE TABLE IF NOT EXISTS investigation_cases (
    case_id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id INTEGER NOT NULL,
    investigator TEXT,
    findings TEXT,
    status TEXT DEFAULT 'pending'
        CHECK(status IN ('pending', 'investigating', 'resolved')),
    opened_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(alert_id)
        REFERENCES aml_alerts(alert_id)
);

CREATE TABLE IF NOT EXISTS investigation_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    notes TEXT,
    action_time TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(case_id)
        REFERENCES investigation_cases(case_id)
);

CREATE INDEX IF NOT EXISTS idx_transactions_timestep
ON transactions(time_step);

CREATE INDEX IF NOT EXISTS idx_transactions_class
ON transactions(class_label);

CREATE INDEX IF NOT EXISTS idx_edges_source
ON transaction_edges(source_transaction);

CREATE INDEX IF NOT EXISTS idx_predictions_transaction
ON model_predictions(transaction_id);

CREATE INDEX IF NOT EXISTS idx_alerts_status
ON aml_alerts(status);

CREATE INDEX IF NOT EXISTS idx_risk_score
ON risk_scores(risk_score);
"""

VIEWS = """
DROP VIEW IF EXISTS v_class_summary;
DROP VIEW IF EXISTS v_time_step_summary;
DROP VIEW IF EXISTS v_high_risk_transactions;

CREATE VIEW v_class_summary AS
SELECT class_label, COUNT(*) AS total_transactions
FROM transactions
GROUP BY class_label;

CREATE VIEW v_time_step_summary AS
SELECT
    time_step,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN class_label = 'illicit' THEN 1 ELSE 0 END)
        AS illicit_transactions,
    SUM(CASE WHEN class_label = 'licit' THEN 1 ELSE 0 END)
        AS licit_transactions
FROM transactions
GROUP BY time_step;

CREATE VIEW v_high_risk_transactions AS
SELECT
    transaction_id,
    risk_score,
    risk_level,
    calculated_at
FROM risk_scores
WHERE risk_score >= 70;
"""

TRIGGERS = """
DROP TRIGGER IF EXISTS trg_case_created;
DROP TRIGGER IF EXISTS trg_case_status_changed;

CREATE TRIGGER trg_case_created
AFTER INSERT ON investigation_cases
BEGIN
    INSERT INTO investigation_history(case_id, action, notes)
    VALUES (
        NEW.case_id,
        'case_opened',
        'Investigation case automatically recorded'
    );
END;

CREATE TRIGGER trg_case_status_changed
AFTER UPDATE OF status ON investigation_cases
WHEN OLD.status <> NEW.status
BEGIN
    INSERT INTO investigation_history(case_id, action, notes)
    VALUES (
        NEW.case_id,
        'status_changed',
        'Status changed from ' || OLD.status || ' to ' || NEW.status
    );
END;
"""

QUERIES = """
-- 1. Display illicit transactions
SELECT *
FROM transactions
WHERE class_label = 'illicit'
LIMIT 20;

-- 2. Count transactions by classification
SELECT class_label, COUNT(*) AS total
FROM transactions
GROUP BY class_label;

-- 3. Find time steps with more than 100 illicit transactions
SELECT time_step, COUNT(*) AS total
FROM transactions
WHERE class_label = 'illicit'
GROUP BY time_step
HAVING COUNT(*) > 100;

-- 4. Find transactions with above-average risk scores
SELECT transaction_id, risk_score
FROM risk_scores
WHERE risk_score > (
    SELECT AVG(risk_score)
    FROM risk_scores
)
ORDER BY risk_score DESC
LIMIT 20;

-- 5. Rank transactions by risk within each risk level
SELECT
    transaction_id,
    risk_score,
    risk_level,
    RANK() OVER (
        PARTITION BY risk_level
        ORDER BY risk_score DESC
    ) AS risk_rank
FROM risk_scores
LIMIT 100;

-- 6. Use a CTE to count alerts by status
WITH alert_summary AS (
    SELECT status, COUNT(*) AS total
    FROM aml_alerts
    GROUP BY status
)
SELECT *
FROM alert_summary;

-- 7. List the highest-risk transactions
SELECT *
FROM v_high_risk_transactions
ORDER BY risk_score DESC
LIMIT 50;

-- 8. Examine query optimization
EXPLAIN QUERY PLAN
SELECT *
FROM transactions
WHERE class_label = 'illicit';
"""


def setup_database():
    print("\n[2/8] Creating SQLite database and SQL objects...")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")

        conn.executescript(SCHEMA)

        # Clear old project results so reruns do not duplicate data.
        for table in [
            "investigation_history",
            "investigation_cases",
            "aml_alerts",
            "risk_scores",
            "model_predictions",
            "transaction_edges",
            "transactions"
        ]:
            conn.execute(f"DELETE FROM {table}")

        conn.executescript(VIEWS)
        conn.executescript(TRIGGERS)
        conn.commit()

    (ROOT / "database" / "schema.sql").write_text(
        SCHEMA, encoding="utf-8"
    )
    (ROOT / "database" / "views.sql").write_text(
        VIEWS, encoding="utf-8"
    )
    (ROOT / "database" / "triggers.sql").write_text(
        TRIGGERS, encoding="utf-8"
    )
    (ROOT / "database" / "queries.sql").write_text(
        QUERIES, encoding="utf-8"
    )

    erd = """
erDiagram
    TRANSACTIONS ||--o{ TRANSACTION_EDGES : source
    TRANSACTIONS ||--o{ MODEL_PREDICTIONS : receives
    TRANSACTIONS ||--o| RISK_SCORES : has
    TRANSACTIONS ||--o{ AML_ALERTS : generates
    AML_ALERTS ||--o{ INVESTIGATION_CASES : creates
    INVESTIGATION_CASES ||--o{ INVESTIGATION_HISTORY : records
"""
    (ROOT / "database" / "erd.mmd").write_text(
        erd, encoding="utf-8"
    )


def import_transactions(feature_path, classes_path):
    print("\n[3/8] Importing transactions in chunks...")

    class_df = read_two_column_csv(
        classes_path, ["txId", "class"]
    )

    class_map = {
        str(tx).strip(): normalize_label(label)
        for tx, label in zip(class_df["txId"], class_df["class"])
    }

    print("Class records loaded:", len(class_map))

    all_ids = set()
    train_X_parts = []
    train_y_parts = []
    test_X_parts = []
    test_y_parts = []

    total_rows = 0

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")

        reader = pd.read_csv(
            feature_path,
            header=None,
            dtype={0: str},
            chunksize=CHUNK_SIZE,
            low_memory=False
        )

        for chunk_number, chunk in enumerate(reader, start=1):
            if chunk.shape[1] < 3:
                raise ValueError(
                    "Feature CSV does not contain the expected columns."
                )

            ids = chunk.iloc[:, 0].astype(str).str.strip().to_numpy()

            steps = pd.to_numeric(
                chunk.iloc[:, 1], errors="coerce"
            ).fillna(0).astype(int).to_numpy()

            numeric = chunk.iloc[:, 2:].apply(
                pd.to_numeric, errors="coerce"
            )

            X = numeric.replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0).to_numpy(dtype=np.float32)

            labels = np.array([
                class_map.get(tx, "unknown") for tx in ids
            ])

            rows = [
                (tx, int(step), label, str(feature_path))
                for tx, step, label in zip(ids, steps, labels)
            ]

            conn.executemany(
                """
                INSERT OR REPLACE INTO transactions
                (transaction_id, time_step, class_label, feature_file)
                VALUES (?, ?, ?, ?)
                """,
                rows
            )

            all_ids.update(ids.tolist())
            total_rows += len(ids)

            known = np.isin(labels, ["licit", "illicit"])

            # Temporal split: earlier time steps train the models;
            # later labelled time steps are used for evaluation.
            train_mask = known & (steps <= 34)
            test_mask = known & (steps > 34)

            if train_mask.any():
                train_X_parts.append(X[train_mask])
                train_y_parts.append(
                    (labels[train_mask] == "illicit").astype(np.int8)
                )

            if test_mask.any():
                test_X_parts.append(X[test_mask])
                test_y_parts.append(
                    (labels[test_mask] == "illicit").astype(np.int8)
                )

            print(
                f"  Chunk {chunk_number}: "
                f"{total_rows:,} transactions processed"
            )

            conn.commit()

    print("Total transaction rows:", f"{total_rows:,}")
    print("Unique transaction IDs:", f"{len(all_ids):,}")

    if not train_X_parts:
        raise RuntimeError("No labelled training records were found.")

    X_train = np.concatenate(train_X_parts, axis=0)
    y_train = np.concatenate(train_y_parts)

    X_test = (
        np.concatenate(test_X_parts, axis=0)
        if test_X_parts else np.empty((0, X_train.shape[1]), dtype=np.float32)
    )
    y_test = (
        np.concatenate(test_y_parts)
        if test_y_parts else np.array([], dtype=np.int8)
    )

    if (
        len(np.unique(y_train)) < 2
        or len(np.unique(y_test)) < 2
    ):
        print(
            "WARNING: Temporal split did not contain both classes. "
            "Using a stratified fallback split."
        )

        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)

        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y_all
        )

    # Limit training memory and keep both classes represented.
    if len(X_train) > MAX_TRAIN_ROWS:
        rng = np.random.default_rng(RANDOM_STATE)
        selected = []

        per_class_limit = MAX_TRAIN_ROWS // 2

        for cls in [0, 1]:
            class_indices = np.flatnonzero(y_train == cls)
            take = min(len(class_indices), per_class_limit)
            if take:
                selected.extend(
                    rng.choice(
                        class_indices,
                        size=take,
                        replace=False
                    ).tolist()
                )

        if len(selected) < MAX_TRAIN_ROWS:
            remaining = np.setdiff1d(
                np.arange(len(y_train)),
                np.array(selected, dtype=int)
            )
            extra = min(
                MAX_TRAIN_ROWS - len(selected),
                len(remaining)
            )
            if extra:
                selected.extend(
                    rng.choice(
                        remaining,
                        size=extra,
                        replace=False
                    ).tolist()
                )

        selected = np.array(selected, dtype=int)
        X_train = X_train[selected]
        y_train = y_train[selected]

    print("Training rows:", f"{len(X_train):,}")
    print("Testing rows:", f"{len(X_test):,}")
    print("Feature count:", X_train.shape[1])

    if X_train.shape[1] != 165:
        print(
            "WARNING: Expected 165 numerical features; "
            f"found {X_train.shape[1]}."
        )

    return all_ids, X_train, y_train, X_test, y_test


def import_edges(edge_path, valid_ids):
    print("\n[4/8] Importing transaction relationships...")

    edge_df = read_two_column_csv(
        edge_path, ["txId1", "txId2"]
    )

    total_edges = 0

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        # Process the edge list in chunks.
        for chunk in pd.read_csv(
            edge_path,
            dtype=str,
            chunksize=20000
        ):
            lower = {str(c).strip().lower(): c for c in chunk.columns}

            if "txid1" in lower and "txid2" in lower:
                source_col = lower["txid1"]
                target_col = lower["txid2"]
                sources = chunk[source_col].astype(str).str.strip()
                targets = chunk[target_col].astype(str).str.strip()
            else:
                # A no-header edge file.
                sources = chunk.iloc[:, 0].astype(str).str.strip()
                targets = chunk.iloc[:, 1].astype(str).str.strip()

            rows = [
                (s, t)
                for s, t in zip(sources, targets)
                if s in valid_ids and t in valid_ids
            ]

            if rows:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO transaction_edges
                    (source_transaction, target_transaction)
                    VALUES (?, ?)
                    """,
                    rows
                )
                total_edges += len(rows)

            conn.commit()

    print("Valid relationship rows processed:", f"{total_edges:,}")


def train_models(X_train, y_train, X_test, y_test):
    print("\n[5/8] Training machine learning models...")

    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=400,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE
        ))
    ])

    random_forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=RANDOM_STATE
        ))
    ])

    logistic.fit(X_train, y_train)
    print("Logistic Regression trained.")

    random_forest.fit(X_train, y_train)
    print("Random Forest trained.")

    # Fit anomaly detector on licit training transactions.
    licit_X = X_train[y_train == 0]

    if len(licit_X) == 0:
        licit_X = X_train

    if len(licit_X) > 10000:
        rng = np.random.default_rng(RANDOM_STATE)
        selected = rng.choice(
            len(licit_X), 10000, replace=False
        )
        licit_X = licit_X[selected]

    isolation_forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("detector", IsolationForest(
            n_estimators=100,
            contamination=0.02,
            max_samples=min(256, len(licit_X)),
            n_jobs=1,
            random_state=RANDOM_STATE
        ))
    ])

    isolation_forest.fit(licit_X)
    print("Isolation Forest trained.")

    joblib.dump(
        logistic, MODEL_DIR / "logistic_regression.joblib"
    )
    joblib.dump(
        random_forest, MODEL_DIR / "random_forest.joblib"
    )
    joblib.dump(
        isolation_forest, MODEL_DIR / "isolation_forest.joblib"
    )

    rf_probability = random_forest.predict_proba(X_test)[:, 1]
    rf_predictions = (rf_probability >= 0.5).astype(int)

    metrics = {
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
        "random_forest": {
            "accuracy": float(accuracy_score(y_test, rf_predictions)),
            "precision_illicit": float(
                precision_score(
                    y_test, rf_predictions,
                    pos_label=1, zero_division=0
                )
            ),
            "recall_illicit": float(
                recall_score(
                    y_test, rf_predictions,
                    pos_label=1, zero_division=0
                )
            ),
            "f1_illicit": float(
                f1_score(
                    y_test, rf_predictions,
                    pos_label=1, zero_division=0
                )
            ),
            "confusion_matrix": confusion_matrix(
                y_test, rf_predictions
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                rf_predictions,
                target_names=["licit", "illicit"],
                zero_division=0,
                output_dict=True
            )
        }
    }

    if len(np.unique(y_test)) == 2:
        metrics["random_forest"]["roc_auc"] = float(
            roc_auc_score(y_test, rf_probability)
        )
        metrics["random_forest"]["pr_auc"] = float(
            average_precision_score(y_test, rf_probability)
        )


    # Evaluate Logistic Regression on the same test records.
    lr_probability = logistic.predict_proba(X_test)[:, 1]
    lr_predictions = (lr_probability >= 0.5).astype(int)

    metrics["logistic_regression"] = {
        "accuracy": float(accuracy_score(y_test, lr_predictions)),
        "precision_illicit": float(
            precision_score(y_test, lr_predictions, pos_label=1, zero_division=0)
        ),
        "recall_illicit": float(
            recall_score(y_test, lr_predictions, pos_label=1, zero_division=0)
        ),
        "f1_illicit": float(
            f1_score(y_test, lr_predictions, pos_label=1, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, lr_predictions
        ).tolist()
    }

    # Treat Isolation Forest anomalies as potential illicit transactions
    # for a basic comparison against the labelled test records.
    iso_predictions = (
        isolation_forest.predict(X_test) == -1
    ).astype(int)

    metrics["isolation_forest"] = {
        "accuracy": float(accuracy_score(y_test, iso_predictions)),
        "precision_illicit": float(
            precision_score(y_test, iso_predictions, pos_label=1, zero_division=0)
        ),
        "recall_illicit": float(
            recall_score(y_test, iso_predictions, pos_label=1, zero_division=0)
        ),
        "f1_illicit": float(
            f1_score(y_test, iso_predictions, pos_label=1, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, iso_predictions
        ).tolist()
    }

    if len(np.unique(y_test)) == 2:
        metrics["logistic_regression"]["roc_auc"] = float(
            roc_auc_score(y_test, lr_probability)
        )
        metrics["logistic_regression"]["pr_auc"] = float(
            average_precision_score(y_test, lr_probability)
        )

    with open(
        OUTPUT_DIR / "model_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=4)

    print("\nRandom Forest evaluation:")
    print("Accuracy :", round(metrics["random_forest"]["accuracy"], 4))
    print("Precision:", round(metrics["random_forest"]["precision_illicit"], 4))
    print("Recall   :", round(metrics["random_forest"]["recall_illicit"], 4))
    print("F1-score :", round(metrics["random_forest"]["f1_illicit"], 4))

    return random_forest, isolation_forest


def calculate_risk(probability):
    score = float(probability) * 100

    if score >= 70:
        level = "high"
    elif score >= 40:
        level = "medium"
    else:
        level = "low"

    return round(score, 4), level


def generate_predictions(feature_path, random_forest, isolation_forest):
    print("\n[6/8] Calculating risk scores and creating AML alerts...")

    total_predictions = 0
    total_alerts = 0

    rf_classifier = random_forest.named_steps["classifier"]
    positive_index = list(rf_classifier.classes_).index(1)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        for chunk_number, chunk in enumerate(
            pd.read_csv(
                feature_path,
                header=None,
                dtype={0: str},
                chunksize=CHUNK_SIZE,
                low_memory=False
            ),
            start=1
        ):
            ids = chunk.iloc[:, 0].astype(str).str.strip().to_numpy()

            numeric = chunk.iloc[:, 2:].apply(
                pd.to_numeric, errors="coerce"
            )

            X = numeric.replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0).to_numpy(dtype=np.float32)

            probabilities = random_forest.predict_proba(X)[:, positive_index]
            predictions = (probabilities >= 0.5).astype(int)
            anomalies = isolation_forest.predict(X)

            prediction_rows = []
            risk_rows = []
            alert_rows = []

            for tx, probability, predicted, anomaly in zip(
                ids, probabilities, predictions, anomalies
            ):
                score, level = calculate_risk(probability)

                prediction_rows.append((
                    tx,
                    "Random Forest",
                    "v1",
                    "illicit" if predicted == 1 else "licit",
                    float(probability)
                ))

                prediction_rows.append((
                    tx,
                    "Isolation Forest",
                    "v1",
                    "anomaly" if anomaly == -1 else "normal",
                    None
                ))

                risk_rows.append((tx, score, level))

                if level == "high":
                    alert_rows.append((
                        tx,
                        "ML_HIGH_RISK",
                        (
                            "Random Forest estimated illicit probability "
                            f"of {score:.2f}%. Human review required."
                        )
                    ))

            conn.executemany(
                """
                INSERT INTO model_predictions
                (transaction_id, model_name, model_version,
                 predicted_class, illicit_probability)
                VALUES (?, ?, ?, ?, ?)
                """,
                prediction_rows
            )

            conn.executemany(
                """
                INSERT INTO risk_scores
                (transaction_id, risk_score, risk_level)
                VALUES (?, ?, ?)
                """,
                risk_rows
            )

            if alert_rows:
                conn.executemany(
                    """
                    INSERT INTO aml_alerts
                    (transaction_id, alert_type, description)
                    VALUES (?, ?, ?)
                    """,
                    alert_rows
                )

            conn.commit()

            total_predictions += len(ids)
            total_alerts += len(alert_rows)

            print(
                f"  Chunk {chunk_number}: "
                f"{len(ids):,} scored; "
                f"{len(alert_rows):,} high-risk alerts"
            )

    print("Transactions scored:", f"{total_predictions:,}")
    print("High-risk alerts:", f"{total_alerts:,}")


def export_results():
    print("\n[7/8] Exporting results...")

    with sqlite3.connect(DB_PATH) as conn:
        class_summary = pd.read_sql_query(
            """
            SELECT class_label, COUNT(*) AS total_transactions
            FROM transactions
            GROUP BY class_label
            ORDER BY total_transactions DESC
            """,
            conn
        )

        time_summary = pd.read_sql_query(
            "SELECT * FROM v_time_step_summary ORDER BY time_step",
            conn
        )

        risk_summary = pd.read_sql_query(
            """
            SELECT risk_level, COUNT(*) AS total_transactions
            FROM risk_scores
            GROUP BY risk_level
            ORDER BY total_transactions DESC
            """,
            conn
        )

        top_risks = pd.read_sql_query(
            """
            SELECT transaction_id, risk_score, risk_level
            FROM risk_scores
            ORDER BY risk_score DESC
            LIMIT 100
            """,
            conn
        )

        alerts = pd.read_sql_query(
            """
            SELECT alert_id, transaction_id, alert_type,
                   description, status, created_at
            FROM aml_alerts
            ORDER BY alert_id DESC
            LIMIT 1000
            """,
            conn
        )

        integrity = conn.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        foreign_key_errors = conn.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

    class_summary.to_csv(
        OUTPUT_DIR / "class_distribution.csv", index=False
    )
    time_summary.to_csv(
        OUTPUT_DIR / "time_step_summary.csv", index=False
    )
    risk_summary.to_csv(
        OUTPUT_DIR / "risk_distribution.csv", index=False
    )
    top_risks.to_csv(
        OUTPUT_DIR / "top_risk_transactions.csv", index=False
    )
    alerts.to_csv(
        OUTPUT_DIR / "aml_alerts.csv", index=False
    )

    print("\nClass distribution:")
    print(class_summary.to_string(index=False))

    print("\nRisk distribution:")
    print(risk_summary.to_string(index=False))

    print("\nDatabase integrity:", integrity)
    print("Foreign key errors:", len(foreign_key_errors))


def main():
    print("=" * 60)
    print("ANTI-MONEY LAUNDERING TRANSACTION ANALYZER")
    print("=" * 60)

    feature_path, classes_path, edges_path = find_dataset_files()

    setup_database()

    valid_ids, X_train, y_train, X_test, y_test = import_transactions(
        feature_path, classes_path
    )

    import_edges(edges_path, valid_ids)

    random_forest, isolation_forest = train_models(
        X_train, y_train, X_test, y_test
    )

    generate_predictions(
        feature_path, random_forest, isolation_forest
    )

    export_results()

    print("\n[8/8] PROJECT PIPELINE COMPLETED")
    print("=" * 60)
    print("Database:", DB_PATH)
    print("Models:  ", MODEL_DIR)
    print("Results: ", OUTPUT_DIR)
    print("\nNext: run the Streamlit dashboard.")


if __name__ == "__main__":
    main()
