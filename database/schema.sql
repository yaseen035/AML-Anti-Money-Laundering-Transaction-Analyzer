
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
