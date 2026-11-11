
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
