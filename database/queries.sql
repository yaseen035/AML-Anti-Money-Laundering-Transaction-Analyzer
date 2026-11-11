
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
