from pathlib import Path
import sqlite3
import json

import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "database" / "aml_analyzer.db"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"

st.set_page_config(
    page_title="AML Transaction Analyzer",
    page_icon="🔍",
    layout="wide"
)

st.title("Anti-Money Laundering Transaction Analyzer")
st.caption("Blockchain transaction monitoring and machine learning analysis")

if not DB_PATH.exists():
    st.error("Database not found. Run src/run_pipeline.py first.")
    st.stop()


def query_db(sql, params=()):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn, params=params)


@st.cache_data(ttl=30)
def get_summary():
    return query_db("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN class_label = 'licit' THEN 1 ELSE 0 END) AS licit,
            SUM(CASE WHEN class_label = 'illicit' THEN 1 ELSE 0 END) AS illicit,
            SUM(CASE WHEN class_label = 'unknown' THEN 1 ELSE 0 END) AS unknown
        FROM transactions
    """)


summary = get_summary()

total = int(summary.iloc[0]["total"] or 0)
licit = int(summary.iloc[0]["licit"] or 0)
illicit = int(summary.iloc[0]["illicit"] or 0)
unknown = int(summary.iloc[0]["unknown"] or 0)

risk_summary = query_db("""
    SELECT risk_level, COUNT(*) AS total
    FROM risk_scores
    GROUP BY risk_level
""")

alert_count = query_db("""
    SELECT COUNT(*) AS total
    FROM aml_alerts
    WHERE status = 'open'
""").iloc[0]["total"]

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select page",
    [
        "Overview",
        "Transaction Explorer",
        "Risk Monitoring",
        "AML Alerts",
        "Statistical Analysis",
        "Model Evaluation"
    ]
)

if page == "Overview":
    st.header("Project Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Licit Transactions", f"{licit:,}")
    c3.metric("Illicit Transactions", f"{illicit:,}")
    c4.metric("Unknown Transactions", f"{unknown:,}")

    c5, c6 = st.columns(2)
    c5.metric("Open AML Alerts", f"{int(alert_count):,}")
    c6.metric("Transactions Scored", f"{total:,}")

    st.subheader("Transaction Classification")

    class_data = pd.DataFrame({
        "Classification": ["Licit", "Illicit", "Unknown"],
        "Transactions": [licit, illicit, unknown]
    })

    fig = px.bar(
        class_data,
        x="Classification",
        y="Transactions",
        color="Classification",
        title="Transaction Class Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Distribution")

    if not risk_summary.empty:
        fig = px.pie(
            risk_summary,
            names="risk_level",
            values="total",
            title="Predicted Transaction Risk Levels"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highest Risk Transactions")

    top_risks = query_db("""
        SELECT transaction_id, risk_score, risk_level
        FROM risk_scores
        ORDER BY risk_score DESC
        LIMIT 15
    """)

    st.dataframe(top_risks, use_container_width=True)

elif page == "Transaction Explorer":
    st.header("Transaction Explorer")

    search = st.text_input("Search transaction ID")
    label = st.selectbox(
        "Classification",
        ["All", "licit", "illicit", "unknown"]
    )

    sql = """
        SELECT transaction_id, time_step, class_label
        FROM transactions
        WHERE transaction_id LIKE ?
    """
    params = [f"%{search}%"]

    if label != "All":
        sql += " AND class_label = ?"
        params.append(label)

    sql += " ORDER BY time_step DESC LIMIT 200"

    results = query_db(sql, tuple(params))

    st.write(f"Records displayed: {len(results)}")
    st.dataframe(results, use_container_width=True)

elif page == "Risk Monitoring":
    st.header("Risk Monitoring")

    risk_filter = st.selectbox(
        "Select risk level",
        ["All", "high", "medium", "low"]
    )

    sql = """
        SELECT transaction_id, risk_score, risk_level, calculated_at
        FROM risk_scores
    """

    if risk_filter != "All":
        sql += " WHERE risk_level = ?"
        results = query_db(sql + " ORDER BY risk_score DESC LIMIT 500",
                           (risk_filter,))
    else:
        results = query_db(sql + " ORDER BY risk_score DESC LIMIT 500")

    st.dataframe(results, use_container_width=True)

    st.download_button(
        "Export displayed risk records",
        results.to_csv(index=False),
        file_name="risk_records.csv",
        mime="text/csv"
    )

elif page == "AML Alerts":
    st.header("AML Alert Management")

    alerts = query_db("""
        SELECT
            alert_id,
            transaction_id,
            alert_type,
            description,
            status,
            created_at
        FROM aml_alerts
        ORDER BY alert_id DESC
        LIMIT 1000
    """)

    st.write("Displayed alerts:", len(alerts))
    st.dataframe(alerts, use_container_width=True)

    st.download_button(
        "Export AML alerts",
        alerts.to_csv(index=False),
        file_name="aml_alerts.csv",
        mime="text/csv"
    )

elif page == "Statistical Analysis":
    st.header("Statistical Analysis")

    time_data = query_db("""
        SELECT time_step, total_transactions,
               illicit_transactions, licit_transactions
        FROM v_time_step_summary
        ORDER BY time_step
    """)

    st.subheader("Transactions by Time Step")

    fig = px.line(
        time_data,
        x="time_step",
        y=["licit_transactions", "illicit_transactions"],
        markers=True,
        title="Licit and Illicit Transactions Across Time"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Descriptive Statistics of Risk Scores")

    risk_stats = query_db("""
        SELECT
            COUNT(*) AS total,
            AVG(risk_score) AS mean_risk,
            MIN(risk_score) AS minimum_risk,
            MAX(risk_score) AS maximum_risk
        FROM risk_scores
    """)

    st.dataframe(risk_stats, use_container_width=True)
    st.dataframe(time_data, use_container_width=True)

elif page == "Model Evaluation":
    st.header("Machine Learning Evaluation")

    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics = json.load(f)

        st.subheader("Model Comparison")

        comparison_rows = []

        for model_name in [
            "logistic_regression",
            "random_forest",
            "isolation_forest"
        ]:
            model_result = metrics.get(model_name, {})
            if model_result:
                comparison_rows.append({
                    "Model": model_name.replace("_", " ").title(),
                    "Accuracy": model_result.get("accuracy"),
                    "Precision": model_result.get("precision_illicit"),
                    "Recall": model_result.get("recall_illicit"),
                    "F1 Score": model_result.get("f1_illicit")
                })

        st.dataframe(
            pd.DataFrame(comparison_rows),
            use_container_width=True
        )

        st.subheader("Random Forest Performance")

        results = metrics["random_forest"]

        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{results['accuracy']:.3f}")
        b.metric("Precision", f"{results['precision_illicit']:.3f}")
        c.metric("Recall", f"{results['recall_illicit']:.3f}")
        d.metric("F1 Score", f"{results['f1_illicit']:.3f}")

        if "roc_auc" in results:
            st.metric("ROC-AUC", f"{results['roc_auc']:.3f}")

        if "pr_auc" in results:
            st.metric("PR-AUC", f"{results['pr_auc']:.3f}")

        st.subheader("Confusion Matrix")

        cm = pd.DataFrame(
            results["confusion_matrix"],
            index=["Actual licit", "Actual illicit"],
            columns=["Predicted licit", "Predicted illicit"]
        )

        st.dataframe(cm, use_container_width=True)

        st.subheader("Training Information")
        st.json({
            "Training rows": metrics["training_rows"],
            "Testing rows": metrics["testing_rows"],
            "Numerical features": metrics["feature_count"]
        })

    else:
        st.warning("Model evaluation file not found. Run the pipeline first.")

st.sidebar.divider()
st.sidebar.caption("Academic project | Computational Mathematics")
st.sidebar.caption("Predictions require human review.")
