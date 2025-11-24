import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

from config import API_BASE, MONITORING_URL

# -----------------------------
# CONFIG
# -----------------------------
DASHBOARD_PASSWORD = "admin123"  # Hardcoded password

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

st.set_page_config(page_title="Monitoring Dashboard", page_icon="📊", layout="wide")

# -----------------------------
# PASSWORD PROTECTION
# -----------------------------
if "dashboard_authenticated" not in st.session_state:
    st.session_state.dashboard_authenticated = False

if not st.session_state.dashboard_authenticated:
    st.title("🔐 Dashboard Login")
    password_input = st.text_input("Enter Dashboard Password:", type="password")
    
    if st.button("Login"):
        if password_input == DASHBOARD_PASSWORD:
            st.session_state.dashboard_authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password!")
    
    st.stop()


@st.cache_data(ttl=30)
def get_monitoring_data():
    """Fetch monitoring payloads from the FastAPI server."""
    try:
        stats = requests.get(f"{MONITORING_URL}/stats", timeout=5).json()
        top_words = requests.get(f"{MONITORING_URL}/top-words?limit=20", timeout=5).json()
        alerts = requests.get(f"{MONITORING_URL}/alerts", timeout=5).json()
        trend = requests.get(f"{MONITORING_URL}/vocabulary-trend?days=7", timeout=5).json()
        return stats, top_words, alerts, trend
    except Exception as exc:
        st.error(f"Error fetching monitoring data: {exc}")
        return None, None, None, None


def create_trend_chart(trend_data):
    """Render vocabulary drift & prediction trend chart."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    dates = trend_data["dates"]
    ratios = [r * 100 for r in trend_data["new_word_ratios"]]
    predictions = trend_data["prediction_counts"]

    color = "tab:red"
    ax1.set_xlabel("Date")
    ax1.set_ylabel("New Word Ratio (%)", color=color)
    ax1.plot(dates, ratios, marker="o", linewidth=2, markersize=6, color=color, label="New Word Ratio")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.axhline(y=10, color="red", linestyle="--", alpha=0.7, label="Alert Threshold")

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Predictions", color=color)
    ax2.bar(range(len(dates)), predictions, alpha=0.3, color=color, label="Predictions")
    ax2.tick_params(axis="y", labelcolor=color)

    ax1.set_title("Vocabulary Drift Trend (7 Days)", fontsize=14, fontweight="bold")
    ax1.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


def create_word_frequency_chart(top_words_data, top_n=10):
    """Render frequency chart for new words."""
    if not top_words_data:
        return None

    df_words = pd.DataFrame(top_words_data).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(df_words))
    bars = ax.barh(y_pos, df_words["frequency"], color=sns.color_palette("viridis", len(df_words)))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_words["word"])
    ax.set_xlabel("Frequency")
    ax.set_title(f"Top {top_n} New Words by Frequency", fontsize=14, fontweight="bold")

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height() / 2, f"{int(width)}", ha="left", va="center", fontweight="bold")

    plt.tight_layout()
    return fig


def create_metrics_gauge(current_ratio, threshold=10):
    """Render a simple horizontal gauge for data drift."""
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh(0, 100, color="lightgray", alpha=0.3)

    color = "green" if current_ratio <= threshold else "red"
    ax.barh(0, current_ratio, color=color, alpha=0.7)
    ax.axvline(x=threshold, color="red", linestyle="--", linewidth=2, label=f"Alert Threshold ({threshold}%)")

    ax.set_xlim(0, max(20, current_ratio + 5))
    ax.set_yticks([])
    ax.set_xlabel("New Word Ratio (%)")
    ax.set_title("Data Drift Gauge", fontweight="bold")
    ax.legend()
    ax.text(
        current_ratio,
        0,
        f"{current_ratio:.1f}%",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    plt.tight_layout()
    return fig


def render_dashboard():
    st.title("🚨 Data Drift Monitoring Dashboard")
    st.markdown("### Real-time monitoring for toxic comment classification")
    
    # Logout button
    col_refresh, col_logout = st.columns([1, 1])
    with col_refresh:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    with col_logout:
        if st.button("🚪 Logout"):
            st.session_state.dashboard_authenticated = False
            st.rerun()

    stats, top_words, alerts, trend = get_monitoring_data()
    if not stats:
        st.error("Unable to fetch monitoring data. Make sure the API server is running.")
        return

    st.subheader("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", stats["daily_stats"]["total_predictions"])
    with col2:
        new_word_ratio = stats["daily_stats"]["new_word_ratio"] * 100
        st.metric("New Word Ratio", f"{new_word_ratio:.2f}%", delta_color="inverse" if new_word_ratio > 10 else "normal")
    with col3:
        st.metric("Unknown Words", f"{stats['daily_stats']['unknown_vocabulary_size']:,}")
    with col4:
        alert_status = stats["alert_status"]
        st.metric("Status", alert_status, delta="⚠️ Alert" if alert_status == "ALERT" else "✅ Normal")

    st.subheader("📊 Visualizations")
    st.markdown("**Data Drift Gauge**")
    st.pyplot(create_metrics_gauge(new_word_ratio))

    vis_col1, vis_col2 = st.columns(2)
    with vis_col1:
        st.markdown("**Vocabulary Drift Trend**")
        if trend and trend["dates"]:
            st.pyplot(create_trend_chart(trend))
        else:
            st.info("No trend data available yet. Make some predictions first!")
    with vis_col2:
        st.markdown("**Top New Words**")
        if top_words and top_words["top_new_words"]:
            st.pyplot(create_word_frequency_chart(top_words["top_new_words"], top_n=10))
        else:
            st.info("No new words detected yet.")

    st.subheader("🚨 Alerts & Detailed Data")
    data_col1, data_col2 = st.columns(2)
    with data_col1:
        st.markdown("**Active Alerts**")
        if alerts and alerts["alerts"]:
            for alert in alerts["alerts"][:8]:
                timestamp = datetime.fromisoformat(alert["timestamp"]).strftime("%m/%d %H:%M")
                severity_color = "red" if alert["severity"] == "HIGH" else "orange"
                st.markdown(
                    f"""
                    <div style="border-left: 4px solid {severity_color}; padding-left: 10px; margin: 5px 0;">
                        <strong>{alert['word']}</strong><br>
                        Frequency: {alert['frequency']} | Severity: <span style="color: {severity_color}">{alert['severity']}</span><br>
                        <small>Detected: {timestamp}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("🎉 No active alerts!")

    with data_col2:
        st.markdown("**Recent New Words**")
        if top_words and top_words["top_new_words"]:
            words_df = pd.DataFrame(top_words["top_new_words"])
            display_df = words_df.head(15).copy()
            display_df["word"] = display_df["word"].apply(lambda x: x[:30] + "..." if len(x) > 30 else x)
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "word": st.column_config.TextColumn("Word", width="medium"),
                    "frequency": st.column_config.NumberColumn("Frequency", width="small"),
                },
            )
            total_new_words = len(words_df)
            avg_frequency = words_df["frequency"].mean() if len(words_df) > 0 else 0
            st.caption(f"Total unique new words: {total_new_words} | Avg frequency: {avg_frequency:.1f}")
        else:
            st.info("No new words to display")

    st.sidebar.write("---")
    st.sidebar.subheader("Admin Actions")
    if st.sidebar.button("🧹 Cleanup Old Data"):
        try:
            response = requests.post(f"{MONITORING_URL}/cleanup?days_to_keep=30")
            if response.status_code == 200:
                st.sidebar.success("Cleanup completed")
            else:
                st.sidebar.error("Cleanup failed")
        except Exception as exc:
            st.sidebar.error(f"Cleanup error: {exc}")

    if st.sidebar.button("📤 Export Monitoring Data"):
        try:
            response = requests.get(f"{MONITORING_URL}/export")
            if response.status_code == 200:
                st.sidebar.success("Export triggered")
            else:
                st.sidebar.error("Export failed")
        except Exception as exc:
            st.sidebar.error(f"Export error: {exc}")

    with st.expander("📋 Raw Monitoring Data"):
        col_raw1, col_raw2 = st.columns(2)
        with col_raw1:
            st.write("**Monitoring Statistics**")
            st.json(stats)
        with col_raw2:
            st.write("**System Information**")
            try:
                model_info = requests.get(f"{API_BASE}/model/info", timeout=5).json()
                st.json(model_info)
            except Exception:
                st.error("Could not fetch model info")


render_dashboard()

