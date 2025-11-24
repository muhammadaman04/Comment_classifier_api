import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
API_BASE = "http://127.0.0.1:8000"
PREDICT_URL = f"{API_BASE}/predict"
MONITORING_URL = f"{API_BASE}/monitoring"

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

st.set_page_config(page_title="Toxic Comment Classifier", page_icon="🤖", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Toxicity Analysis", "Monitoring Dashboard"])

@st.cache_data(ttl=30)
def get_monitoring_data():
    """Efficiently fetch monitoring data"""
    try:
        stats = requests.get(f"{MONITORING_URL}/stats", timeout=5).json()
        top_words = requests.get(f"{MONITORING_URL}/top-words?limit=20", timeout=5).json()
        alerts = requests.get(f"{MONITORING_URL}/alerts", timeout=5).json()
        trend = requests.get(f"{MONITORING_URL}/vocabulary-trend?days=7", timeout=5).json()
        return stats, top_words, alerts, trend
    except Exception as e:
        st.error(f"Error fetching monitoring data: {e}")
        return None, None, None, None

def create_trend_chart(trend_data):
    """Create trend chart using matplotlib"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    dates = trend_data['dates']
    ratios = [r * 100 for r in trend_data['new_word_ratios']]
    predictions = trend_data['prediction_counts']
    
    # Plot new word ratio
    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('New Word Ratio (%)', color=color)
    line1 = ax1.plot(dates, ratios, marker='o', linewidth=2, markersize=6, color=color, label='New Word Ratio')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Alert Threshold')
    
    # Create second y-axis for predictions
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Predictions', color=color)
    bars = ax2.bar(range(len(dates)), predictions, alpha=0.3, color=color, label='Predictions')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Formatting
    ax1.set_title('Vocabulary Drift Trend (7 Days)', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    
    return fig

def create_word_frequency_chart(top_words_data, top_n=10):
    """Create horizontal bar chart for top words"""
    if not top_words_data:
        return None
        
    df_words = pd.DataFrame(top_words_data)
    df_words = df_words.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(df_words))
    bars = ax.barh(y_pos, df_words['frequency'], color=sns.color_palette("viridis", len(df_words)))
    
    # Customize the chart
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_words['word'])
    ax.set_xlabel('Frequency')
    ax.set_title(f'Top {top_n} New Words by Frequency', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_metrics_gauge(current_ratio, threshold=10):
    """Create a simple gauge chart for new word ratio"""
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # Create gauge background
    ax.barh(0, 100, color='lightgray', alpha=0.3)
    
    # Fill based on current ratio
    color = 'green' if current_ratio <= threshold else 'red'
    ax.barh(0, current_ratio, color=color, alpha=0.7)
    
    # Add threshold line
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Alert Threshold ({threshold}%)')
    
    # Customize
    ax.set_xlim(0, max(20, current_ratio + 5))
    ax.set_yticks([])
    ax.set_xlabel('New Word Ratio (%)')
    ax.set_title('Data Drift Gauge', fontweight='bold')
    ax.legend()
    
    # Add value text
    ax.text(current_ratio, 0, f'{current_ratio:.1f}%', 
            ha='center', va='center', fontweight='bold', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

if page == "Toxicity Analysis":
    # -----------------------------
    # TOXICITY ANALYSIS PAGE
    # -----------------------------
    st.title("🧪 Toxic Comment Classification")
    st.write("Enter a comment below to test your FastAPI model.")

    # ---------------------------------------------
    # USER INPUT
    # ---------------------------------------------
    user_input = st.text_area("✏️ Enter your comment:", height=150)

    if st.button("Analyze"):
        if not user_input.strip():
            st.warning("⚠️ Please enter a comment first.")
        else:
            with st.spinner("Contacting API..."):
                try:
                    payload = {"comment": user_input}
                    response = requests.post(PREDICT_URL, json=payload)

                    if response.status_code != 200:
                        st.error(f"API Error: {response.text}")
                    else:
                        result = response.json()

                        # -----------------------------
                        # DISPLAY RESULTS
                        # -----------------------------
                        st.subheader("🟦 Overall Prediction")
                        if result["is_toxic"]:
                            st.error("⚠️ This comment is toxic")
                        else:
                            st.success("✅ This comment is not toxic")

                        st.write("---")
                        st.subheader("📌 Explanation")
                        st.write(result["explanation"])

                        st.write("---")
                        st.subheader("📊 Label-wise Predictions")

                        st.write("**Detailed Breakdown:**")
                        for pred in result["predictions"]:
                            label = pred["label"].replace("_", " ").title()
                            prob = pred["probability"]
                            is_pos = pred["is_positive"]

                            if is_pos:
                                st.markdown(f"🔴 **{label}** — {prob:.4f}")
                            else:
                                st.markdown(f"🟢 {label} — {prob:.4f}")

                except Exception as e:
                    st.error(f"Request failed: {e}")

    st.write("---")
    st.caption("Built with Streamlit + FastAPI 🚀")

else:
    # -----------------------------
    # MONITORING DASHBOARD PAGE
    # -----------------------------
    st.title("🚨 Data Drift Monitoring Dashboard")
    st.markdown("### Real-time Vocabulary Monitoring for Toxic Comment Classification")
    
    # Refresh button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Get monitoring data
    stats, top_words, alerts, trend = get_monitoring_data()
    
    if not stats:
        st.error("Unable to fetch monitoring data. Make sure the API server is running.")
        st.stop()
    
    # Key Metrics
    st.subheader("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Predictions", 
            stats["daily_stats"]["total_predictions"]
        )
    
    with col2:
        new_word_ratio = stats["daily_stats"]["new_word_ratio"] * 100
        st.metric(
            "New Word Ratio", 
            f"{new_word_ratio:.2f}%",
            delta=None,
            delta_color="inverse" if new_word_ratio > 10 else "normal"
        )
    
    with col3:
        st.metric(
            "Unknown Words", 
            f"{stats['daily_stats']['unknown_vocabulary_size']:,}"
        )
    
    with col4:
        alert_status = stats["alert_status"]
        st.metric(
            "Status", 
            alert_status,
            delta="⚠️ Alert" if alert_status == "ALERT" else "✅ Normal"
        )
    
    # Visualizations Section
    st.subheader("📊 Visualizations")
    
    # Gauge Chart
    st.markdown("**Data Drift Gauge**")
    gauge_fig = create_metrics_gauge(new_word_ratio)
    st.pyplot(gauge_fig)
    
    # Charts in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Vocabulary Drift Trend**")
        if trend and trend['dates']:
            trend_fig = create_trend_chart(trend)
            st.pyplot(trend_fig)
        else:
            st.info("No trend data available yet. Make some predictions first!")
    
    with col2:
        st.markdown("**Top New Words**")
        if top_words and top_words["top_new_words"]:
            words_fig = create_word_frequency_chart(top_words["top_new_words"], top_n=10)
            st.pyplot(words_fig)
        else:
            st.info("No new words detected yet. Make some predictions first!")
    
    # Alerts and Data Section
    st.subheader("🚨 Alerts & Detailed Data")
    data_col1, data_col2 = st.columns(2)
    
    with data_col1:
        st.markdown("**Active Alerts**")
        if alerts and alerts["alerts"]:
            for i, alert in enumerate(alerts["alerts"][:8]):
                timestamp = datetime.fromisoformat(alert['timestamp']).strftime("%m/%d %H:%M")
                severity_color = "red" if alert['severity'] == "HIGH" else "orange"
                
                st.markdown(f"""
                <div style="border-left: 4px solid {severity_color}; padding-left: 10px; margin: 5px 0;">
                    <strong>{alert['word']}</strong><br>
                    Frequency: {alert['frequency']} | Severity: <span style="color: {severity_color}">{alert['severity']}</span><br>
                    <small>Detected: {timestamp}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🎉 No active alerts!")
    
    with data_col2:
        st.markdown("**Recent New Words**")
        if top_words and top_words["top_new_words"]:
            words_df = pd.DataFrame(top_words["top_new_words"])
            display_df = words_df.head(15).copy()
            display_df['word'] = display_df['word'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "word": st.column_config.TextColumn("Word", width="medium"),
                    "frequency": st.column_config.NumberColumn("Frequency", width="small")
                }
            )
            
            # Summary stats
            total_new_words = len(words_df)
            avg_frequency = words_df['frequency'].mean() if len(words_df) > 0 else 0
            st.caption(f"Total unique new words: {total_new_words} | Avg frequency: {avg_frequency:.1f}")
        else:
            st.info("No new words to display")
    
    # Data Management in Sidebar
    st.sidebar.subheader("Data Management")
    if st.sidebar.button("🔄 Force Save Data"):
        try:
            response = requests.post(f"{API_BASE}/monitoring/cleanup?days_to_keep=30")
            if response.status_code == 200:
                st.sidebar.success("Data saved successfully!")
            else:
                st.sidebar.error("Save failed")
        except Exception as e:
            st.sidebar.error(f"Save error: {e}")
    
    if st.sidebar.button("🧹 Cleanup Old Data"):
        try:
            response = requests.post(f"{API_BASE}/monitoring/cleanup?days_to_keep=30")
            if response.status_code == 200:
                st.sidebar.success("Old data cleaned up!")
            else:
                st.sidebar.error("Cleanup failed")
        except Exception as e:
            st.sidebar.error(f"Cleanup error: {e}")
    
    if st.sidebar.button("📤 Export Data"):
        try:
            response = requests.get(f"{API_BASE}/monitoring/export")
            if response.status_code == 200:
                st.sidebar.success("Data exported!")
            else:
                st.sidebar.error("Export failed")
        except Exception as e:
            st.sidebar.error(f"Export error: {e}")
    
    # Raw Data Section
    with st.expander("📋 Raw Monitoring Data"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Monitoring Statistics**")
            st.json(stats)
        
        with col2:
            st.write("**System Information**")
            try:
                model_info = requests.get(f"{API_BASE}/model/info", timeout=5).json()
                st.json(model_info)
            except:
                st.error("Could not fetch model info")

# Footer
st.sidebar.write("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard monitors data drift by tracking new words "
    "that are not in the model's vocabulary and generates alerts "
    "for frequent unknown words."
)