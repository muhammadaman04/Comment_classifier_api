# ============================================
# CENTRALIZED CONFIGURATION
# ============================================
# Change this IP address to match your local network IP
# Find your IP: Windows: ipconfig | findstr IPv4
#                Mac/Linux: ifconfig or ip addr

# Your local network IP address (change this when your IP changes)
LOCAL_IP = "192.168.0.57"

# API Configuration
API_BASE = f"http://{LOCAL_IP}:8000"
PREDICT_URL = f"{API_BASE}/predict"
MONITORING_URL = f"{API_BASE}/monitoring"

# Streamlit Configuration
STREAMLIT_PORT = 8501

