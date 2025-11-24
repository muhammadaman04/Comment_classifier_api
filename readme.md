# 🚀 Toxic Comment Classification — FastAPI + Streamlit

A complete end-to-end system for multi-label toxic comment detection powered by a **TorchScript GRU model**, **Byte-Pair Encoding tokenizer**, **FastAPI backend**, and a **Streamlit UI**.

---

## 📁 Project Structure

```
project/
│
├── app/
│   ├── main.py                      # FastAPI backend
│   ├── schemas/
│   │     └── api_models.py
│   ├── utils/
│   │     └── text_cleaning.py
│   ├── monitoring/
│   │     └── monitoring.py
│
├── models/
│   ├── gru_model.pt                 # TorchScript model
│   ├── bpe_tokenizer-vocab.json     # Tokenizer vocab
│   └── bpe_tokenizer-merges.txt     # Tokenizer merges
│
├── pages/
│   ├── 1_Toxicity_Analysis.py       # Classifier page
│   ├── 2_Monitoring_Dashboard.py    # Dashboard page
│   └── 3_Admin.py                    # Admin panel
│
├── streamlit_app.py                 # Feed page (default)
├── config.py                        # ⚙️ CENTRALIZED IP CONFIG
├── data_storage.py                  # Shared data storage
├── requirements.txt
└── README.md
```

---

## ⚙️ 0. Configure IP Address (IMPORTANT!)

**Before running anything, set your local network IP address:**

1. Find your IP address:
   - **Windows:** `ipconfig | findstr IPv4`
   - **Mac/Linux:** `ifconfig` or `ip addr`

2. Open `config.py` and update the IP:
   ```python
   LOCAL_IP = "10.108.72.196"  # ← Change this to your IP
   ```

3. **That's it!** All pages will automatically use this IP.

---

## 📦 1. Install Dependencies

### 🔹 Step 1 — Create Virtual Environment

```bash
python -m venv venv
```

### 🔹 Step 2 — Activate Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Mac / Linux:**

```bash
source venv/bin/activate
```

### 🔹 Step 3 — Install Required Packages

```bash
pip install -r requirements.txt
```

---

## ⚙️ 2. Run the FastAPI Backend

### 🔹 Step 1 — Make sure model + tokenizer files exist

Inside the **models/** folder:

* `gru_model.pt`
* `bpe_tokenizer-vocab.json`
* `bpe_tokenizer-merges.txt`

### 🔹 Step 2 — Start FastAPI server (Network Access)

From the root folder:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Note:** `--host 0.0.0.0` allows access from other devices on your network.

### 🔹 Step 3 — Verify API is running

**On your PC:**
- Health check: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

**On your phone/other devices:**
- Replace `127.0.0.1` with your IP from `config.py`
- Example: `http://10.108.72.196:8000/`

---

## 🖥️ 3. Run the Streamlit Frontend

### 🔹 Step 1 — Open a new terminal

Make sure your virtual environment is activated again.

### 🔹 Step 2 — Run Streamlit (Network Access)

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

**Note:** `--server.address 0.0.0.0` allows access from other devices.

### 🔹 Step 3 — Access the app

**On your PC:**
- Local: `http://localhost:8501`

**On your phone/other devices:**
- Replace `localhost` with your IP from `config.py`
- Example: `http://10.108.72.196:8501`

---

## 📱 4. Access from Mobile Devices

Once both servers are running with network access:

1. Make sure your phone is on the **same Wi-Fi network**
2. Open browser on phone
3. Go to: `http://YOUR_IP:8501` (use IP from `config.py`)
4. The Feed page will load automatically!

**All pages available:**
- Feed (default)
- Toxicity Analysis
- Monitoring Dashboard
- Admin Panel

---

## 🔗 5. How Configuration Works

All API URLs are centralized in `config.py`:

```python
LOCAL_IP = "10.108.72.196"  # ← Change only this!

API_BASE = f"http://{LOCAL_IP}:8000"
PREDICT_URL = f"{API_BASE}/predict"
MONITORING_URL = f"{API_BASE}/monitoring"
```

**When your IP changes:**
1. Find new IP: `ipconfig | findstr IPv4` (Windows)
2. Update `config.py` with new IP
3. Restart both servers

---

## 🧪 6. Testing the System

### ✔️ Test API alone (with cURL)

**On your PC:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"comment\": \"You are stupid\"}"
```

**Using your IP (from config.py):**
```bash
curl -X POST "http://10.108.72.196:8000/predict" -H "Content-Type: application/json" -d "{\"comment\": \"You are stupid\"}"
```

### ✔️ Test through Streamlit

Simply type any comment into the text box and click **Analyze**.

---

## 🛑 Troubleshooting

### ❌ **Model file not found**

Ensure your folder structure is exactly:

```
models/gru_model.pt
models/bpe_tokenizer-vocab.json
models/bpe_tokenizer-merges.txt
```

### ❌ CORS Error

Already solved in `main.py`:

```python
allow_origins=["*"]
```

### ❌ Streamlit cannot reach FastAPI

1. Check if FastAPI is running: `http://127.0.0.1:8000`
2. Verify IP in `config.py` matches your actual IP
3. Make sure both servers use `0.0.0.0` for network access

### ❌ Phone can't connect

1. Make sure phone is on **same Wi-Fi network**
2. Check Windows Firewall allows ports 8000 and 8501
3. Verify IP in `config.py` is correct
4. Try accessing API directly: `http://YOUR_IP:8000/`

---

## 📝 Quick Reference Commands

**Find your IP:**
```bash
# Windows
ipconfig | findstr IPv4

# Mac/Linux
ifconfig | grep "inet "
```

**Run FastAPI (Network Access):**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Run Streamlit (Network Access):**
```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

---
