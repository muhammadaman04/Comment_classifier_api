Below is a clean, professional, **step-by-step README.md** that explains exactly how to run both the **FastAPI backend** and the **Streamlit frontend**, including folder structure, installation, commands, and testing instructions.

---

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
│   ├── core/
│         └── (optional config files)
│
├── models/
│   ├── gru_model.pt                 # TorchScript model
│   ├── bpe_tokenizer-vocab.json     # Tokenizer vocab
│   └── bpe_tokenizer-merges.txt     # Tokenizer merges
│
├── streamlit_app.py                 # Streamlit frontend
├── requirements.txt
└── README.md
```

---

# 📦 1. Install Dependencies

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


# ⚙️ 2. Run the FastAPI Backend

### 🔹 Step 1 — Make sure model + tokenizer files exist

Inside the **models/** folder:

* `gru_model.pt`
* `bpe_tokenizer-vocab.json`
* `bpe_tokenizer-merges.txt`

### 🔹 Step 2 — Start FastAPI server

From the root folder:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 🔹 Step 3 — Verify API is running

Open in browser:

```
http://127.0.0.1:8000/
```

API docs (Swagger UI):

```
http://127.0.0.1:8000/docs
```

---

# 🖥️ 3. Run the Streamlit Frontend

### 🔹 Step 1 — Open a new terminal

Make sure your virtual environment is activated again.

### 🔹 Step 2 — Run Streamlit

```bash
streamlit run streamlit_app.py
```

### 🔹 Step 3 — The app will open automatically in browser

Example URL:

```
http://localhost:8501
```

---

# 🔗 4. Connecting Streamlit with FastAPI

In `streamlit_app.py`, make sure the API URL matches:

```python
API_URL = "http://127.0.0.1:8000/predict"
```

If FastAPI is deployed online, replace with the hosted API URL.

---

# 🧪 5. Testing the System

### ✔️ Test API alone (with cURL)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d "{\"comment\": \"You are stupid\"}"
```

### ✔️ Test through Streamlit

Simply type any comment into the text box and click **Analyze**.

---

# 🛑 Troubleshooting

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

Check if the backend is running:

```
http://127.0.0.1:8000
```

---



 