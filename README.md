# AnomalyX - AI-Powered Real-Time Anomaly Detection & Intelligence

**AnomalyX** is an enterprise-grade, AI-powered real-time anomaly detection and root-cause intelligence web application. It combines scikit-learn machine learning algorithms (Isolation Forest, Local Outlier Factor, One-Class SVM, Z-Score Ensemble) with a high-performance FastAPI backend and a sleek, dark glassmorphism dashboard.

---

## 🚀 Features

- **Real-Time Telemetry Stream**: Live visual rendering of CPU, Memory, Network IO, and Latency metrics with instant anomaly detection highlights.
- **Multi-Algorithm ML Engine**: Switch between Isolation Forest, Local Outlier Factor (LOF), One-Class Support Vector Machines (SVM), and Statistical Z-Score Ensembles.
- **Root-Cause Explainability**: Evaluates normalized feature Z-scores to calculate percentage contributions of individual metrics driving each anomaly.
- **Interactive Dataset Ingestion**: Drag-and-drop custom `.CSV` or `.JSON` files for automated dataset anomaly scoring.
- **Contamination Sensitivity Tuning**: Real-time threshold slider adjusting decision boundaries dynamically.
- **Alert Stream & CSV Export**: Real-time anomaly alert log with severity filtering (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`) and CSV export capabilities.

---

## 📁 Repository Directory Structure

```
BangloreHackathon/
│
├── frontend/                     # Modern Dark Glassmorphic Dashboard
│   ├── index.html                # Semantic HTML5 Structure
│   ├── styles.css                # Glassmorphic CSS Design System & Variables
│   └── app.js                    # Dashboard Controller & Chart.js Visualizer
│
├── backend/                      # High-Performance FastAPI REST Server
│   ├── __init__.py               # Package Export
│   ├── main.py                   # FastAPI Application Entrypoint & Static Mount
│   ├── routes.py                 # REST API Endpoints (/predict, /stream, /upload, /alerts)
│   └── schemas.py                # Pydantic Request/Response Data Validation
│
├── ml_module/                    # Machine Learning Intelligence Engine
│   ├── __init__.py               # Package Export
│   ├── detector.py               # Multi-Algorithm AnomalyDetector Ensemble
│   ├── preprocessor.py           # Feature Standardisation & Median Imputation
│   ├── explainer.py              # Root-Cause Feature Attribution Engine
│   └── generator.py              # Multi-Metric Telemetry & Anomaly Stream Ingestor
│
├── Specification Documents/      # Comprehensive Technical Specs
│   ├── Frontend prompts          # Frontend Widget & UI Specs
│   ├── Frontend Master prompt   # Glassmorphic Design System Tokens & Specs
│   ├── Backend prompts           # REST API Endpoint Signatures & Payloads
│   ├── Backend Master prompt    # FastAPI Architecture & Middleware Specs
│   ├── ML module prompt          # Machine Learning Parameters & Metrics
│   ├── ML Master prompt          # Ensemble Pipeline & Preprocessor Specs
│   ├── Component Details         # Component Breakdown & Interface Contracts
│   ├── Techical Workflow        # End-to-End Sequence Diagrams & Data Lifecycle
│   └── Execution plan and checklist # Roadmap & Verification Checklist
│
├── requirements.txt              # Python Dependencies
├── .gitignore                    # Git Ignore Rules
└── README.md                     # Project Overview & Setup Guide
```

---

## 🛠️ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/AnomalyX.get
cd BangloreHackathon
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
python -m uvicorn backend.main:app --port 8000 --reload
```

Open your browser to **[http://localhost:8000/](http://localhost:8000/)** to access the dashboard.
