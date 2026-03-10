
# 🔵 Machine Learning–Based Employee Attrition Prediction & Risk Scoring System

> **Predict which employees are most likely to leave — before they resign.**  
> An end-to-end ML system that transforms HR analytics from reactive to proactive.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [The Problem](#-the-problem)
- [Solution Architecture](#-solution-architecture)
- [Dataset](#-dataset)
- [ML Pipeline](#-ml-pipeline)
- [Model Performance](#-model-performance)
- [Risk Scoring Framework](#-risk-scoring-framework)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Key Findings](#-key-findings)
- [Deliverables](#-deliverables)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🎯 Project Overview

This project builds a **production-ready employee attrition prediction system** for Palo Alto Networks using machine learning. It analyzes 1,470 employee records across 31 features to:

- ✅ **Predict** whether an employee will leave (binary classification)
- ✅ **Score** each employee with an attrition probability (0–100%)
- ✅ **Categorize** employees into Low / Medium / High risk tiers
- ✅ **Explain** why each employee is at risk using SHAP values
- ✅ **Visualize** everything through an interactive Streamlit dashboard

> **Internship Program:** Unified Mentor  
> **Organization:** Palo Alto Networks  
> **Domain:** Machine Learning · HR Analytics · Predictive Intelligence

---

## 🚨 The Problem

Palo Alto Networks faces three critical attrition challenges:

| Challenge | Impact |
|---|---|
| Sudden, unanticipated resignations | Operational disruption, project delays |
| Loss of high-performing employees | Revenue risk, knowledge drain |
| Reactive counter-offers too late | Low success rate, damaged trust |

**The root cause:** HR lacks a systematic, data-driven early warning system.  
**The cost:** Replacing one technology employee costs **50–200% of annual salary** (~$75,000+).

---

## 🏗️ Solution Architecture

```
Raw HR Data (1,470 employees × 31 features)
        │
        ▼
┌─────────────────────────────────────────────────┐
│              DATA PREPROCESSING                  │
│  Label Encoding → One-Hot Encoding → Scaling     │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│            FEATURE ENGINEERING                   │
│  IncomePerYear · EngagementScore · WorkloadStress│
│  PromotionDelay · RoleStagnation · LoyaltyIndex  │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│         CLASS BALANCING (SMOTE)                  │
│  84% Stayed → 16% Left  ──►  50% : 50%          │
└────────────────────┬────────────────────────────┘
                     │
              ┌──────┴──────┐
              ▼             ▼
    ┌──────────────┐  ┌─────────────────────┐
    │  Baseline     │  │   Advanced Models   │
    │  Logistic    │  │  Random Forest      │
    │  Regression  │  │  XGBoost ⭐ (Best)  │
    └──────────────┘  └─────────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │   RISK SCORING ENGINE    │
              │  Attrition Probability   │
              │  Low / Medium / High     │
              └──────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │  SHAP EXPLAINABILITY     │
              │  Global Feature Impact   │
              │  Individual Risk Reasons │
              └──────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │   STREAMLIT DASHBOARD    │
              │  5 Interactive Tabs      │
              │  Real-time What-If Tool  │
              └──────────────────────────┘
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | Palo Alto Networks HR Records |
| **Employees** | 1,470 |
| **Features** | 31 raw + 6 engineered = 37 total |
| **Target** | `Attrition` (0 = Stayed, 1 = Left) |
| **Class Distribution** | 83.9% Stayed · 16.1% Left |
| **Missing Values** | None |

### Feature Categories

```
📁 Demographics      → Age, Gender, MaritalStatus, DistanceFromHome
📁 Job Attributes    → Department, JobRole, JobLevel, OverTime, BusinessTravel
📁 Compensation      → MonthlyIncome, DailyRate, HourlyRate, StockOptionLevel
📁 Satisfaction      → JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance
📁 Career Trajectory → YearsAtCompany, YearsSinceLastPromotion, TotalWorkingYears
📁 Engineered (NEW)  → EngagementScore, WorkloadStress, IncomePerYear, LoyaltyIndex
```

---

## 🔬 ML Pipeline

### 1. Data Preprocessing
- **Label Encoding** → `OverTime`, `Gender`
- **One-Hot Encoding** → `BusinessTravel`, `Department`, `EducationField`, `JobRole`, `MaritalStatus`
- **StandardScaler** → All numerical features normalized

### 2. Feature Engineering

| Feature | Formula | Business Rationale |
|---|---|---|
| `IncomePerYear` | `MonthlyIncome / (TotalWorkingYears + 1)` | Detects underpaid employees |
| `EngagementScore` | Mean of 5 satisfaction scores | Composite disengagement signal |
| `WorkloadStress` | `OverTime=1 AND WorkLifeBalance≤2` | Burnout risk flag |
| `PromotionDelay` | `YearsAtCompany - YearsSinceLastPromotion` | Career stagnation indicator |
| `RoleStagnation` | `YearsInCurrentRole - YearsSinceLastPromotion` | Role growth gap |
| `LoyaltyIndex` | `YearsAtCompany / (NumCompaniesWorked + 1)` | Job-hopping propensity |

### 3. Class Balancing — SMOTE
```
Before SMOTE:  Stayed: 984  |  Left: 192   (84:16 ratio)
After  SMOTE:  Stayed: 984  |  Left: 984   (50:50 balanced)
```

### 4. Train / Test Split
```
Total: 1,470 employees
├── Train: 1,176  (80%) — SMOTE applied here only
└── Test:    294  (20%) — stratified, original distribution preserved
```

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.782 | 0.531 | 0.712 | 0.608 | 0.820 |
| Random Forest | 0.847 | 0.648 | 0.745 | 0.693 | 0.856 |
| **XGBoost ⭐** | **0.863** | **0.701** | **0.788** | **0.742** | **0.874** |

> **Why Recall matters most:** Missing an employee who will leave (false negative) is far more costly than flagging a stable employee for an extra check-in (false positive). XGBoost catches **78.8%** of at-risk employees.

---

## 🎯 Risk Scoring Framework

Every employee receives an **Attrition Risk Score** (0–100%) and a risk category:

```
┌─────────────────────────────────────────────────────────────┐
│                  ATTRITION RISK TIERS                       │
├──────────────┬────────────────┬────────────────────────────┤
│  🟢 LOW RISK │  🟡 MEDIUM RISK │       🔴 HIGH RISK         │
│    0 – 29%   │    30 – 59%    │          60 – 100%         │
├──────────────┼────────────────┼────────────────────────────┤
│  892 (60.7%) │  421 (28.6%)   │         157 (10.7%)        │
├──────────────┼────────────────┼────────────────────────────┤
│ Regular      │ Proactive 1:1s │ URGENT: Immediate          │
│ check-ins    │ Career review  │ retention intervention     │
└──────────────┴────────────────┴────────────────────────────┘
```

> The risk threshold is **fully configurable** via the dashboard sidebar slider.

---

## 🖥️ Streamlit Dashboard

A 5-tab interactive web application providing complete HR visibility:

| Tab | Features |
|---|---|
| **📊 Risk Overview** | Risk donut chart · Probability histogram · High-risk employee table · Attrition by overtime / travel / marital status |
| **👤 Employee Profile** | Animated risk gauge meter · Satisfaction radar chart · Full employee details · HR recommended action |
| **🏢 Department View** | Risk count by department · Avg risk score bars · Job role breakdown table |
| **🔍 Explainability** | Top-15 feature importance chart · Key insights panel · **Live What-If scenario explorer** |
| **🤖 Model Performance** | Confusion matrix · Metrics comparison · Full ML pipeline summary |

### Sidebar Controls
- 🎛️ **Department & Job Role filters**
- ⚠️ **High Risk threshold slider** (40%–80%)
- 👤 **Employee ID lookup** (0–1469)

---

## 📁 Project Structure

```
📦 employee-attrition-prediction/
│
├── 📓 Employee_Attrition_Prediction.ipynb   # Complete Colab notebook (18 cells)
├── 🖥️  app.py                                # Streamlit dashboard (5 tabs)
├── 📋 requirements.txt                       # Python dependencies
├── 📊 Palo_Alto_Networks__1_.csv            # HR dataset (1,470 employees)
├── 📄 Research_Paper_Attrition.docx         # Full academic research paper
├── 📊 Executive_Summary_Attrition.docx      # Executive summary for HR leadership
└── 📖 README.md                             # This file
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9+
- pip

### Step 1 — Clone the Repository
```bash
git clone https://github.com/your-username/employee-attrition-prediction.git
cd employee-attrition-prediction
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install streamlit xgboost shap imbalanced-learn scikit-learn pandas numpy matplotlib seaborn
```

### Step 3 — Run the Dashboard
```bash
streamlit run app.py
```

The app opens automatically at **`http://localhost:8501`**

### Step 4 — Run the Notebook (Optional)
Open `Employee_Attrition_Prediction.ipynb` in Google Colab or Jupyter:
```bash
jupyter notebook Employee_Attrition_Prediction.ipynb
```

---

## 🚀 Usage

### Running the Streamlit App
```bash
# Make sure the CSV file is in the same directory as app.py
streamlit run app.py
```

### Deploying to Streamlit Cloud
1. Push all files to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo — `requirements.txt` handles all installs automatically

### Using the Notebook in Google Colab
1. Upload `Employee_Attrition_Prediction.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Run **Cell 1** to install libraries
3. Run **Cell 2** — upload the CSV when prompted
4. Run all remaining cells top to bottom

---

## 🔍 Key Findings

### Top Attrition Drivers

| Rank | Feature | Importance | Insight |
|---|---|---|---|
| 1 | **OverTime** | 0.1842 | Overtime workers are **3× more likely** to leave |
| 2 | **MonthlyIncome** | 0.1456 | Leavers earn **42% less** than those who stay |
| 3 | **YearsAtCompany** | 0.0982 | **0–3 year** employees account for 43% of attritions |
| 4 | **EngagementScore** | 0.0874 | Disengagement composite is a top engineered signal |
| 5 | **Age** | 0.0743 | Employees aged **25–30** show highest risk |
| 6 | **JobSatisfaction** | 0.0612 | Score 1 employees show **22.8%** attrition rate |
| 7 | **StockOptionLevel** | 0.0587 | No equity = missing retention anchor |
| 8 | **DistanceFromHome** | 0.0521 | Long commutes erode work-life balance over time |

### Financial Impact Estimate

```
Current Annual Attrition Cost      →  $17.8M   (237 employees × $75K avg)
High-Risk Employee Exposure        →  $11.8M   (157 employees × $75K avg)
Potential Savings (40% retention)  →  $4.7M+   annually, conservative estimate
```

---

## 📦 Deliverables

| # | Deliverable | Description | Status |
|---|---|---|---|
| 1 | `app.py` | Streamlit interactive dashboard | ✅ Complete |
| 2 | `Employee_Attrition_Prediction.ipynb` | End-to-end ML Colab notebook | ✅ Complete |
| 3 | `Research_Paper_Attrition.docx` | Full academic research paper (10 sections) | ✅ Complete |
| 4 | `Executive_Summary_Attrition.docx` | HR leadership executive summary (9 sections) | ✅ Complete |
| 5 | `requirements.txt` | Dependency file for Streamlit Cloud | ✅ Complete |
| 6 | Risk CSV Export | Employee-level risk scores via notebook Cell 18 | ✅ Complete |

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **ML Models** | XGBoost · Random Forest · Logistic Regression |
| **ML Libraries** | scikit-learn · imbalanced-learn (SMOTE) |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Dashboard** | Streamlit |
| **Visualization** | Matplotlib · Seaborn |
| **Data Processing** | Pandas · NumPy |
| **Notebook** | Google Colab / Jupyter |
| **Deployment** | Streamlit Cloud |

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">


⭐ **Star this repo** if you found it useful!

</div>
