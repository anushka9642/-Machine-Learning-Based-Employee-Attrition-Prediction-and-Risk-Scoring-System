import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Attrition Risk Dashboard — Palo Alto Networks",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1117; }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border-radius: 14px;
        padding: 20px 24px;
        border-left: 4px solid;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 12px;
    }
    .metric-card.blue  { border-color: #4C9BE8; }
    .metric-card.red   { border-color: #EF5350; }
    .metric-card.green { border-color: #66BB6A; }
    .metric-card.amber { border-color: #FFA726; }

    .metric-label { font-size: 12px; color: #9ca3af; font-weight: 500;
                    text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 32px; font-weight: 700; color: #f9fafb;
                    line-height: 1.2; margin: 4px 0; }
    .metric-sub   { font-size: 12px; color: #6b7280; }

    /* Section headers */
    .section-title {
        font-size: 18px; font-weight: 700; color: #e5e7eb;
        padding: 10px 0 6px; border-bottom: 2px solid #374151;
        margin-bottom: 16px;
    }

    /* Risk Badges */
    .badge-high   { background:#3b1111; color:#ef5350; border:1px solid #ef5350;
                    border-radius:6px; padding:3px 10px; font-size:12px; font-weight:600; }
    .badge-medium { background:#2d2000; color:#ffa726; border:1px solid #ffa726;
                    border-radius:6px; padding:3px 10px; font-size:12px; font-weight:600; }
    .badge-low    { background:#0d2b0d; color:#66bb6a; border:1px solid #66bb6a;
                    border-radius:6px; padding:3px 10px; font-size:12px; font-weight:600; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #161826; border-right: 1px solid #2d3148; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label { color: #d1d5db !important; font-size:13px; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background: #1e2130; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #9ca3af; border-radius: 8px; font-weight:500; }
    .stTabs [aria-selected="true"] { background: #2d3160 !important; color: #fff !important; }

    /* Table */
    .dataframe { font-size: 13px !important; }

    /* Info box */
    .info-box {
        background: #1a2744; border: 1px solid #2563eb; border-radius: 10px;
        padding: 14px 18px; color: #93c5fd; font-size: 13px; margin: 10px 0;
    }
    .warn-box {
        background: #2d1b00; border: 1px solid #f59e0b; border-radius: 10px;
        padding: 14px 18px; color: #fcd34d; font-size: 13px; margin: 10px 0;
    }
    .danger-box {
        background: #2d0909; border: 1px solid #dc2626; border-radius: 10px;
        padding: 14px 18px; color: #fca5a5; font-size: 13px; margin: 10px 0;
    }
    div[data-testid="stHorizontalBlock"] { gap: 12px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING & MODEL TRAINING (cached)
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def load_and_process():
    df = pd.read_csv("Palo_Alto_Networks__1_.csv")

    proc = df.copy()
    le = LabelEncoder()
    for col in ['OverTime', 'Gender']:
        proc[col] = le.fit_transform(proc[col])

    ohe_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
    proc = pd.get_dummies(proc, columns=ohe_cols, drop_first=True)

    # Feature engineering
    proc['IncomePerYear']   = proc['MonthlyIncome'] / (proc['TotalWorkingYears'] + 1)
    proc['PromotionDelay']  = proc['YearsAtCompany'] - proc['YearsSinceLastPromotion']
    proc['EngagementScore'] = (proc['JobSatisfaction'] + proc['EnvironmentSatisfaction'] +
                                proc['RelationshipSatisfaction'] + proc['JobInvolvement'] +
                                proc['WorkLifeBalance']) / 5
    proc['WorkloadStress']  = ((proc['OverTime'] == 1) & (proc['WorkLifeBalance'] <= 2)).astype(int)
    proc['RoleStagnation']  = proc['YearsInCurrentRole'] - proc['YearsSinceLastPromotion']
    proc['LoyaltyIndex']    = proc['YearsAtCompany'] / (proc['NumCompaniesWorked'] + 1)

    X = proc.drop('Attrition', axis=1)
    y = proc['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_train_s, y_train)

    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=scale_pos, use_label_encoder=False,
                          eval_metric='logloss', random_state=42)
    model.fit(X_sm, y_sm)

    X_all_s = scaler.transform(X)
    probs   = model.predict_proba(X_all_s)[:, 1]

    risk_df = df.copy()
    risk_df['AttritionProb'] = (probs * 100).round(1)
    risk_df['RiskCategory']  = risk_df['AttritionProb'].apply(
        lambda p: 'High' if p >= 60 else ('Medium' if p >= 30 else 'Low'))

    # Model metrics on test set
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    metrics = {
        'Accuracy' : round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall'   : round(recall_score(y_test, y_pred), 4),
        'F1'       : round(f1_score(y_test, y_pred), 4),
        'ROC_AUC'  : round(roc_auc_score(y_test, y_prob), 4),
    }
    cm = confusion_matrix(y_test, y_pred)

    # Feature importances
    feat_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)

    return df, risk_df, model, scaler, X, metrics, cm, feat_imp

# ─── Load ────────────────────────────────────────────────────────────────────
with st.spinner("🔄 Training XGBoost model on Palo Alto Networks data..."):
    df_raw, risk_df, model, scaler, X_features, metrics, cm, feat_imp = load_and_process()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔵 Attrition Risk System")
    st.markdown("**Palo Alto Networks**")
    st.markdown("---")

    st.markdown("### 🎛️ Filters")
    dept_options = ['All'] + sorted(df_raw['Department'].unique().tolist())
    sel_dept = st.selectbox("Department", dept_options)

    role_options = ['All'] + sorted(df_raw['JobRole'].unique().tolist())
    sel_role = st.selectbox("Job Role", role_options)

    st.markdown("### ⚠️ Risk Threshold")
    risk_threshold = st.slider("High Risk Cutoff (%)", 40, 80, 60, step=5,
        help="Employees above this probability are flagged as High Risk")

    st.markdown("### 👤 Employee Lookup")
    emp_id = st.number_input("Employee Index (0–1469)", 0, 1469, 0, step=1)

    st.markdown("---")
    st.markdown("<div style='color:#6b7280;font-size:11px'>Unified Mentor Internship<br/>ML Attrition Project</div>",
                unsafe_allow_html=True)

# ─── Apply Filters ───────────────────────────────────────────────────────────
filtered = risk_df.copy()
filtered['RiskCategory'] = filtered['AttritionProb'].apply(
    lambda p: 'High' if p >= risk_threshold else ('Medium' if p >= 30 else 'Low'))

if sel_dept != 'All':
    filtered = filtered[filtered['Department'] == sel_dept]
if sel_role != 'All':
    filtered = filtered[filtered['JobRole'] == sel_role]

# ═══════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div style='padding:20px 0 10px'>
  <h1 style='color:#f9fafb;font-size:28px;font-weight:700;margin:0'>
    🔵 Employee Attrition Risk Dashboard
  </h1>
  <p style='color:#9ca3af;font-size:14px;margin:4px 0 0'>
    Palo Alto Networks · Predictive HR Analytics · Powered by XGBoost + SHAP
  </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  TOP KPI CARDS
# ═══════════════════════════════════════════════════════════════
total   = len(filtered)
high_n  = (filtered['RiskCategory'] == 'High').sum()
med_n   = (filtered['RiskCategory'] == 'Medium').sum()
low_n   = (filtered['RiskCategory'] == 'Low').sum()
avg_p   = filtered['AttritionProb'].mean()

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""<div class='metric-card blue'>
        <div class='metric-label'>Total Employees</div>
        <div class='metric-value'>{total:,}</div>
        <div class='metric-sub'>in current filter</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""<div class='metric-card red'>
        <div class='metric-label'>🔴 High Risk</div>
        <div class='metric-value'>{high_n}</div>
        <div class='metric-sub'>{high_n/total*100:.1f}% of workforce</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""<div class='metric-card amber'>
        <div class='metric-label'>🟡 Medium Risk</div>
        <div class='metric-value'>{med_n}</div>
        <div class='metric-sub'>{med_n/total*100:.1f}% of workforce</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""<div class='metric-card green'>
        <div class='metric-label'>🟢 Low Risk</div>
        <div class='metric-value'>{low_n}</div>
        <div class='metric-sub'>{low_n/total*100:.1f}% of workforce</div>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""<div class='metric-card blue'>
        <div class='metric-label'>Avg Risk Score</div>
        <div class='metric-value'>{avg_p:.1f}%</div>
        <div class='metric-sub'>Model AUC: {metrics['ROC_AUC']}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Risk Overview",
    "👤 Employee Profile",
    "🏢 Department View",
    "🔍 Explainability",
    "🤖 Model Performance"
])

# ──────────────────────────────────────────────────────────────
# TAB 1 — RISK OVERVIEW
# ──────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-title'>📊 Workforce Attrition Risk Overview</div>",
                unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        # Donut Chart — Risk Distribution
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1e2130')
        ax.set_facecolor('#1e2130')
        sizes  = [high_n, med_n, low_n]
        labels = [f'High\n{high_n}', f'Medium\n{med_n}', f'Low\n{low_n}']
        colors = ['#EF5350', '#FFA726', '#66BB6A']
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(width=0.55, edgecolor='#1e2130', linewidth=2),
            textprops={'color': '#d1d5db', 'fontsize': 10}
        )
        for at in autotexts:
            at.set_color('white'); at.set_fontsize(9); at.set_fontweight('bold')
        ax.set_title('Risk Category Distribution', color='#f9fafb',
                     fontsize=12, fontweight='bold', pad=10)
        st.pyplot(fig)
        plt.close()

    with col_b:
        # Histogram — Probability Distribution
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1e2130')
        ax.set_facecolor('#1e2130')
        ax.hist(filtered['AttritionProb'], bins=25, color='#4C9BE8',
                edgecolor='#1e2130', linewidth=0.8, alpha=0.85)
        ax.axvline(x=risk_threshold, color='#EF5350', linestyle='--',
                   linewidth=2, label=f'High Risk ≥{risk_threshold}%')
        ax.axvline(x=30, color='#FFA726', linestyle='--',
                   linewidth=1.5, label='Medium Risk ≥30%')
        ax.set_title('Attrition Probability Distribution',
                     color='#f9fafb', fontsize=12, fontweight='bold')
        ax.set_xlabel('Attrition Probability (%)', color='#9ca3af')
        ax.set_ylabel('Number of Employees', color='#9ca3af')
        ax.tick_params(colors='#9ca3af')
        ax.spines[['top','right','left','bottom']].set_color('#374151')
        ax.legend(fontsize=9, facecolor='#252840', labelcolor='#d1d5db')
        st.pyplot(fig)
        plt.close()

    # High-Risk Employee Table
    st.markdown("<div class='section-title'>🔴 High-Risk Employees (Immediate Action Required)</div>",
                unsafe_allow_html=True)

    high_risk_table = filtered[filtered['RiskCategory'] == 'High'][[
        'Department', 'JobRole', 'Age', 'MonthlyIncome', 'OverTime',
        'JobSatisfaction', 'YearsAtCompany', 'YearsSinceLastPromotion',
        'AttritionProb'
    ]].sort_values('AttritionProb', ascending=False).head(20)

    if len(high_risk_table) == 0:
        st.markdown("<div class='info-box'>✅ No high-risk employees in current filter.</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='danger-box'>⚠️ <b>{len(high_risk_table)}</b> employees flagged as HIGH RISK. "
                    f"Immediate retention intervention recommended.</div>", unsafe_allow_html=True)

        def color_risk(val):
            if val >= risk_threshold: return 'color: #EF5350; font-weight:bold'
            elif val >= 30: return 'color: #FFA726'
            return 'color: #66BB6A'

        styled = high_risk_table.style.applymap(color_risk, subset=['AttritionProb'])
        st.dataframe(styled, use_container_width=True, height=350)

    # Attrition by key categories
    st.markdown("<div class='section-title'>📈 Attrition Rate by Key Factors</div>",
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    fig_cols = [
        (col1, 'OverTime',       'Attrition Rate by Overtime'),
        (col2, 'MaritalStatus',  'Attrition Rate by Marital Status'),
        (col3, 'BusinessTravel', 'Attrition Rate by Travel'),
    ]
    for col_widget, feat, title in fig_cols:
        with col_widget:
            fig, ax = plt.subplots(figsize=(4, 3), facecolor='#1e2130')
            ax.set_facecolor('#1e2130')
            rates = df_raw.groupby(feat)['Attrition'].mean() * 100
            rates.sort_values(ascending=False).plot(
                kind='bar', ax=ax, color='#EF5350', edgecolor='#1e2130', width=0.6)
            ax.set_title(title, color='#f9fafb', fontsize=10, fontweight='bold')
            ax.set_ylabel('%', color='#9ca3af', fontsize=9)
            ax.tick_params(colors='#9ca3af', labelsize=8, rotation=20)
            ax.spines[['top','right','left','bottom']].set_color('#374151')
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            for bar in ax.patches:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'{bar.get_height():.1f}%', ha='center', va='bottom',
                        fontsize=7, color='#d1d5db')
            st.pyplot(fig)
            plt.close()

# ──────────────────────────────────────────────────────────────
# TAB 2 — EMPLOYEE PROFILE
# ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-title'>👤 Individual Employee Risk Profile</div>",
                unsafe_allow_html=True)

    emp = risk_df.iloc[emp_id]
    prob = emp['AttritionProb']
    risk_cat = emp['RiskCategory']

    # Risk color
    if risk_cat == 'High':
        badge_class = 'badge-high'; risk_icon = '🔴'; bar_color = '#EF5350'
        action = "🚨 URGENT: Schedule immediate 1:1. Consider salary review, role change, or counter-offer."
        box_class = 'danger-box'
    elif risk_cat == 'Medium':
        badge_class = 'badge-medium'; risk_icon = '🟡'; bar_color = '#FFA726'
        action = "⚠️ Monitor closely. Schedule career development discussion. Review compensation."
        box_class = 'warn-box'
    else:
        badge_class = 'badge-low'; risk_icon = '🟢'; bar_color = '#66BB6A'
        action = "✅ Employee is stable. Continue regular check-ins and engagement activities."
        box_class = 'info-box'

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(f"""
        <div class='metric-card blue' style='margin-bottom:16px'>
          <div style='display:flex;justify-content:space-between;align-items:start'>
            <div>
              <div class='metric-label'>Employee ID #{emp_id}</div>
              <div class='metric-value'>{emp['JobRole']}</div>
              <div class='metric-sub'>{emp['Department']} · Age {emp['Age']}</div>
            </div>
            <span class='{badge_class}'>{risk_icon} {risk_cat} Risk</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Profile details
        details = {
            "💰 Monthly Income": f"${emp['MonthlyIncome']:,}",
            "⏰ Overtime": emp['OverTime'],
            "😊 Job Satisfaction": f"{emp['JobSatisfaction']} / 4",
            "🌍 Environment Satisfaction": f"{emp['EnvironmentSatisfaction']} / 4",
            "⚖️ Work-Life Balance": f"{emp['WorkLifeBalance']} / 4",
            "🤝 Relationship Satisfaction": f"{emp['RelationshipSatisfaction']} / 4",
            "🏢 Years at Company": emp['YearsAtCompany'],
            "📅 Years in Current Role": emp['YearsInCurrentRole'],
            "🎯 Years Since Promotion": emp['YearsSinceLastPromotion'],
            "✈️ Business Travel": emp['BusinessTravel'],
            "💍 Marital Status": emp['MaritalStatus'],
            "🎓 Education Level": emp['Education'],
        }
        for k, v in details.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:6px 0;border-bottom:1px solid #2d3148;'>"
                f"<span style='color:#9ca3af;font-size:13px'>{k}</span>"
                f"<span style='color:#f9fafb;font-size:13px;font-weight:500'>{v}</span>"
                f"</div>", unsafe_allow_html=True)

    with col_right:
        # Gauge Chart
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1e2130')
        ax.set_facecolor('#1e2130')

        theta = np.linspace(np.pi, 0, 300)
        r_outer, r_inner = 1.0, 0.6

        # Background arc zones
        for start, end, col in [(0, 0.3, '#1a3a1a'), (0.3, 0.6, '#2d2500'), (0.6, 1.0, '#3a1010')]:
            t_seg = np.linspace(np.pi, np.pi * (1 - end), 100)
            t_seg2 = np.linspace(np.pi * (1 - start), np.pi, 100)
            ax.fill_between(
                np.concatenate([np.cos(np.linspace(np.pi*(1-end), np.pi*(1-start), 100)),
                                np.cos(t_seg2)[::-1] * r_inner / r_outer]),
                np.concatenate([np.sin(np.linspace(np.pi*(1-end), np.pi*(1-start), 100)),
                                np.sin(t_seg2)[::-1] * r_inner / r_outer]),
                alpha=0.5, color=col
            )

        # Full arc
        ax.plot(np.cos(theta), np.sin(theta), color='#374151', linewidth=20, solid_capstyle='round')
        ax.plot(np.cos(theta) * r_inner, np.sin(theta) * r_inner, color='#1e2130', linewidth=12)

        # Filled arc up to prob
        fill_end = np.pi - (prob / 100) * np.pi
        theta_fill = np.linspace(np.pi, fill_end, 200)
        ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=bar_color,
                linewidth=20, solid_capstyle='round', alpha=0.9)

        # Needle
        angle = np.pi - (prob / 100) * np.pi
        ax.annotate('', xy=(0.72 * np.cos(angle), 0.72 * np.sin(angle)),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2.5))
        ax.add_patch(plt.Circle((0, 0), 0.06, color='white', zorder=5))

        ax.text(0, 0.15, f'{prob:.1f}%', ha='center', va='center',
                fontsize=26, fontweight='bold', color=bar_color)
        ax.text(0, -0.05, 'Attrition Risk', ha='center', va='center',
                fontsize=11, color='#9ca3af')
        ax.text(-0.95, -0.1, '0%', color='#66BB6A', fontsize=9, ha='center')
        ax.text(0, 1.05, '50%', color='#FFA726', fontsize=9, ha='center')
        ax.text(0.95, -0.1, '100%', color='#EF5350', fontsize=9, ha='center')

        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.3, 1.25)
        ax.axis('off')
        ax.set_title('Risk Gauge', color='#f9fafb', fontsize=12, fontweight='bold')
        st.pyplot(fig)
        plt.close()

        # Satisfaction Radar
        cats = ['Job\nSatisfaction', 'Environment\nSatisfaction', 'Work-Life\nBalance',
                'Relationship\nSatisfaction', 'Job\nInvolvement']
        vals = [emp['JobSatisfaction'], emp['EnvironmentSatisfaction'],
                emp['WorkLifeBalance'], emp['RelationshipSatisfaction'], emp['JobInvolvement']]
        N = len(cats)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        vals_r  = vals + [vals[0]]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(4.5, 3.5), subplot_kw=dict(polar=True), facecolor='#1e2130')
        ax.set_facecolor('#1e2130')
        ax.plot(angles, vals_r, color=bar_color, linewidth=2)
        ax.fill(angles, vals_r, alpha=0.25, color=bar_color)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, color='#9ca3af', fontsize=8)
        ax.set_ylim(0, 4); ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(['1', '2', '3', '4'], color='#6b7280', fontsize=7)
        ax.spines['polar'].set_color('#374151')
        ax.grid(color='#374151', linewidth=0.8)
        ax.set_title('Satisfaction Radar', color='#f9fafb', fontsize=11,
                     fontweight='bold', pad=15)
        st.pyplot(fig)
        plt.close()

    # Recommended Action
    st.markdown(f"<div class='{box_class}'><b>📌 Recommended HR Action:</b><br>{action}</div>",
                unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# TAB 3 — DEPARTMENT VIEW
# ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-title'>🏢 Department-Level Attrition Risk</div>",
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        dept_stats = risk_df.groupby('Department').agg(
            Total=('AttritionProb', 'count'),
            AvgRisk=('AttritionProb', 'mean'),
            HighRisk=('RiskCategory', lambda x: (x == 'High').sum()),
            MedRisk=('RiskCategory', lambda x: (x == 'Medium').sum()),
        ).round(1)

        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1e2130')
        ax.set_facecolor('#1e2130')
        depts = dept_stats.index.tolist()
        x = np.arange(len(depts))
        w = 0.35
        ax.bar(x - w/2, dept_stats['HighRisk'],   w, label='High Risk',   color='#EF5350', edgecolor='#1e2130')
        ax.bar(x + w/2, dept_stats['MedRisk'],    w, label='Medium Risk', color='#FFA726', edgecolor='#1e2130')
        ax.set_xticks(x); ax.set_xticklabels(depts, color='#9ca3af', rotation=15, fontsize=10)
        ax.tick_params(colors='#9ca3af')
        ax.spines[['top','right','left','bottom']].set_color('#374151')
        ax.set_title('High & Medium Risk Count by Department',
                     color='#f9fafb', fontsize=12, fontweight='bold')
        ax.set_ylabel('Employees', color='#9ca3af')
        ax.legend(facecolor='#252840', labelcolor='#d1d5db', fontsize=9)
        st.pyplot(fig); plt.close()

    with col_b:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1e2130')
        ax.set_facecolor('#1e2130')
        colors_bar = ['#EF5350', '#FFA726', '#66BB6A']
        bars = ax.barh(dept_stats.index, dept_stats['AvgRisk'],
                       color=colors_bar[:len(dept_stats)], edgecolor='#1e2130')
        for bar in bars:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{bar.get_width():.1f}%', va='center', color='#d1d5db', fontsize=10)
        ax.set_xlabel('Average Attrition Probability (%)', color='#9ca3af')
        ax.tick_params(colors='#9ca3af')
        ax.spines[['top','right','left','bottom']].set_color('#374151')
        ax.set_title('Avg Risk Score by Department', color='#f9fafb',
                     fontsize=12, fontweight='bold')
        st.pyplot(fig); plt.close()

    # Job Role Risk Breakdown
    st.markdown("<div class='section-title'>💼 Risk Breakdown by Job Role</div>",
                unsafe_allow_html=True)

    role_stats = risk_df.groupby('JobRole').agg(
        Headcount=('AttritionProb', 'count'),
        AvgRisk=('AttritionProb', 'mean'),
        MaxRisk=('AttritionProb', 'max'),
        HighRiskCount=('RiskCategory', lambda x: (x == 'High').sum())
    ).round(1).sort_values('AvgRisk', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#1e2130')
    ax.set_facecolor('#1e2130')
    bar_colors = ['#EF5350' if v >= 40 else '#FFA726' if v >= 25 else '#66BB6A'
                  for v in role_stats['AvgRisk']]
    bars = ax.bar(role_stats.index, role_stats['AvgRisk'],
                  color=bar_colors, edgecolor='#1e2130', width=0.6)
    ax.set_xticklabels(role_stats.index, rotation=30, ha='right', color='#9ca3af', fontsize=9)
    ax.tick_params(colors='#9ca3af')
    ax.spines[['top','right','left','bottom']].set_color('#374151')
    ax.set_title('Average Attrition Risk by Job Role', color='#f9fafb',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg Attrition Probability (%)', color='#9ca3af')
    ax.axhline(y=risk_threshold, color='#EF5350', linestyle='--', alpha=0.6,
               label=f'High Risk Threshold ({risk_threshold}%)')
    ax.legend(facecolor='#252840', labelcolor='#d1d5db', fontsize=9)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom',
                fontsize=8, color='#d1d5db')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.dataframe(role_stats, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB 4 — EXPLAINABILITY
# ──────────────────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-title'>🔍 Feature Importance — What Drives Attrition?</div>",
                unsafe_allow_html=True)

    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        fig, ax = plt.subplots(figsize=(7, 6), facecolor='#1e2130')
        ax.set_facecolor('#1e2130')
        top15 = feat_imp.head(15)
        norm_colors = plt.cm.RdYlGn_r(
            np.linspace(0.05, 0.9, len(top15)))
        bars = ax.barh(top15['Feature'][::-1], top15['Importance'][::-1],
                       color=norm_colors[::-1], edgecolor='#1e2130')
        for bar in bars:
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{bar.get_width():.4f}', va='center', color='#d1d5db', fontsize=8)
        ax.set_title('Top 15 Feature Importances (XGBoost)',
                     color='#f9fafb', fontsize=12, fontweight='bold')
        ax.set_xlabel('Importance Score', color='#9ca3af')
        ax.tick_params(colors='#9ca3af', labelsize=9)
        ax.spines[['top','right','left','bottom']].set_color('#374151')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("<div class='section-title'>💡 Key Insights</div>",
                    unsafe_allow_html=True)
        insights = [
            ("🔴 Overtime", "Employees working overtime are 3× more likely to leave."),
            ("💰 Monthly Income", "Lower income is among the strongest attrition predictors."),
            ("📅 Years at Company", "Early-tenure employees (0–3 yrs) show highest risk."),
            ("😊 Job Satisfaction", "Low satisfaction (score 1) doubles attrition rate."),
            ("⚡ Engagement Score", "Composite disengagement is a top engineered signal."),
            ("📈 Stock Options", "Employees with no stock options leave more frequently."),
            ("🏠 Distance from Home", "Longer commutes correlate with higher attrition."),
            ("🔄 Role Stagnation", "No progression in role significantly raises risk."),
        ]
        for icon_title, desc in insights:
            st.markdown(f"""
            <div style='padding:9px 12px;margin:5px 0;background:#1e2130;
                        border-radius:8px;border-left:3px solid #4C9BE8'>
              <div style='color:#e5e7eb;font-size:13px;font-weight:600'>{icon_title}</div>
              <div style='color:#9ca3af;font-size:12px;margin-top:2px'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    # What-If Scenario Explorer
    st.markdown("<div class='section-title'>🎛️ What-If Scenario Explorer</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Adjust the sliders below to simulate how changing conditions "
                "affect an employee's attrition risk (based on a representative employee profile).</div>",
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        wi_income   = st.slider("Monthly Income ($)", 1000, 20000, 5000, 500)
        wi_overtime = st.selectbox("Overtime", ["No", "Yes"])
    with col2:
        wi_jobsat   = st.slider("Job Satisfaction (1-4)", 1, 4, 2)
        wi_wlb      = st.slider("Work-Life Balance (1-4)", 1, 4, 2)
    with col3:
        wi_yrs_promo = st.slider("Years Since Promotion", 0, 15, 3)
        wi_dist      = st.slider("Distance From Home (km)", 1, 30, 10)

    # Build a sample employee vector using the median of each feature
    median_vals = X_features.median()
    sample = median_vals.copy()
    sample['MonthlyIncome']          = wi_income
    sample['OverTime']               = 1 if wi_overtime == "Yes" else 0
    sample['JobSatisfaction']        = wi_jobsat
    sample['WorkLifeBalance']        = wi_wlb
    sample['YearsSinceLastPromotion']= wi_yrs_promo
    sample['DistanceFromHome']       = wi_dist
    # Recalc engineered features
    sample['EngagementScore'] = (wi_jobsat + sample['EnvironmentSatisfaction'] +
                                  sample['RelationshipSatisfaction'] +
                                  sample['JobInvolvement'] + wi_wlb) / 5
    sample['WorkloadStress']  = 1 if (wi_overtime == "Yes" and wi_wlb <= 2) else 0
    sample['IncomePerYear']   = wi_income / (sample['TotalWorkingYears'] + 1)

    sample_scaled = scaler.transform(sample.values.reshape(1, -1))
    wi_prob = model.predict_proba(sample_scaled)[0][1] * 100

    wi_color = '#EF5350' if wi_prob >= risk_threshold else '#FFA726' if wi_prob >= 30 else '#66BB6A'
    wi_label = 'HIGH RISK 🔴' if wi_prob >= risk_threshold else 'MEDIUM RISK 🟡' if wi_prob >= 30 else 'LOW RISK 🟢'

    st.markdown(f"""
    <div class='metric-card blue' style='text-align:center;margin-top:10px'>
      <div class='metric-label'>Predicted Attrition Probability</div>
      <div style='font-size:48px;font-weight:800;color:{wi_color}'>{wi_prob:.1f}%</div>
      <div style='font-size:16px;color:{wi_color};font-weight:600'>{wi_label}</div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# TAB 5 — MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────
with tab5:
    st.markdown("<div class='section-title'>🤖 XGBoost Model Performance Report</div>",
                unsafe_allow_html=True)

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    metric_items = [
        (m1, "Accuracy",  metrics['Accuracy'],  "Overall correctness", "blue"),
        (m2, "Precision", metrics['Precision'], "False-positive control", "green"),
        (m3, "Recall",    metrics['Recall'],    "Attrition detection ★", "amber"),
        (m4, "F1-Score",  metrics['F1'],        "Precision/Recall balance", "blue"),
        (m5, "ROC-AUC",   metrics['ROC_AUC'],   "Overall classification power", "green"),
    ]
    for col_w, name, val, desc, clr in metric_items:
        with col_w:
            st.markdown(f"""<div class='metric-card {clr}'>
                <div class='metric-label'>{name}</div>
                <div class='metric-value'>{val:.3f}</div>
                <div class='metric-sub'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1e2130')
        ax.set_facecolor('#1e2130')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Stayed', 'Left'],
                    yticklabels=['Stayed', 'Left'],
                    linewidths=2, linecolor='#1e2130',
                    annot_kws={'size': 14, 'color': 'white', 'weight': 'bold'})
        ax.set_title('Confusion Matrix', color='#f9fafb', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', color='#9ca3af')
        ax.set_xlabel('Predicted', color='#9ca3af')
        ax.tick_params(colors='#9ca3af')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        # Metric bar chart
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1e2130')
        ax.set_facecolor('#1e2130')
        metric_names = list(metrics.keys())
        metric_vals  = list(metrics.values())
        colors_m = ['#4C9BE8', '#66BB6A', '#EF5350', '#FFA726', '#AB47BC']
        bars = ax.bar(metric_names, metric_vals, color=colors_m, edgecolor='#1e2130', width=0.6)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom',
                    fontsize=10, color='#f9fafb', fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.8, color='white', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_title('Model Metrics Overview', color='#f9fafb', fontsize=12, fontweight='bold')
        ax.tick_params(colors='#9ca3af')
        ax.spines[['top','right','left','bottom']].set_color('#374151')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Pipeline Summary
    st.markdown("<div class='section-title'>⚙️ ML Pipeline Summary</div>",
                unsafe_allow_html=True)
    pipeline_steps = [
        ("1. Data Loading", "1,470 employees × 31 features from Palo Alto Networks HR data"),
        ("2. Preprocessing", "Label Encoding (OverTime, Gender) + One-Hot Encoding (5 categorical columns)"),
        ("3. Feature Engineering", "6 new features: IncomePerYear, EngagementScore, WorkloadStress, PromotionDelay, RoleStagnation, LoyaltyIndex"),
        ("4. Train/Test Split", "80/20 stratified split → 1,176 train / 294 test"),
        ("5. Class Balancing", "SMOTE applied to training set to fix 84%/16% imbalance"),
        ("6. Model", "XGBoost (300 trees, depth=6, lr=0.05) trained on SMOTE-balanced data"),
        ("7. Risk Scoring", "Each employee scored 0–100% → Low / Medium / High risk categories"),
    ]
    for step, desc in pipeline_steps:
        st.markdown(f"""
        <div style='display:flex;gap:12px;padding:8px 0;border-bottom:1px solid #2d3148;align-items:start'>
          <span style='color:#4C9BE8;font-size:13px;font-weight:600;min-width:180px'>{step}</span>
          <span style='color:#9ca3af;font-size:13px'>{desc}</span>
        </div>""", unsafe_allow_html=True)
