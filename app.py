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
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

# ══════════════════════════════════════════════════════════════
#  DESIGN TOKENS  (single source of truth for all colors)
# ══════════════════════════════════════════════════════════════
BG_PAGE  = '#0d0f1a'   # darkest — page background
BG_CARD  = '#13162a'   # card / chart background
BG_CARD2 = '#1a1e35'   # slightly lighter card
BORDER   = '#2a2f52'   # subtle border lines
TXT_H    = '#e2e8f0'   # headings / large text     → very light slate
TXT_B    = '#a0aec0'   # body text                 → medium slate
TXT_D    = '#718096'   # dimmed / sub-labels       → dim slate
RED      = '#fc8181'   # high risk
AMBER    = '#f6ad55'   # medium risk
GREEN    = '#68d391'   # low risk
BLUE     = '#63b3ed'   # accent / neutral
PURPLE   = '#b794f4'   # accent 2
TEAL     = '#4fd1c5'   # accent 3

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Attrition Risk Dashboard — Palo Alto Networks",
    page_icon="🔵", layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"] {{ font-family:'Inter',sans-serif; }}
  .main, .block-container, .stApp {{ background-color:{BG_PAGE}; }}

  /* KPI Cards */
  .kpi {{
    background:{BG_CARD}; border-radius:16px; padding:18px 22px;
    border:1px solid {BORDER}; margin-bottom:10px;
    box-shadow:0 4px 24px rgba(0,0,0,.45); position:relative; overflow:hidden;
  }}
  .kpi::before {{
    content:''; position:absolute; top:0; left:0; right:0;
    height:3px; border-radius:16px 16px 0 0;
  }}
  .kpi.bl::before {{ background:linear-gradient(90deg,{BLUE},{PURPLE}); }}
  .kpi.rd::before {{ background:linear-gradient(90deg,{RED},#fc8181);  }}
  .kpi.gr::before {{ background:linear-gradient(90deg,{GREEN},{TEAL}); }}
  .kpi.am::before {{ background:linear-gradient(90deg,{AMBER},#fbd38d);}}
  .kpi-lbl {{ font-size:10px; color:{TXT_D}; font-weight:600;
              text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }}
  .kpi-val {{ font-size:32px; font-weight:800; color:{TXT_H}; line-height:1.1; }}
  .kpi-sub {{ font-size:11px; color:{TXT_B}; margin-top:3px; }}

  /* Section titles */
  .stitle {{
    font-size:14px; font-weight:700; color:{TXT_H};
    padding:8px 0; border-bottom:1px solid {BORDER};
    margin-bottom:14px;
  }}

  /* Badges */
  .bh {{ background:rgba(252,129,129,.15); color:{RED};   border:1px solid rgba(252,129,129,.4);
         border-radius:7px; padding:4px 12px; font-size:12px; font-weight:700; }}
  .bm {{ background:rgba(246,173, 85,.15); color:{AMBER}; border:1px solid rgba(246,173, 85,.4);
         border-radius:7px; padding:4px 12px; font-size:12px; font-weight:700; }}
  .bl2{{ background:rgba(104,211,145,.15); color:{GREEN}; border:1px solid rgba(104,211,145,.4);
         border-radius:7px; padding:4px 12px; font-size:12px; font-weight:700; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{ background:#0a0c18; border-right:1px solid {BORDER}; }}
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label {{ color:{TXT_B} !important; font-size:12px; }}
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {{ color:{TXT_H} !important; }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    background:{BG_CARD}; border-radius:12px; padding:4px; border:1px solid {BORDER};
  }}
  .stTabs [data-baseweb="tab"] {{
    color:{TXT_B}; border-radius:8px; font-weight:500; font-size:13px; padding:8px 16px;
  }}
  .stTabs [aria-selected="true"] {{
    background:linear-gradient(135deg,#1e2a5e,#2a1e5e) !important;
    color:{TXT_H} !important;
  }}

  /* Alert boxes */
  .ainfo  {{ background:rgba( 99,179,237,.09); border:1px solid rgba( 99,179,237,.35);
             border-radius:10px; padding:12px 16px; color:{BLUE};  font-size:13px; margin:8px 0; }}
  .awarn  {{ background:rgba(246,173, 85,.09); border:1px solid rgba(246,173, 85,.35);
             border-radius:10px; padding:12px 16px; color:{AMBER}; font-size:13px; margin:8px 0; }}
  .adanger{{ background:rgba(252,129,129,.09); border:1px solid rgba(252,129,129,.35);
             border-radius:10px; padding:12px 16px; color:{RED};   font-size:13px; margin:8px 0; }}
  .aok    {{ background:rgba(104,211,145,.09); border:1px solid rgba(104,211,145,.35);
             border-radius:10px; padding:12px 16px; color:{GREEN}; font-size:13px; margin:8px 0; }}

  /* Profile rows */
  .pr {{ display:flex; justify-content:space-between; align-items:center;
         padding:8px 0; border-bottom:1px solid {BORDER}; }}
  .pk {{ color:{TXT_B}; font-size:13px; }}
  .pv {{ color:{TXT_H}; font-size:13px; font-weight:600; }}

  /* Pipeline rows */
  .piperow {{ display:flex; gap:14px; padding:9px 0;
              border-bottom:1px solid {BORDER}; align-items:start; }}
  .pipestep {{ color:{BLUE};  font-size:12px; font-weight:700; min-width:190px; }}
  .pipedesc {{ color:{TXT_B}; font-size:12px; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  DATA LOADING & MODEL TRAINING
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_and_process():
    df = pd.read_csv("Palo_Alto_Networks__1_.csv")
    proc = df.copy()
    le = LabelEncoder()
    for col in ['OverTime', 'Gender']:
        proc[col] = le.fit_transform(proc[col])
    proc = pd.get_dummies(proc,
        columns=['BusinessTravel','Department','EducationField','JobRole','MaritalStatus'],
        drop_first=True)

    proc['IncomePerYear']   = proc['MonthlyIncome'] / (proc['TotalWorkingYears'] + 1)
    proc['PromotionDelay']  = proc['YearsAtCompany'] - proc['YearsSinceLastPromotion']
    proc['EngagementScore'] = (proc['JobSatisfaction'] + proc['EnvironmentSatisfaction'] +
                                proc['RelationshipSatisfaction'] + proc['JobInvolvement'] +
                                proc['WorkLifeBalance']) / 5
    proc['WorkloadStress']  = ((proc['OverTime']==1) & (proc['WorkLifeBalance']<=2)).astype(int)
    proc['RoleStagnation']  = proc['YearsInCurrentRole'] - proc['YearsSinceLastPromotion']
    proc['LoyaltyIndex']    = proc['YearsAtCompany'] / (proc['NumCompaniesWorked'] + 1)

    X = proc.drop('Attrition', axis=1)
    y = proc['Attrition']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    sc = StandardScaler()
    Xs_tr = sc.fit_transform(X_tr)
    Xs_te = sc.transform(X_te)
    Xs_tr, y_tr = SMOTE(random_state=42).fit_resample(Xs_tr, y_tr)

    mdl = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=(y_tr==0).sum()/(y_tr==1).sum(),
                        use_label_encoder=False, eval_metric='logloss', random_state=42)
    mdl.fit(Xs_tr, y_tr)

    probs   = mdl.predict_proba(sc.transform(X))[:, 1]
    risk_df = df.copy()
    risk_df['AttritionProb'] = (probs * 100).round(1)
    risk_df['RiskCategory']  = risk_df['AttritionProb'].apply(
        lambda p: 'High' if p>=60 else ('Medium' if p>=30 else 'Low'))

    yp = mdl.predict(Xs_te)
    yprob = mdl.predict_proba(Xs_te)[:, 1]
    metrics = {
        'Accuracy' : round(accuracy_score(y_te, yp),  4),
        'Precision': round(precision_score(y_te, yp), 4),
        'Recall'   : round(recall_score(y_te, yp),    4),
        'F1'       : round(f1_score(y_te, yp),        4),
        'ROC_AUC'  : round(roc_auc_score(y_te, yprob),4),
    }
    fi = pd.DataFrame({'Feature':X.columns,'Importance':mdl.feature_importances_})\
           .sort_values('Importance', ascending=False).head(15)

    return df, risk_df, mdl, sc, X, metrics, confusion_matrix(y_te, yp), fi


def cstyle(ax, title='', xlabel='', ylabel='', legend=False):
    """Apply consistent dark-theme chart styling."""
    ax.set_facecolor(BG_CARD)
    ax.figure.set_facecolor(BG_CARD)
    if title:   ax.set_title(title,   color=TXT_H,  fontsize=12, fontweight='bold', pad=10)
    if xlabel:  ax.set_xlabel(xlabel, color=TXT_B,  fontsize=10)
    if ylabel:  ax.set_ylabel(ylabel, color=TXT_B,  fontsize=10)
    ax.tick_params(colors=TXT_B, labelsize=9)
    ax.spines[['top','right','left','bottom']].set_color(BORDER)
    if legend:
        ax.legend(facecolor=BG_CARD2, labelcolor=TXT_B, fontsize=9,
                  edgecolor=BORDER, framealpha=0.95)


# ── Load ──────────────────────────────────────────────────────
with st.spinner("🔄 Training XGBoost model on Palo Alto Networks data…"):
    df_raw, risk_df, model, scaler, X_features, metrics, cm, feat_imp = load_and_process()


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"<h2 style='color:{TXT_H};margin:0'>🔵 AttritionIQ</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{TXT_B};font-size:12px;margin:2px 0 0'>Palo Alto Networks · HR Analytics</p>",
                unsafe_allow_html=True)
    st.markdown(f"<hr style='border-color:{BORDER};margin:12px 0'>", unsafe_allow_html=True)

    st.markdown(f"<p style='color:{TXT_H};font-weight:600;font-size:13px;margin-bottom:6px'>🎛️ Filters</p>",
                unsafe_allow_html=True)
    sel_dept = st.selectbox("Department", ['All'] + sorted(df_raw['Department'].unique().tolist()))
    sel_role = st.selectbox("Job Role",   ['All'] + sorted(df_raw['JobRole'].unique().tolist()))

    st.markdown(f"<hr style='border-color:{BORDER};margin:12px 0'>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{TXT_H};font-weight:600;font-size:13px;margin-bottom:6px'>⚠️ Risk Threshold</p>",
                unsafe_allow_html=True)
    risk_threshold = st.slider("High Risk Cutoff (%)", 40, 80, 60, 5)

    st.markdown(f"<hr style='border-color:{BORDER};margin:12px 0'>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{TXT_H};font-weight:600;font-size:13px;margin-bottom:6px'>👤 Employee Lookup</p>",
                unsafe_allow_html=True)
    emp_id = st.number_input("Employee Index (0–1469)", 0, 1469, 0, step=1)

    st.markdown(f"<hr style='border-color:{BORDER};margin:16px 0'>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{TXT_D};font-size:11px'>Unified Mentor Internship<br/>ML Attrition Project · XGBoost</p>",
                unsafe_allow_html=True)


# ── Filter data ───────────────────────────────────────────────
filtered = risk_df.copy()
filtered['RiskCategory'] = filtered['AttritionProb'].apply(
    lambda p: 'High' if p>=risk_threshold else ('Medium' if p>=30 else 'Low'))
if sel_dept != 'All': filtered = filtered[filtered['Department']==sel_dept]
if sel_role != 'All': filtered = filtered[filtered['JobRole']   ==sel_role]


# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='padding:16px 0 8px;border-bottom:1px solid {BORDER};margin-bottom:20px'>
  <h1 style='color:{TXT_H};font-size:26px;font-weight:800;margin:0;letter-spacing:-.5px'>
    🔵 Employee Attrition Risk Dashboard
  </h1>
  <p style='color:{TXT_B};font-size:13px;margin:5px 0 0'>
    Palo Alto Networks &nbsp;·&nbsp; Predictive HR Analytics &nbsp;·&nbsp;
    Powered by XGBoost + SHAP
  </p>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  KPI ROW
# ══════════════════════════════════════════════════════════════
total  = len(filtered)
high_n = (filtered['RiskCategory']=='High').sum()
med_n  = (filtered['RiskCategory']=='Medium').sum()
low_n  = (filtered['RiskCategory']=='Low').sum()
avg_p  = filtered['AttritionProb'].mean()

for col_w, cls, lbl, val, sub in zip(
    st.columns(5),
    ['bl','rd','am','gr','bl'],
    ['👥 Total Employees','🔴 High Risk','🟡 Medium Risk','🟢 Low Risk','📊 Avg Risk Score'],
    [f'{total:,}', str(high_n), str(med_n), str(low_n), f'{avg_p:.1f}%'],
    ['in current filter',
     f'{high_n/total*100:.1f}% of workforce',
     f'{med_n/total*100:.1f}% of workforce',
     f'{low_n/total*100:.1f}% of workforce',
     f'Model AUC: {metrics["ROC_AUC"]}']
):
    with col_w:
        st.markdown(f"""
        <div class='kpi {cls}'>
          <div class='kpi-lbl'>{lbl}</div>
          <div class='kpi-val'>{val}</div>
          <div class='kpi-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📊 Risk Overview","👤 Employee Profile",
    "🏢 Department View","🔍 Explainability","🤖 Model Performance"
])


# ─────────────────────────────────────────────────────────────
# TAB 1 — RISK OVERVIEW
# ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='stitle'>📊 Workforce Attrition Risk Overview</div>",
                unsafe_allow_html=True)

    ca, cb = st.columns(2)

    with ca:   # Donut
        fig, ax = plt.subplots(figsize=(5,4), facecolor=BG_CARD)
        ax.set_facecolor(BG_CARD)
        wedges, texts, autos = ax.pie(
            [high_n, med_n, low_n],
            labels=[f'High  {high_n}', f'Medium  {med_n}', f'Low  {low_n}'],
            colors=[RED, AMBER, GREEN], autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(width=0.58, edgecolor=BG_CARD, linewidth=2.5),
            textprops={'color': TXT_B, 'fontsize': 10, 'fontweight': '600'}
        )
        for a in autos:
            a.set_color(BG_PAGE); a.set_fontsize(9); a.set_fontweight('bold')
        ax.set_title('Risk Category Distribution', color=TXT_H, fontsize=12,
                     fontweight='bold', pad=12)
        st.pyplot(fig); plt.close()

    with cb:   # Histogram
        fig, ax = plt.subplots(figsize=(5,4), facecolor=BG_CARD)
        ax.hist(filtered['AttritionProb'], bins=25, color=BLUE,
                edgecolor=BG_CARD, linewidth=0.8, alpha=0.85)
        ax.axvline(x=risk_threshold, color=RED,   linestyle='--', linewidth=2,
                   label=f'High Risk ≥{risk_threshold}%')
        ax.axvline(x=30,             color=AMBER, linestyle='--', linewidth=1.8,
                   label='Medium ≥30%')
        cstyle(ax, 'Attrition Probability Distribution',
               'Attrition Probability (%)', 'Employees', legend=True)
        st.pyplot(fig); plt.close()

    # High-Risk Table
    st.markdown("<div class='stitle'>🔴 High-Risk Employees — Immediate Action Required</div>",
                unsafe_allow_html=True)
    hrt = filtered[filtered['RiskCategory']=='High'][[
        'Department','JobRole','Age','MonthlyIncome','OverTime',
        'JobSatisfaction','YearsAtCompany','YearsSinceLastPromotion','AttritionProb'
    ]].sort_values('AttritionProb', ascending=False).head(20)

    if len(hrt)==0:
        st.markdown("<div class='aok'>✅ No high-risk employees in current filter.</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='adanger'>⚠️ <b>{len(hrt)}</b> employees flagged HIGH RISK — "
                    f"immediate retention intervention recommended.</div>", unsafe_allow_html=True)
        def cr(v):
            if v >= risk_threshold: return f'color:{RED};font-weight:700'
            elif v >= 30:           return f'color:{AMBER};font-weight:600'
            return f'color:{GREEN}'
        st.dataframe(hrt.style.map(cr, subset=['AttritionProb']),
             use_container_width=True, height=340)

    # Mini bar charts
    st.markdown("<div class='stitle'>📈 Attrition Rate by Key Factors</div>",
                unsafe_allow_html=True)
    for col_w, feat, title in zip(
        st.columns(3),
        ['OverTime','MaritalStatus','BusinessTravel'],
        ['By Overtime Status','By Marital Status','By Business Travel']
    ):
        with col_w:
            fig, ax = plt.subplots(figsize=(4,3), facecolor=BG_CARD)
            rates = df_raw.groupby(feat)['Attrition'].mean()*100
            rates.sort_values(ascending=False).plot(
                kind='bar', ax=ax,
                color=[RED,AMBER,GREEN,BLUE,PURPLE][:len(rates)],
                edgecolor=BG_CARD, width=0.65)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.tick_params(axis='x', rotation=20)
            cstyle(ax, title, ylabel='Rate (%)')
            for bar in ax.patches:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                        f'{bar.get_height():.1f}%', ha='center', va='bottom',
                        fontsize=8, color=TXT_B, fontweight='600')
            st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────────────────────
# TAB 2 — EMPLOYEE PROFILE
# ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='stitle'>👤 Individual Employee Risk Profile</div>",
                unsafe_allow_html=True)

    emp      = risk_df.iloc[emp_id]
    prob     = emp['AttritionProb']
    risk_cat = emp['RiskCategory']

    if risk_cat=='High':
        badge_cls,risk_icon,bar_color = 'bh','🔴',RED
        action  = "🚨 URGENT: Schedule immediate 1:1. Consider salary review, role change, or counter-offer."
        box_cls = 'adanger'
    elif risk_cat=='Medium':
        badge_cls,risk_icon,bar_color = 'bm','🟡',AMBER
        action  = "⚠️ Monitor closely. Schedule career development discussion. Review compensation."
        box_cls = 'awarn'
    else:
        badge_cls,risk_icon,bar_color = 'bl2','🟢',GREEN
        action  = "✅ Employee is stable. Continue regular check-ins and engagement activities."
        box_cls = 'aok'

    cl, cr2 = st.columns([1,1])

    with cl:
        st.markdown(f"""
        <div class='kpi bl' style='margin-bottom:16px'>
          <div style='display:flex;justify-content:space-between;align-items:start'>
            <div>
              <div class='kpi-lbl'>Employee ID #{emp_id}</div>
              <div style='font-size:20px;font-weight:800;color:{TXT_H};margin:4px 0 2px'>
                {emp['JobRole']}</div>
              <div style='color:{TXT_B};font-size:13px'>{emp['Department']} · Age {emp['Age']}</div>
            </div>
            <span class='{badge_cls}'>{risk_icon} {risk_cat} Risk</span>
          </div>
        </div>""", unsafe_allow_html=True)

        for k, v in {
            "💰 Monthly Income":           f"${emp['MonthlyIncome']:,}",
            "⏰ Overtime":                 emp['OverTime'],
            "😊 Job Satisfaction":         f"{emp['JobSatisfaction']} / 4",
            "🌍 Environment Satisfaction": f"{emp['EnvironmentSatisfaction']} / 4",
            "⚖️ Work-Life Balance":        f"{emp['WorkLifeBalance']} / 4",
            "🤝 Relationship Satisfaction":f"{emp['RelationshipSatisfaction']} / 4",
            "🏢 Years at Company":         emp['YearsAtCompany'],
            "📅 Years in Current Role":    emp['YearsInCurrentRole'],
            "🎯 Years Since Promotion":    emp['YearsSinceLastPromotion'],
            "✈️ Business Travel":          emp['BusinessTravel'],
            "💍 Marital Status":           emp['MaritalStatus'],
            "🎓 Education Level":          emp['Education'],
        }.items():
            st.markdown(
                f"<div class='pr'><span class='pk'>{k}</span>"
                f"<span class='pv'>{v}</span></div>",
                unsafe_allow_html=True)

    with cr2:
        # Gauge
        fig, ax = plt.subplots(figsize=(5,4), facecolor=BG_CARD)
        ax.set_facecolor(BG_CARD)
        theta = np.linspace(np.pi, 0, 300)
        # Track
        ax.plot(np.cos(theta), np.sin(theta), color='#1e2248',
                linewidth=22, solid_capstyle='round')
        # Zone tints
        for lo, hi, c in [(0,.30,f'{GREEN}40'),(0.30,.60,f'{AMBER}40'),(0.60,1.0,f'{RED}40')]:
            t = np.linspace(np.pi, np.pi*(1-hi), 120)
            ax.plot(np.cos(t), np.sin(t), color=c, linewidth=22, solid_capstyle='butt')
        # Progress
        ax.plot(np.cos(np.linspace(np.pi, np.pi-(prob/100)*np.pi, 200)),
                np.sin(np.linspace(np.pi, np.pi-(prob/100)*np.pi, 200)),
                color=bar_color, linewidth=22, solid_capstyle='round', alpha=0.95)
        # Hollow
        ax.plot(np.cos(theta)*0.62, np.sin(theta)*0.62,
                color=BG_CARD, linewidth=14, solid_capstyle='round')
        # Needle
        ang = np.pi-(prob/100)*np.pi
        ax.annotate('', xy=(.75*np.cos(ang),.75*np.sin(ang)), xytext=(0,0),
                    arrowprops=dict(arrowstyle='->', color=TXT_H, lw=2.5))
        ax.add_patch(plt.Circle((0,0),.07, color=TXT_H,  zorder=6))
        ax.add_patch(plt.Circle((0,0),.04, color=BG_CARD, zorder=7))
        ax.text(0,.18, f'{prob:.1f}%', ha='center', va='center',
                fontsize=28, fontweight='800', color=bar_color)
        ax.text(0,-.04, 'Attrition Risk', ha='center', va='center',
                fontsize=11, color=TXT_B)
        ax.text(-.92,-.08,'0%',   color=GREEN, fontsize=9, ha='center', fontweight='700')
        ax.text(  0,1.08, '50%',  color=AMBER, fontsize=9, ha='center', fontweight='700')
        ax.text( .92,-.08,'100%', color=RED,   fontsize=9, ha='center', fontweight='700')
        ax.set_xlim(-1.2,1.2); ax.set_ylim(-.3,1.3); ax.axis('off')
        ax.set_title('Risk Gauge', color=TXT_H, fontsize=12, fontweight='bold')
        st.pyplot(fig); plt.close()

        # Radar
        cats   = ['Job\nSatisf.','Environ.\nSatisf.','Work-Life\nBalance',
                  'Relation.\nSatisf.','Job\nInvolv.']
        vals   = [emp['JobSatisfaction'], emp['EnvironmentSatisfaction'],
                  emp['WorkLifeBalance'],  emp['RelationshipSatisfaction'], emp['JobInvolvement']]
        angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
        vr = vals+[vals[0]]; ag = angles+angles[:1]
        fig, ax = plt.subplots(figsize=(4.5,3.5), subplot_kw=dict(polar=True), facecolor=BG_CARD)
        ax.set_facecolor(BG_CARD)
        ax.plot(ag, vr, color=bar_color, linewidth=2.5)
        ax.fill(ag, vr, alpha=0.22, color=bar_color)
        ax.set_xticks(angles)
        ax.set_xticklabels(cats, color=TXT_B, fontsize=8)
        ax.set_ylim(0,4); ax.set_yticks([1,2,3,4])
        ax.set_yticklabels(['1','2','3','4'], color=TXT_D, fontsize=7)
        ax.spines['polar'].set_color(BORDER)
        ax.grid(color=BORDER, linewidth=0.8)
        ax.set_title('Satisfaction Radar', color=TXT_H, fontsize=11,
                     fontweight='bold', pad=14)
        st.pyplot(fig); plt.close()

    st.markdown(f"<div class='{box_cls}'><b>📌 Recommended HR Action:</b><br>{action}</div>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TAB 3 — DEPARTMENT VIEW
# ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='stitle'>🏢 Department-Level Attrition Risk</div>",
                unsafe_allow_html=True)

    dept_stats = risk_df.groupby('Department').agg(
        Total   =('AttritionProb','count'),
        AvgRisk =('AttritionProb','mean'),
        HighRisk=('RiskCategory', lambda x:(x=='High').sum()),
        MedRisk =('RiskCategory', lambda x:(x=='Medium').sum()),
    ).round(1)

    ca, cb = st.columns(2)
    with ca:
        fig, ax = plt.subplots(figsize=(6,4), facecolor=BG_CARD)
        x = np.arange(len(dept_stats)); w = 0.35
        ax.bar(x-w/2, dept_stats['HighRisk'], w, label='High Risk',   color=RED,   edgecolor=BG_CARD, alpha=0.9)
        ax.bar(x+w/2, dept_stats['MedRisk'],  w, label='Medium Risk', color=AMBER, edgecolor=BG_CARD, alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(dept_stats.index, rotation=15, fontsize=10)
        cstyle(ax, 'Risk Count by Department', ylabel='Employees', legend=True)
        st.pyplot(fig); plt.close()

    with cb:
        fig, ax = plt.subplots(figsize=(6,4), facecolor=BG_CARD)
        dc = [RED if v>=40 else AMBER if v>=25 else GREEN for v in dept_stats['AvgRisk']]
        bars = ax.barh(dept_stats.index, dept_stats['AvgRisk'],
                       color=dc, edgecolor=BG_CARD, height=0.55, alpha=0.9)
        for bar in bars:
            ax.text(bar.get_width()+.4, bar.get_y()+bar.get_height()/2,
                    f'{bar.get_width():.1f}%', va='center',
                    color=TXT_H, fontsize=11, fontweight='700')
        cstyle(ax, 'Avg Risk Score by Department', xlabel='Avg Probability (%)')
        st.pyplot(fig); plt.close()

    st.markdown("<div class='stitle'>💼 Risk Breakdown by Job Role</div>",
                unsafe_allow_html=True)
    role_stats = risk_df.groupby('JobRole').agg(
        Headcount    =('AttritionProb','count'),
        AvgRisk      =('AttritionProb','mean'),
        MaxRisk      =('AttritionProb','max'),
        HighRiskCount=('RiskCategory', lambda x:(x=='High').sum())
    ).round(1).sort_values('AvgRisk', ascending=False)

    fig, ax = plt.subplots(figsize=(10,4), facecolor=BG_CARD)
    rc = [RED if v>=40 else AMBER if v>=25 else GREEN for v in role_stats['AvgRisk']]
    bars = ax.bar(role_stats.index, role_stats['AvgRisk'],
                  color=rc, edgecolor=BG_CARD, width=0.65, alpha=0.9)
    ax.axhline(y=risk_threshold, color=RED, linestyle='--', alpha=0.7, linewidth=1.8,
               label=f'High Risk Threshold ({risk_threshold}%)')
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.4,
                f'{bar.get_height():.1f}%', ha='center', va='bottom',
                fontsize=8, color=TXT_H, fontweight='600')
    ax.set_xticklabels(role_stats.index, rotation=30, ha='right')
    cstyle(ax, 'Average Attrition Risk by Job Role', ylabel='Avg Probability (%)', legend=True)
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    st.dataframe(role_stats, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# TAB 4 — EXPLAINABILITY
# ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='stitle'>🔍 Feature Importance — What Drives Attrition?</div>",
                unsafe_allow_html=True)

    ca, cb = st.columns([1.3,1])
    with ca:
        fig, ax = plt.subplots(figsize=(7,6), facecolor=BG_CARD)
        top15 = feat_imp.head(15)
        # Red→Blue gradient: top feature = RED, bottom = BLUE
        n = len(top15)
        grad = [
            '#{:02x}{:02x}{:02x}'.format(
                int(252-(252- 99)*i/(n-1)),
                int(129+(179-129)*i/(n-1)),
                int(129+(237-129)*i/(n-1))
            ) for i in range(n)
        ]
        bars = ax.barh(top15['Feature'][::-1], top15['Importance'][::-1],
                       color=grad[::-1], edgecolor=BG_CARD, height=0.7)
        for bar in bars:
            ax.text(bar.get_width()+.001, bar.get_y()+bar.get_height()/2,
                    f'{bar.get_width():.4f}', va='center',
                    color=TXT_H, fontsize=9, fontweight='600')
        cstyle(ax, 'Top 15 Feature Importances (XGBoost)', xlabel='Importance Score')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with cb:
        st.markdown("<div class='stitle'>💡 Key Attrition Insights</div>",
                    unsafe_allow_html=True)
        for title, desc, accent in [
            ("🔴 Overtime",           "Employees working overtime are 3× more likely to leave.", RED),
            ("💰 Monthly Income",     "Lower income is among the strongest attrition predictors.", AMBER),
            ("📅 Years at Company",   "Early-tenure employees (0–3 yrs) show highest risk.", BLUE),
            ("😊 Job Satisfaction",   "Low satisfaction (score 1) doubles attrition rate.", GREEN),
            ("⚡ Engagement Score",   "Composite disengagement is a top engineered signal.", PURPLE),
            ("📈 Stock Options",      "Employees with no stock options leave more frequently.", TEAL),
            ("🏠 Distance from Home", "Longer commutes correlate with higher attrition.", AMBER),
            ("🔄 Role Stagnation",    "No progression in role significantly raises risk.", RED),
        ]:
            st.markdown(f"""
            <div style='padding:9px 14px;margin:5px 0;background:{BG_CARD2};
                        border-radius:10px;border-left:3px solid {accent}'>
              <div style='color:{TXT_H};font-size:13px;font-weight:600'>{title}</div>
              <div style='color:{TXT_B};font-size:12px;margin-top:3px'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    # What-If
    st.markdown("<div class='stitle'>🎛️ What-If Scenario Explorer</div>",
                unsafe_allow_html=True)
    st.markdown(f"<div class='ainfo'>Adjust the sliders to simulate how changing conditions "
                f"affect an employee's attrition risk.</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        wi_income   = st.slider("Monthly Income ($)", 1000, 20000, 5000, 500)
        wi_overtime = st.selectbox("Overtime", ["No","Yes"])
    with c2:
        wi_jobsat = st.slider("Job Satisfaction (1–4)", 1, 4, 2)
        wi_wlb    = st.slider("Work-Life Balance (1–4)", 1, 4, 2)
    with c3:
        wi_promo = st.slider("Years Since Promotion", 0, 15, 3)
        wi_dist  = st.slider("Distance From Home (km)", 1, 30, 10)

    samp = X_features.median().copy()
    samp['MonthlyIncome']          = wi_income
    samp['OverTime']               = 1 if wi_overtime=="Yes" else 0
    samp['JobSatisfaction']        = wi_jobsat
    samp['WorkLifeBalance']        = wi_wlb
    samp['YearsSinceLastPromotion']= wi_promo
    samp['DistanceFromHome']       = wi_dist
    samp['EngagementScore'] = (wi_jobsat + samp['EnvironmentSatisfaction'] +
                                samp['RelationshipSatisfaction'] +
                                samp['JobInvolvement'] + wi_wlb) / 5
    samp['WorkloadStress']  = 1 if (wi_overtime=="Yes" and wi_wlb<=2) else 0
    samp['IncomePerYear']   = wi_income / (samp['TotalWorkingYears'] + 1)

    wi_prob  = model.predict_proba(scaler.transform(samp.values.reshape(1,-1)))[0][1]*100
    wi_color = RED if wi_prob>=risk_threshold else AMBER if wi_prob>=30 else GREEN
    wi_label = 'HIGH RISK 🔴' if wi_prob>=risk_threshold else \
               'MEDIUM RISK 🟡' if wi_prob>=30 else 'LOW RISK 🟢'

    st.markdown(f"""
    <div class='kpi bl' style='text-align:center;margin-top:12px;padding:28px'>
      <div class='kpi-lbl'>Predicted Attrition Probability</div>
      <div style='font-size:56px;font-weight:900;color:{wi_color};line-height:1.1'>{wi_prob:.1f}%</div>
      <div style='font-size:16px;color:{wi_color};font-weight:700;margin-top:6px'>{wi_label}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TAB 5 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────
with tab5:
    st.markdown("<div class='stitle'>🤖 XGBoost Model Performance Report</div>",
                unsafe_allow_html=True)

    for col_w, name, val, desc, cls in zip(
        st.columns(5),
        ['Accuracy','Precision','Recall','F1-Score','ROC-AUC'],
        [metrics['Accuracy'],metrics['Precision'],metrics['Recall'],
         metrics['F1'],metrics['ROC_AUC']],
        ['Overall correctness','False-positive control',
         'Attrition detection ★','Precision/Recall balance',
         'Classification power'],
        ['bl','gr','rd','am','gr']
    ):
        with col_w:
            st.markdown(f"""
            <div class='kpi {cls}'>
              <div class='kpi-lbl'>{name}</div>
              <div class='kpi-val'>{val:.3f}</div>
              <div class='kpi-sub'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        fig, ax = plt.subplots(figsize=(5,4), facecolor=BG_CARD)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Stayed','Left'], yticklabels=['Stayed','Left'],
                    linewidths=2, linecolor=BG_CARD,
                    annot_kws={'size':16,'color':TXT_H,'weight':'bold'})
        cstyle(ax, 'Confusion Matrix','Predicted','Actual')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with cb:
        fig, ax = plt.subplots(figsize=(5,4), facecolor=BG_CARD)
        bars = ax.bar(list(metrics.keys()), list(metrics.values()),
                      color=[BLUE,GREEN,RED,AMBER,PURPLE],
                      edgecolor=BG_CARD, width=0.6, alpha=0.92)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.007,
                    f'{bar.get_height():.3f}', ha='center', va='bottom',
                    fontsize=10, color=TXT_H, fontweight='700')
        ax.set_ylim(0,1.12)
        ax.axhline(y=0.8, color=TXT_D, linestyle='--', alpha=0.5, linewidth=1.2)
        cstyle(ax, 'Model Metrics Overview')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div class='stitle'>⚙️ ML Pipeline Summary</div>",
                unsafe_allow_html=True)
    for step, desc in [
        ("1. Data Loading",        "1,470 employees × 31 features from Palo Alto Networks HR data"),
        ("2. Preprocessing",       "Label Encoding (OverTime, Gender) + One-Hot Encoding (5 categorical columns)"),
        ("3. Feature Engineering", "6 new features: IncomePerYear, EngagementScore, WorkloadStress, PromotionDelay, RoleStagnation, LoyaltyIndex"),
        ("4. Train/Test Split",    "80/20 stratified split → 1,176 train / 294 test"),
        ("5. Class Balancing",     "SMOTE applied to training set to fix 84%/16% imbalance"),
        ("6. Model",               "XGBoost (300 trees, depth=6, lr=0.05) trained on SMOTE-balanced data"),
        ("7. Risk Scoring",        "Each employee scored 0–100% → Low / Medium / High risk categories"),
    ]:
        st.markdown(f"""
        <div class='piperow'>
          <span class='pipestep'>{step}</span>
          <span class='pipedesc'>{desc}</span>
        </div>""", unsafe_allow_html=True)
