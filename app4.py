import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =====================================================
# 1. Page Configuration
# =====================================================
st.set_page_config(
    page_title="MICP Soil Remediation AI (Pro)",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. UI Theme (绿色环保风格)
# =====================================================
st.markdown("""
<style>
    :root {
        --primary-color: #00897B;
        --bg-color: #F4F6F7;
    }
    .stApp { background-color: var(--bg-color); }
    
    /* 卡片样式 */
    .css-card {
        border-radius: 12px; padding: 24px; background-color: white;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04); margin-bottom: 20px;
        border: 1px solid #EFF0F1;
    }
    
    /* 预测结果框 */
    .prediction-box {
        background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
        color: white; padding: 30px; border-radius: 15px;
        text-align: center; box-shadow: 0 8px 16px rgba(24, 90, 157, 0.15);
    }
    .prediction-value { font-size: 3.5rem; font-weight: 700; margin: 0; }
    
    /* 侧边栏与字体 */
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    h1, h2, h3 { color: #263238; }
    
    /* 下拉菜单加粗 */
    .stSelectbox label { font-weight: 600; color: #00897B; }
    
    /* System Info 样式 */
    .sys-info {
        background-color: #E0F2F1;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #00897B;
        color: #00695C;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. Database & Mapping (14种细菌 + 7种钙源)
# =====================================================

FEATURES = [
    'Bacterial addition（ml）',
    'calcium ion（mmol/L）',
    'urea（g）',
    'Lead concentration（mg/kg）',
    'pH',
    'reaction temperature（℃）',
    'curing time（day）',
    'bacterium_encoded',       
    'Calcium_source_encoded'   
]

# 🦠 细菌数据库
BACTERIA_MAP = {
    '0: ML-2': 0,
    '1: B-21-B-22-KJ-46-KJ-47': 1,
    '2: GXAS49-I': 2,
    '3: HJ2': 3,
    '4: Kocuria flava CR1': 4,
    '5: Lysinibacillus fusiformis': 5,
    '6: Pseudomonasstutzeri': 6,
    '7: Sporosarcina pasteurii': 7,
    '8: Stenotrophomonas rhizophila': 8,
    '9: Streptomyces mutabilis SHT17': 9,
    '10: Terrabacter tumescens strains': 10,
    '11: UPB1': 11,
    '12: Variovorax boronicumulan': 12,
    '13: klebsiellagrimontii': 13
}

# 🧪 钙源数据库
CALCIUM_SOURCE_MAP = {
    '0: Ca(NO3)2 (Calcium Nitrate)': 0,
    '1: CaCl2 (Calcium Chloride)': 1,
    '2: CaCl2·2H2O': 2,
    '3: CaCl2·nHAP': 3,
    '4: CaO (Calcium Oxide)': 4,
    '5: C4H6CaO4 (Calcium Acetate)': 5,
    '6: None': 6
}

TARGET_DISPLAY = 'Pb Fixation Efficiency (%)'

# =====================================================
# 4. Load Logic (模型 + Scaler)
# =====================================================
@st.cache_resource
def load_resources():
    try:
        model = joblib.load("random_forest_curing_rate_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler, True
    except FileNotFoundError:
        # Mock for Demo
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        mock_model = RandomForestRegressor(n_estimators=100)
        mock_scaler = StandardScaler()
        X_mock = np.random.rand(20, 9)
        mock_scaler.fit(X_mock)
        mock_model.fit(mock_scaler.transform(X_mock), np.random.rand(20)*100)
        return mock_model, mock_scaler, False

rf_model, scaler, resources_loaded = load_resources()

# =====================================================
# 5. UI Layout & Sidebar
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("## 🌍 MICP Platform")
    st.caption("Soil Remediation Intelligence")
    st.markdown("---")
    
    menu = st.radio("Navigation", 
                    ["🏠 Dashboard", "📂 Batch Analysis", "🧮 Single Simulation", "🕒 History", "📊 Model Insights"], 
                    label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### ⚙️ System Info")
    
    # ✅ 更新：System Info 模块
    status = "Online" if resources_loaded else "Demo Mode"
    st.markdown(f"""
    <div class="sys-info">
        <b>Model:</b> {status}<br>
        <b>Target:</b> Fixed efficiency<br>
        <b>Method:</b> MICP
    </div>
    """, unsafe_allow_html=True)

# --- DASHBOARD (✅ 文案已更新) ---
if menu == "🏠 Dashboard":
    st.title("🌱 Soil Lead Fixation Prediction Platform")
    st.subheader("Based on Microbially Induced Calcite Precipitation (MICP)")
    
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    
    # 主描述
    st.markdown("""
    This platform utilizes machine learning (**Random Forest And Bayes**) to predict the efficiency of **Heavy Metal (Lead) Immobilization** in soil through bio-mineralization processes.
    
    ### Core Capabilities:
    * 🧪 **Optimize Recipes:** Determine the proportions of different feature conditions.
    * 📉 **Risk Assessment:** Predict the immobilization efficiency of soil heavy metal lead by MICP under different conditions.
    * ⚡ **Rapid Screening:** Process large experimental datasets in seconds.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 数据库预览
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🦠 Bacteria Database")
        st.info(f"Integrated {len(BACTERIA_MAP)} Strains")
        with st.expander("View All Strains"):
            st.json(BACTERIA_MAP)
    with c2:
        st.markdown("### 🧪 Calcium Database")
        st.info(f"Integrated {len(CALCIUM_SOURCE_MAP)} Sources")
        with st.expander("View All Sources"):
            st.json(CALCIUM_SOURCE_MAP)

# --- BATCH PREDICTION ---
elif menu == "📂 Batch Analysis":
    st.subheader("📂 Batch Analysis (with Auto-Scaling)")
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.info("Ensure CSV includes 'bacterium_encoded' (0-13) and 'Calcium_source_encoded' (0-6) columns.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        if not all(col in data.columns for col in FEATURES):
            st.error(f"❌ Columns Mismatch. Required: {FEATURES}")
        else:
            with st.spinner('Scaling & Predicting...'):
                X_batch_scaled = scaler.transform(data[FEATURES])
                preds = rf_model.predict(X_batch_scaled)
                
                data[TARGET_DISPLAY] = preds
                st.success("✅ Prediction Complete")
                st.dataframe(data.style.background_gradient(subset=[TARGET_DISPLAY], cmap="Greens"))
                
                # Save to History
                st.session_state.history.append({
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Type": "Batch",
                    "Description": f"{len(data)} Samples Processed",
                    "Result": "See CSV"
                })
                
                csv = data.to_csv(index=False).encode("utf-8-sig")
                st.download_button("⬇️ Download Results", csv, "batch_results.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# --- SINGLE PREDICTION ---
elif menu == "🧮 Single Simulation":
    st.subheader("🧮 Single Experiment Simulation")
    
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    with st.form("sim_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🦠 Bacteria Selection")
            bact_name = st.selectbox("Select Strain", options=list(BACTERIA_MAP.keys()), index=7)
            bact_val = BACTERIA_MAP[bact_name]
            bacterial_vol = st.number_input("Bacterial Addition (ml)", 0.0, 500.0, 5.0)

        with col2:
            st.markdown("#### 🧪 Calcium Selection")
            calc_name = st.selectbox("Select Source", options=list(CALCIUM_SOURCE_MAP.keys()), index=1)
            calc_val = CALCIUM_SOURCE_MAP[calc_name]
            calcium_conc = st.number_input("Ca2+ Concentration (mmol/L)", 0.0, 3000.0, 100.0)

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1: urea = st.number_input("Urea (g)", 0.0, 1000.0, 10.0)
        with c2: lead = st.number_input("Initial Pb (mg/kg)", 0.0, 20000.0, 500.0)
        with c3: ph = st.slider("pH Level", 0.0, 14.0, 7.0, 0.1)
        with c4: 
            temp = st.number_input("Temp (℃)", 0.0, 100.0, 25.0)
            time = st.number_input("Time (day)", 0.0, 365.0, 14.0)
            
        submit = st.form_submit_button("🚀 Simulate Efficiency", type="primary", use_container_width=True)

    if submit:
        input_data = pd.DataFrame([{
            'Bacterial addition（ml）': bacterial_vol,
            'calcium ion（mmol/L）': calcium_conc,
            'urea（g）': urea,
            'Lead concentration（mg/kg）': lead,
            'pH': ph,
            'reaction temperature（℃）': temp,
            'curing time（day）': time,
            'bacterium_encoded': bact_val,
            'Calcium_source_encoded': calc_val
        }])

        input_scaled = scaler.transform(input_data)
        pred = rf_model.predict(input_scaled)[0]

        # Save to History
        st.session_state.history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Type": "Single",
            "Description": f"Strain: {bact_name.split(':')[0]} | Pb: {lead}",
            "Result": f"{pred:.2f}%"
        })

        c_res1, c_res2 = st.columns([1, 1.5])
        with c_res1:
             st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-label">Fixation Efficiency</div>
                <div class="prediction-value">{pred:.2f}%</div>
                <div style="font-size:0.8rem; margin-top:8px; opacity:0.9">
                Strain: {bact_name.split(':')[0]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c_res2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = pred,
                title = {'text': "Remediation Meter"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00897B"}, 'steps': [{'range': [0, 100], 'color': "#E0F2F1"}]}
            ))
            fig.update_layout(height=260, margin=dict(t=40,b=20,l=40,r=40))
            st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- HISTORY ---
elif menu == "🕒 History":
    st.subheader("🕒 Prediction Logs")
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    
    if len(st.session_state.history) == 0:
        st.info("No records found in this session.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)
        
        csv_hist = df_hist.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Export Log CSV", csv_hist, "history_log.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# --- MODEL INSIGHTS ---
elif menu == "📊 Model Insights":
    st.subheader("📊 Feature Importance")
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    
    if hasattr(rf_model, 'feature_importances_'):
        imp = pd.DataFrame({'Feature': FEATURES, 'Importance': rf_model.feature_importances_})
        imp = imp.sort_values('Importance', ascending=True)
        imp['Feature'] = imp['Feature'].str.replace('_encoded', '').str.replace('（', ' (').str.replace('）', ')')
        
        fig = px.bar(imp, x='Importance', y='Feature', orientation='h', 
                     text_auto='.3f', color='Importance', color_continuous_scale='Teal')
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)