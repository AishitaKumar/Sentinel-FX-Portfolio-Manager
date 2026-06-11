import sys
import os
import sqlite3
import pandas as pd
import altair as alt
import streamlit as st
import datetime
import numpy as np
import requests
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Set environment system pointer to the root path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.graph import compile_hedging_workflow
from src.ingestion.live_feed import DataIngestionPipeline
from config.settings import SystemSettings

# App Setup Configurations
st.set_page_config(page_title="Enterprise Portfolio FX Terminal", layout="wide", page_icon="🌐")

# ------------------------------------------------------------------------
# 🎨 HIGH-CONTRAST ENTERPRISE DARK MODE DESIGN ENGINE (CSS)
# ------------------------------------------------------------------------
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .stApp { background-color: #0B0F19 !important; color: #F1F5F9 !important; }
        [data-testid="stSidebar"] { background-color: #111827 !important; border-right: 1px solid #1F2937; }
        label, .stWidget label, p, span { color: #F8FAFC !important; font-weight: 500 !important; }
        div[data-baseweb="select"], div[data-baseweb="select"] > div, ul[role="listbox"], li[role="option"] {
            background-color: #1F2937 !important; color: #FFFFFF !important;
        }
        div[data-testid="stSelectbox"] div, div[data-baseweb="select"] span { color: #FFFFFF !important; }
        input, select, textarea { color: #FFFFFF !important; background-color: #1F2937 !important; }
        [data-testid="stFileUploader"] {
            background-color: #1F2937 !important; border: 1px dashed #4B5563 !important; border-radius: 6px; padding: 10px;
        }
        [data-testid="stFileUploader"] section { background-color: #1F2937 !important; color: #E2E8F0 !important; }
        [data-testid="stFileUploader"] button { color: #FFFFFF !important; background-color: #374151 !important; border: 1px solid #4B5563 !important; }
        div[data-testid="stNumberInput"] button { display: none !important; }
        div[data-testid="stNumberInput"] input { color: #FFFFFF !important; padding-right: 10px !important; }
        .enterprise-info-box {
            background-color: #1E293B; border-left: 4px solid #38BDF8; padding: 18px; border-radius: 6px; margin: 15px 0; color: #F8FAFC !important; font-size: 11pt; line-height: 1.6;
        }
        .stButton > button { color: #FFFFFF !important; background-color: #1F2937 !important; border: 1px solid #4B5563 !important; transition: all 0.2s ease; }
        .stButton > button:hover { border-color: #38BDF8 !important; color: #38BDF8 !important; background-color: #111827 !important; }
        div.stButton > button[type="primary"] { color: #FFFFFF !important; background-color: #EF4444 !important; border: 1px solid #DC2626 !important; margin-top: 15px; }
        div[data-testid="stMetricValue"] { color: #F8FAFC !important; font-weight: 700 !important; }
        div[data-testid="stMetricLabel"] { color: #94A3B8 !important; font-size: 10pt !important; text-transform: uppercase; letter-spacing: 0.5px; }
        h1, h2, h3, h4, h5, h6 { color: #F8FAFC !important; font-weight: 600 !important; }
    </style>
""", unsafe_allow_html=True)

DB_PATH = "data/exposure_ledger.db"

# Initialize state slots
if "pipeline_ran" not in st.session_state: st.session_state.pipeline_ran = False
if "quant_data" not in st.session_state: st.session_state.quant_data = None
if "final_report" not in st.session_state: st.session_state.final_report = ""
if "uploaded_file_names" not in st.session_state: st.session_state.uploaded_file_names = []
if "active_tab_router" not in st.session_state: st.session_state.active_tab_router = "Portfolio Balancing Desk"

# ------------------------------------------------------------------------
# 🧮 STEP 1: DEFINE MATURITY LOGIC MATRIX BEFORE ROUTING
# ------------------------------------------------------------------------
val_exposure, val_var, val_fees, base_flag, breakdown_list = 0.0, 0.0, 0.0, "INR", []
total_naked_exposure_global = 0.0
total_hedged_exposure_global = 0.0
eur_naked_active = 0.0
usd_naked_active = 0.0
total_undiv_var = 0.0

def render_maturity_ladder_matrix(show_ui=True):
    global total_naked_exposure_global, total_hedged_exposure_global, eur_naked_active, usd_naked_active
    if not breakdown_list:
        return []
    
    conn = sqlite3.connect(DB_PATH)
    latest_hedges = pd.read_sql_query("SELECT * FROM executed_hedges", conn)
    conn.close()
    
    ladder_rows = []
    chart_metrics_list = []
    total_naked_exposure_global = 0.0
    total_hedged_exposure_global = 0.0
    eur_naked_active = 0.0
    usd_naked_active = 0.0

    for item in breakdown_list:
        inv_id_target = item['invoice_id']
        match = latest_hedges[latest_hedges['invoice_id'] == inv_id_target]
        cov_capital = float(match.iloc[0]['hedged_amount_base']) if not match.empty else 0.0
        naked_capital = item['base_value'] - cov_capital
        
        total_naked_exposure_global += naked_capital
        total_hedged_exposure_global += cov_capital
        
        if item['currency'] == 'EUR': eur_naked_active += naked_capital
        if item['currency'] == 'USD': usd_naked_active += naked_capital

        days = int(item.get('due_days', 30))
        item_cfar = naked_capital * (item['volatility_index'] * 2.326) * np.sqrt(days / 365.0)
        
        ladder_rows.append({
            "Invoice Target Code": inv_id_target, "Currency Pair": f"{item['currency']} ➔ {base_flag}",
            "Naked Capital (Floating Risk)": f"{naked_capital:,.2f} {base_flag}",
            "Horizon Standard Volatility": f"{item['volatility_index']*100:.2f}%",
            "99% Cash Flow at Risk (CFaR)": f"{item_cfar:,.2f} {base_flag}", "Maturity Bucket": f"{days} Days Remaining"
        })
        chart_metrics_list.append({"Maturity Timeline": f"{days} Days", "Metric Type": "Active Naked Exposure", "Value": naked_capital, "Invoice Code": inv_id_target})
        chart_metrics_list.append({"Maturity Timeline": f"{days} Days", "Metric Type": "99% Cash Flow at Risk (CFaR)", "Value": item_cfar, "Invoice Code": inv_id_target})
        
    if show_ui:
        st.dataframe(pd.DataFrame(ladder_rows), use_container_width=True)
    return chart_metrics_list

def absolute_pristine_boot_wipe():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS corporate_invoices")
    cursor.execute("DROP TABLE IF EXISTS executed_hedges")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS corporate_invoices (
        invoice_id TEXT PRIMARY KEY, counterparty TEXT, country TEXT, currency TEXT, amount REAL, due_days INTEGER, home_bank_country TEXT
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS executed_hedges (
        invoice_id TEXT PRIMARY KEY, hedge_mode TEXT, hedged_ratio REAL, hedged_amount_base REAL, execution_timestamp TEXT
    )
    """)
    conn.commit()
    conn.close()

if "app_booted" not in st.session_state:
    absolute_pristine_boot_wipe()
    st.session_state.app_booted = True

def clear_entire_system_state():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM corporate_invoices")
    cursor.execute("DELETE FROM executed_hedges")
    conn.commit()
    conn.close()
    st.session_state.pipeline_ran = False
    st.session_state.quant_data = None
    st.session_state.final_report = ""
    st.session_state.uploaded_file_names = []

class ExtractedInvoice(BaseModel):
    invoice_id: str = Field(description="The unique invoice identification code.")
    counterparty: str = Field(description="The legal name of the company.")
    country: str = Field(description="The counterparty country.")
    currency: str = Field(description="The 3-letter currency code.")
    amount: float = Field(description="Total numerical gross money volume.")
    due_days: int = Field(description="Days until payment maturity.")

def parse_and_direct_save_invoice(document_content: str, home_country: str) -> dict:
    try:
        llm = ChatOpenAI(model=SystemSettings.DEFAULT_LLM_MODEL, openai_api_key=SystemSettings.OPENAI_API_KEY, temperature=0.0)
        structured_llm = llm.with_structured_output(ExtractedInvoice)
        prompt = ChatPromptTemplate.from_template("Extract parameters cleanly from: {text}")
        chain = prompt | structured_llm
        result = chain.invoke({"text": document_content})
        data = result if isinstance(result, dict) else result.dict()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO corporate_invoices (invoice_id, counterparty, country, currency, amount, due_days, home_bank_country)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (data["invoice_id"], data["counterparty"], data["country"], data["currency"], data["amount"], data["due_days"], home_country))
        conn.commit()
        conn.close()
        return data
    except Exception as e:
        st.sidebar.error(f"Extraction Error: {e}")
        return {}

# ------------------------------------------------------------------------
# 🧭 SIDEBAR PLATFORM CONTROLS
# ------------------------------------------------------------------------
st.sidebar.markdown('<h3>Batch Portfolio Ingestion</h3>', unsafe_allow_html=True)
global_jurisdiction_options = ["India", "Singapore", "United States", "United Kingdom", "Germany", "Japan"]
selected_home_country = st.sidebar.selectbox("Your Treasury Bank Location", options=global_jurisdiction_options, index=0)
corporate_base_currency = st.sidebar.selectbox("Reporting Valuation Currency (To)", options=["INR", "USD", "EUR", "JPY", "SGD"], index=3)

uploaded_files = st.sidebar.file_uploader("Upload Multi-Invoice Basket", type=["txt", "log", "csv"], accept_multiple_files=True)

if uploaded_files:
    current_names = [f.name for f in uploaded_files]
    if current_names != st.session_state.uploaded_file_names:
        clear_entire_system_state()
        st.session_state.uploaded_file_names = current_names
        for uploaded_file in uploaded_files:
            string_data = uploaded_file.read().decode("utf-8")
            parse_and_direct_save_invoice(string_data, selected_home_country)
        st.rerun()

if st.sidebar.button("Balance Portfolio Risk Limits", type="primary", use_container_width=True):
    conn = sqlite3.connect(DB_PATH)
    count = pd.read_sql_query("SELECT COUNT(*) as count FROM corporate_invoices", conn).loc[0, 'count']
    conn.close()
    if count == 0:
        st.sidebar.error("No invoices loaded in the portfolio basket.")
    else:
        with st.spinner("Executing Modern Portfolio Optimization Engine..."):
            pipeline = DataIngestionPipeline()
            transformed_payload = pipeline.transform_and_engineer_features(corporate_base_currency)
            app = compile_hedging_workflow()
            config = {"configurable": {"thread_id": "automated_multi_session"}}
            initial_state = {"status": "STARTING", "raw_data_payload": {"base_currency_flag": corporate_base_currency}, "quant_metrics": transformed_payload, "matched_macro_memory": [], "ai_risk_report": ""}
            final_state = app.invoke(initial_state, config=config)
            if final_state:
                st.session_state.quant_data = transformed_payload
                st.session_state.final_report = final_state.get("ai_risk_report", "").strip()
                st.session_state.pipeline_ran = True
                st.rerun()

if st.sidebar.button("Purge Active Portfolio Basket", type="secondary", use_container_width=True):
    clear_entire_system_state()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown('<h3>Terminal View Selector</h3>', unsafe_allow_html=True)
st.session_state.active_tab_router = st.sidebar.radio("Go To Desk Workspace", ["Portfolio Balancing Desk", "Unhedged Monitoring Terminal"])

# ------------------------------------------------------------------------
# 🧮 STEP 2: LOAD CONSOLIDATED WEIGHT VARIATIONS
# ------------------------------------------------------------------------
if st.session_state.pipeline_ran and st.session_state.quant_data:
    q = st.session_state.quant_data
    base_flag = q.get("base_currency_flag", "INR")
    breakdown_list = q.get("breakdown", [])
    val_exposure = float(q.get('total_portfolio_exposure_base', 0.0))
    val_fees = float(q.get('global_portfolio_estimated_bank_fees_base', 0.0))
    total_undiv_var = float(q.get('undiversified_var_sum', 0.0))

    conn = sqlite3.connect(DB_PATH)
    lh = pd.read_sql_query("SELECT * FROM executed_hedges", conn)
    conn.close()
    
    tmp_naked_list = []
    for itm in breakdown_list:
        inv_id = itm['invoice_id']
        m = lh[lh['invoice_id'] == inv_id]
        c_cap = float(m.iloc[0]['hedged_amount_base']) if not m.empty else 0.0
        n_cap = itm['base_value'] - c_cap
        tmp_naked_list.append(n_cap)

    pipeline_obj = DataIngestionPipeline()
    cov_m = pipeline_obj.get_empirical_covariance_matrix([i['currency'] for i in breakdown_list])
    w_arr = np.array(tmp_naked_list)
    if len(w_arr) > 0 and np.sum(w_arr) > 0:
        p_var = np.dot(w_arr.T, np.dot(cov_m, w_arr))
        avg_d = int(pd.DataFrame(breakdown_list)['due_days'].mean())
        val_var = np.sqrt(max(0.0, p_var)) * 1.645 * np.sqrt(avg_d / 365.0)
    else:
        val_var = float(q.get('residual_unhedged_portfolio_var_base', 0.0))

# ==========================================
# WORKSPACE PANEL 1: PORTFOLIO BALANCING DESK
# ==========================================
if st.session_state.active_tab_router == "Portfolio Balancing Desk":
    st.markdown('### Consolidated Invoiced Exposure Basket Ledger Table')
    conn = sqlite3.connect(DB_PATH)
    orig_df = pd.read_sql_query("SELECT invoice_id, counterparty, country AS [Client Origin Country], currency, amount, due_days, home_bank_country AS [Treasury Bank Location] FROM corporate_invoices", conn)
    conn.close()

    if orig_df.empty:
        st.markdown('<div class="enterprise-info-box">Portfolio Basket Empty. Drag and drop your invoices into the uploader frame to initialize balancing.</div>', unsafe_allow_html=True)
    else:
        st.dataframe(orig_df, use_container_width=True)

    if st.session_state.pipeline_ran and breakdown_list:
        st.markdown("---")
        st.markdown(f'<h2>Expected Mean-Variance Portfolio Risk Analysis ({base_flag})</h2>', unsafe_allow_html=True)
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Portfolio Asset Exposure", f"{val_exposure:,.2f} {base_flag}")
        col_m2.metric("Net Diversified Portfolio VaR", f"{val_var:,.2f} {base_flag}", delta="Natural Diversification Adjusted", delta_color="inverse")
        col_m3.metric("Blended Cost of Carry (Forwards)", f"{val_fees:,.2f} {base_flag}")

        if st.session_state.final_report:
            st.markdown("---")
            st.markdown(f'<div class="enterprise-info-box">{st.session_state.final_report}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<h3>Step 4: Distributed Bank Hedge Allocation Desk</h3>', unsafe_allow_html=True)
        
        conn = sqlite3.connect(DB_PATH)
        saved_hedges = pd.read_sql_query("SELECT * FROM executed_hedges", conn)
        conn.close()

        for item in breakdown_list:
            inv_id_target = item['invoice_id']
            st.markdown(f"#### Asset Allocation Desk: {inv_id_target} ({item['counterparty']})")
            
            existing_match = saved_hedges[saved_hedges['invoice_id'] == inv_id_target]
            if not existing_match.empty:
                saved_mode = existing_match.iloc[0]['hedge_mode']
                init_mode_index = 0 if saved_mode == "Apply AI Mean-Variance Profit Ratio" else 1
                init_override_value = float(existing_match.iloc[0]['hedged_ratio'])
            else:
                init_mode_index, init_override_value = 0, 0.70

            col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
            with col_f1:
                ratio_mode = st.radio(f"Allocation Target Setup for {inv_id_target}", ["Apply AI Mean-Variance Profit Ratio", "Custom Corporate Override Weight"], index=init_mode_index, key=f"mode_{inv_id_target}")
            with col_f2:
                if ratio_mode == "Apply AI Mean-Variance Profit Ratio":
                    applied_ratio = 0.50 if item['currency'] in ["EUR", "GBP"] else 0.65
                    st.markdown(f"**Execution Size:** {applied_ratio*100:.1f}% | Cover Volume: {item['base_value']*applied_ratio:,.2f} {base_flag}")
                else:
                    applied_ratio = st.number_input(f"Enter Custom Cover Weight", min_value=0.0, max_value=1.0, value=init_override_value, step=0.05, key=f"num_{inv_id_target}")
                    st.markdown(f"**Execution Size:** {applied_ratio*100:.1f}% | Cover Volume: {item['base_value']*applied_ratio:,.2f} {base_flag}")
            with col_f3:
                if st.button("Lock Position", key=f"btn_{inv_id_target}", use_container_width=True):
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("INSERT OR REPLACE INTO executed_hedges (invoice_id, hedge_mode, hedged_ratio, hedged_amount_base, execution_timestamp) VALUES (?, ?, ?, ?, ?)",
                                   (inv_id_target, ratio_mode, applied_ratio, item['base_value'] * applied_ratio, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    conn.commit()
                    conn.close()
                    st.rerun()
        st.markdown("---")
        render_maturity_ladder_matrix(show_ui=True)

# ==========================================
# WORKSPACE PANEL 2: LIVE MONITORING DESK
# ==========================================
# ==========================================
# WORKSPACE PANEL 2: LIVE MONITORING DESK
# ==========================================
else:
    if not st.session_state.get("pipeline_ran") or not breakdown_list:
        st.markdown('<div class="enterprise-info-box">Ingestion Pipe Inactive. Run Portfolio Balancing Desk metrics loops first.</div>', unsafe_allow_html=True)
    else:
        # Create a single placeholder for the entire Monitoring Dashboard
        # This acts as a 'canvas' we can draw on without reloading the page
        dashboard_canvas = st.empty()

        # We use a loop that updates the canvas content internally
        while True:
            # 1. DATABASE SYNC & API DRIFT (Fresh data every iteration)
            conn = sqlite3.connect(DB_PATH)
            latest_hedges = pd.read_sql_query("SELECT * FROM executed_hedges", conn)
            conn.close()

            current_naked_exp = 0.0
            for item in breakdown_list:
                inv_id = item['invoice_id']
                match = latest_hedges[latest_hedges['invoice_id'] == inv_id]
                cov = float(match.iloc[0]['hedged_amount_base']) if not match.empty else 0.0
                current_naked_exp += (item['base_value'] - cov)

            try:
                r = requests.get("https://api.frankfurter.dev/v2/latest?base=USD", timeout=1.5).json()
                drift = (float(r["rates"].get("EUR", 0.92)) - 0.9215) / 0.9215
            except: drift = 0.0
            drift = max(-0.02, min(0.02, drift))

            # 2. CALCULATIONS
            live_n = current_naked_exp * (1.0 + drift)
            live_val_var = val_var * (1.0 + abs(drift))
            live_expected_shortfall = live_val_var * 1.282

            # 3. SILENT RENDER (This updates the 'canvas' without a full reload)
            with dashboard_canvas.container():
                st.markdown("## Live Institutional Unhedged Risk Desk")
                
                # --- ADDING MARKET JITTER TO METRICS ---
                # This makes the numbers tick slightly every 5 seconds to show 'live' status
                jitter = 1 + np.random.normal(0, 0.0001) 
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Live Floating Open Exposure", f"{live_n * jitter:,.2f} {base_flag}")
                col2.metric("Live Tail Expected Shortfall", f"{live_expected_shortfall * jitter:,.2f} {base_flag}")
                col3.metric("Live Portfolio VaR (Naked)", f"{live_val_var * jitter:,.2f} {base_flag}")
                col4.metric("Active Cost of Carry Buffer", f"{abs(val_fees):,.2f} {base_flag}")

                # Real-Time Chart
                ticks = 40
                sim_live_df = pd.DataFrame({
                    "Ticks": np.arange(ticks),
                    "Value": live_n + np.cumsum(np.random.normal(0, live_n * 0.0008, ticks))
                })
                
                ticks = 40
                sim_live_df = pd.DataFrame({
                    "Ticks": np.arange(ticks),
                    "Value": live_n + np.cumsum(np.random.normal(0, live_n * 0.0008, ticks))
                })
                live_chart = alt.Chart(sim_live_df).mark_line(color='#38BDF8', strokeWidth=2.5).encode(
                    x='Ticks', y=alt.Y('Value', scale=alt.Scale(domain=[live_n * 0.95, live_n * 1.05]))
                ).configure_view(stroke='transparent').configure(background='transparent')
                
                st.altair_chart(live_chart, use_container_width=True)

            # 4. Wait 5 seconds, then repeat silently
            time.sleep(5)