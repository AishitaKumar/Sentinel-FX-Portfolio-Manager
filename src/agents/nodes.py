import sys
import os
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ingestion.live_feed import DataIngestionPipeline
from src.quant_engine.risk_models import QuantRiskEngine
from src.vector_memory.embedding_store import SemanticMarketMemory
from config.settings import SystemSettings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.agents.states import AgentPipelineState

# 1. Ingestion Node (Reads base currency from state initialized by frontend)
def ingestion_node(state: AgentPipelineState) -> Dict[str, Any]:
    print("📥 [Node: Ingestion] Fetching live market vectors...")
    # Read chosen base currency from state, default to USD if empty
    base_curr = state.get("raw_data_payload", {}).get("base_currency_flag", "USD") if state.get("raw_data_payload") else "USD"
    
    pipeline = DataIngestionPipeline()
    payload = pipeline.transform_and_engineer_features(base_currency=base_curr)
    return {"raw_data_payload": payload, "status": "INGESTED", "timestamp": payload.get("timestamp")}

# 2. Quant Math Node
def quant_node(state: AgentPipelineState) -> Dict[str, Any]:
    print("📐 [Node: Quant Risk] Crunching mathematical Value at Risk (VaR)...")
    raw_payload = state.get("raw_data_payload")
    engine = QuantRiskEngine()
    metrics = engine.optimize_hedge_ratio(raw_payload)
    return {"quant_metrics": metrics, "status": "QUANT_PROCESSED"}

# 3. AI Strategist Node
def ai_strategist_node(state: AgentPipelineState) -> Dict[str, Any]:
    print("🧠 [Node: AI Strategist] Synthesizing analytics...")
    quant_m = state.get("quant_metrics")
    base_currency = quant_m.get("base_currency_flag", "USD")
    current_market_noise = "Concerns grow over FX volatility as central bank monitoring tightens."
    
    memory_engine = SemanticMarketMemory()
    context_match = memory_engine.search_similar_market_regimes(current_market_noise, limit=1)
    
    llm = ChatOpenAI(model=SystemSettings.DEFAULT_LLM_MODEL, openai_api_key=SystemSettings.OPENAI_API_KEY, temperature=0.1)
    
    prompt = ChatPromptTemplate.from_template("""
    You are a Chief FX Risk Officer running an enterprise hedging execution agent. 
    Evaluate the quantitative calculations and synthesize them with historical macro precedent.

    --- CURRENT PORTFOLIO METRICS (All values are calculated in Corporate Base Currency: {base_currency}) ---
    Total Exposure Under Management: {total_portfolio_exposure_base} {base_currency}
    Total Portfolio Value at Risk (VaR) on Unhedged Capital: {total_portfolio_at_risk_base} {base_currency}
    Mathematically Suggested Blended Hedge Ratio: {optimized_blended_hedge_ratio}
    
    Detailed Breakdown:
    {breakdown}

    --- RELEVANT HISTORICAL MACRO MEMORY ---
    Closest Matching Historical Regime: {historical_context}

    --- EXECUTIVE INSTRUCTIONS ---
    Provide a professional macro risk assessment report. 
    Explicitly reference that the calculations are analyzed in corporate baseline reporting currency: {base_currency}.
    Explain why the mathematical engine recommends these hedges based on asset risk profiles and sign off with a summary.
    """)
    
    chain = prompt | llm
    ai_response = chain.invoke({
        "base_currency": base_currency,
        "total_portfolio_exposure_base": quant_m.get("total_portfolio_exposure_base"),
        "total_portfolio_at_risk_base": quant_m.get("residual_unhedged_portfolio_var_base"),
        "optimized_blended_hedge_ratio": quant_m.get("optimized_blended_hedge_ratio"),
        "breakdown": quant_m.get("breakdown"),
        "historical_context": context_match[0]["historical_insight"] if context_match else "None available"
    })
    
    return {"ai_risk_report": ai_response.content, "matched_macro_memory": context_match, "status": "STRATEGY_GENERATED"}