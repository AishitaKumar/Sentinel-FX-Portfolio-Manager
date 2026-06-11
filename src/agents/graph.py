import os
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config.settings import SystemSettings

class AgentWorkflowState(TypedDict):
    status: str
    raw_data_payload: Dict[str, Any]
    quant_metrics: Dict[str, Any]
    matched_macro_memory: List[str]
    ai_risk_report: str

def quant_risk_terminal_node(state: AgentWorkflowState) -> Dict[str, Any]:
    quant_metrics = state.get("quant_metrics", {})
    if not quant_metrics or "breakdown" not in quant_metrics:
        return {
            "status": "QUANT_PROCESSING_FAILED",
            "quant_metrics": {
                "base_currency_flag": state.get("raw_data_payload", {}).get("base_currency_flag", "INR"),
                "total_portfolio_exposure_base": 0.0,
                "residual_unhedged_portfolio_var_base": 0.0,
                "global_portfolio_estimated_bank_fees_base": 0.0,
                "undiversified_var_sum": 0.0,
                "diversification_benefit": 0.0,
                "breakdown": []
            }
        }
    return {"status": "QUANT_VERIFIED", "quant_metrics": quant_metrics}

        
def macro_brain_knowledge_node(state: AgentWorkflowState) -> Dict[str, Any]:
    quant_metrics = state.get("quant_metrics", {})
    breakdown_list = quant_metrics.get("breakdown", [])
    active_currencies = {item.get("currency") for item in breakdown_list if "currency" in item}
    macro_memories = []
    
    if "EUR" in active_currencies:
        macro_memories.append("The European Central Bank (ECB) is closely monitoring deposit facilities to counteract regional yield compression risks.")
    if "USD" in active_currencies:
        macro_memories.append("The Federal Reserve is holding steady near terminal interest rate bands, keeping near-term commercial USD liquidity tight.")
    if "GBP" in active_currencies:
        macro_memories.append("The Bank of England is aggressively tracking sticky core inflation indices, reinforcing positive regional currency pairs co-movements.")
        
    return {"status": "MACRO_INJECTED", "matched_macro_memory": macro_memories}

def chief_risk_officer_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """
    TOTAL INDIVIDUAL PURGE FIX: Completely forbids loops, lists, tables, or itemized accounts.
    Generates a crisp, straightforward corporate strategic assessment in plain prose paragraphs.
    """
    quant_metrics = state.get("quant_metrics", {})
    macro_memory = state.get("matched_macro_memory", [])

    # --- GUARDRAIL VALIDATION LAYER ---
    breakdown_list = quant_metrics.get("breakdown", [])
    for item in breakdown_list:
        ratio = item.get("hedge_ratio", 0.5)
        # Institutional safety corridor: 20% to 85%
        if ratio < 0.20 or ratio > 0.85:
            return {
                "status": "VALIDATION_FAILED", 
                "ai_risk_report": f"CRITICAL: Hedge ratio {ratio} for {item['invoice_id']} outside safety corridor."
            }
    # --- END GUARDRAIL ---
    
    base_currency = quant_metrics.get("base_currency_flag", "INR")
    total_exposure = quant_metrics.get("total_portfolio_exposure_base", 0.0)
    net_var = quant_metrics.get("residual_unhedged_portfolio_var_base", 0.0)
    div_benefit = quant_metrics.get("diversification_benefit", 0.0)
    
    macro_context_text = "\n".join([f"- {memory}" for memory in macro_memory])

    llm = ChatOpenAI(
        model=SystemSettings.DEFAULT_LLM_MODEL, 
        openai_api_key=SystemSettings.OPENAI_API_KEY, 
        temperature=0.0
    )
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert institutional corporate treasury director. Write a direct, sensible, prose-driven strategic risk analysis summary for the executive team.
    
    CRITICAL RESTRICTION: Do NOT include greetings, introductions, or formal memo elements. Do NOT list individual invoices, numbers, or targets individually (no Invoice Code sections, no itemized bullet points). Write purely in clean, professional paragraphs explaining the overall portfolio status.
    
    AGGREGATED TREASURY METRICS ({base_currency}):
    - Aggregated Capital Exposure: {total_exposure:,.2f} {base_currency}
    - Portfolio Value at Risk (VaR): {net_var:,.2f} {base_currency}
    - Capital Insulation Diversification Savings: {div_benefit:,.2f} {base_currency}
    
    CENTRAL BANK CONTEXTS:
    {macro_context_text}
    
    EXECUTIVE PROSE STRUCTURE RULES:
    1. Begin directly with a bold header: "### Treasury Risk Assessment & Market Outlook". In clean prose, explain that managing an open aggregated foreign exchange volume of {total_exposure:,.2f} {base_currency} exposes corporate cash flows to a portfolio Value at Risk (VaR) of {net_var:,.2f} {base_currency}. Detail how high cross-currency positive correlations (+0.82 between EUR/GBP) and safe-haven USD behaviors yield a natural diversification benefit of {div_benefit:,.2f} {base_currency}, insulating the balance sheet and preventing the treasury from buying duplicate insurance contracts.
    2. Continue with a second bold header: "### Time-Horizon and Macro Policy Adjustments". Write a clear paragraph explaining that risk velocities vary dynamically across settlement windows. Note that near-term 30-day USD exposure requires tight tracking due to tight intra-month commercial liquidity enforced by the Federal Reserve's terminal rate bands, while medium-term 90-day European exposures provide a wider runway for macro drift as regional central banks grapple with core inflation sticky parameters.
    3. Conclude with a third bold header: "### Strategic Hedging Framework". Write a solid concluding paragraph detailing why a dynamic allocation framework is mathematically superior to flat speculative approaches (such as automatically hedging 100% or letting everything float naked). Frame it around minimizing bank fee drag, protecting core cash flow operating margins from downside market shock events, and establishing clear, defensible corporate risk budgets.
    
    Generate the clean report prose now:
    """)
    
    chain = prompt | llm
    response = chain.invoke({
        "base_currency": base_currency,
        "total_exposure": total_exposure,
        "net_var": net_var,
        "div_benefit": div_benefit,
        "macro_context_text": macro_context_text
    })
    
    return {"status": "REPORT_GENERATED", "ai_risk_report": response.content}

def compile_hedging_workflow():
    workflow = StateGraph(AgentWorkflowState)
    workflow.add_node("QuantTerminalValidation", quant_risk_terminal_node)
    workflow.add_node("MacroKnowledgeIngestion", macro_brain_knowledge_node)
    workflow.add_node("ChiefRiskOfficerBriefing", chief_risk_officer_node)
    
    workflow.set_entry_point("QuantTerminalValidation")
    workflow.add_edge("QuantTerminalValidation", "MacroKnowledgeIngestion")
    workflow.add_edge("MacroKnowledgeIngestion", "ChiefRiskOfficerBriefing")
    workflow.add_edge("ChiefRiskOfficerBriefing", END)
    
    return workflow.compile()