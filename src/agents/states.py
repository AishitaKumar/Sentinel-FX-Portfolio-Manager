from typing import TypedDict, List, Dict, Any

class AgentPipelineState(TypedDict):
    # Pipeline Tracking Metadata
    timestamp: str
    status: str                         # "INGESTED", "QUANT_PROCESSED", "STRATEGY_GENERATED", "PENDING_APPROVAL"
    
    # Financial Payload (Data passed from Ingestion & Quant layers)
    raw_data_payload: Dict[str, Any]
    quant_metrics: Dict[str, Any]
    
    # Macro Context Layer (Populated by Vector Search)
    matched_macro_memory: List[Dict[str, Any]]
    
    # Agentic Commentary & Reasoning
    ai_risk_report: str
    final_action_plan: Dict[str, Any]