import numpy as np
import pandas as pd

class QuantRiskEngine:
    def __init__(self):
        pass

    def calculate_value_at_risk(self, exposure: float, volatility: float) -> float:
        """Parametric Value at Risk (95% confidence ceiling)."""
        return round(exposure * (volatility * 1.645), 2)

    def calculate_jurisdictional_bank_fee(self, country: str, target_instrument: str, base_amount: float) -> float:
        """Calculates transactional hedging overhead spreads based on global geographical risk."""
        country_clean = country.strip().lower()
        
        if country_clean in ['united states', 'united kingdom', 'germany', 'france', 'singapore']:
            base_fee_rate = 0.0040  
        elif country_clean in ['japan', 'australia', 'canada']:
            base_fee_rate = 0.0065  
        else:
            base_fee_rate = 0.0150  

        if "option" in target_instrument.lower():
            base_fee_rate += 0.0075 
            
        return round(base_amount * base_fee_rate, 2)

    def optimize_hedge_ratio(self, pipeline_payload: dict) -> dict:
        if not pipeline_payload:
            return {"error": "Empty data context passed."}

        raw_matrix = pipeline_payload.get("raw_exposure_matrix", [])
        base_currency = pipeline_payload.get("base_currency_flag", "USD")
        calculated_exposures = []
        
        total_portfolio_var_unhedged = 0.0
        total_estimated_bank_fees = 0.0

        for row in raw_matrix:
            currency = row["currency"]
            vol = row["volatility_index"]
            country = row["country"]
            
            # Run calculations strictly on the unhedged values relative to our base currency
            base_unhedged = row["base_unhedged_value"]
            unhedged_var = self.calculate_value_at_risk(base_unhedged, vol)
            total_portfolio_var_unhedged += unhedged_var

            if vol > 0.13:
                target_ratio = 0.85
                instrument = "Forward Contract (Aggressive Outright Lock)"
            elif vol > 0.07:
                target_ratio = 0.60
                instrument = "Option Collar Strategy (Flexible Floor/Ceiling)"
            else:
                target_ratio = 0.30
                instrument = "Spot/Trailing Hedge execution"

            hedging_target_volume_base = base_unhedged * target_ratio
            bank_fee = self.calculate_jurisdictional_bank_fee(country, instrument, hedging_target_volume_base)
            total_estimated_bank_fees += bank_fee

            calculated_exposures.append({
                "invoice_id": row["invoice_id"],
                "counterparty": row["counterparty"],
                "country": country,
                "currency": currency,
                "total_base_value": row["base_value"],
                "already_hedged_base": row["base_hedged_value"],
                "naked_unhedged_base": base_unhedged,
                "volatility_index": vol,
                "unhedged_value_at_risk_base": unhedged_var,
                "recommended_new_hedge_ratio": target_ratio,
                "target_instrument": instrument,
                "estimated_bank_fee_base": bank_fee
            })

        return {
            "analysis_timestamp": pipeline_payload.get("timestamp"),
            "base_currency_flag": base_currency,
            "total_portfolio_exposure_base": pipeline_payload.get("total_risk_exposure_base"),
            "total_unhedged_exposure_base": pipeline_payload.get("total_unhedged_exposure_base"),
            "residual_unhedged_portfolio_var_base": round(total_portfolio_var_unhedged, 2),
            "global_portfolio_estimated_bank_fees_base": round(total_estimated_bank_fees, 2),
            "breakdown": calculated_exposures
        }