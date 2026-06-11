import os
import sqlite3
import requests
import pandas as pd
import numpy as np
from datetime import datetime

class DataIngestionPipeline:
    def __init__(self, db_path="data/exposure_ledger.db"):
        self.db_path = db_path
        self.api_url = "https://open.er-api.com/v6/latest/"

    def fetch_live_market_rates(self, base_currency: str) -> dict:
        try:
            response = requests.get(f"{self.api_url}{base_currency}")
            data = response.json()
            if data.get("result") == "success":
                return data.get("rates", {})
            return {}
        except Exception as e:
            print(f"Error harvesting live spot metrics: {e}")
            return {}

    def get_empirical_covariance_matrix(self, currencies: list) -> np.ndarray:
        n = len(currencies)
        if n == 0:
            return np.zeros((0, 0))
            
        base_correlations = {
            ("EUR", "USD"): 0.38, ("EUR", "GBP"): 0.82, ("EUR", "JPY"): -0.12, ("EUR", "SGD"): 0.45,
            ("USD", "GBP"): 0.41, ("USD", "JPY"): -0.28, ("USD", "SGD"): 0.76,
            ("GBP", "JPY"): -0.08, ("GBP", "SGD"): 0.48,
            ("JPY", "SGD"): -0.18
        }
        
        fallback_vols = {"EUR": 0.0650, "GBP": 0.0720, "USD": 0.0590, "JPY": 0.0940, "INR": 0.0780, "SGD": 0.0415}
        
        cov_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                c1 = currencies[i]
                c2 = currencies[j]
                vol1 = fallback_vols.get(c1, 0.0850)
                vol2 = fallback_vols.get(c2, 0.0850)
                
                if c1 == c2:
                    cov_matrix[i][j] = vol1 * vol1
                else:
                    pair = (c1, c2) if (c1, c2) in base_correlations else (c2, c1)
                    rho = base_correlations.get(pair, 0.25)
                    cov_matrix[i][j] = rho * vol1 * vol2
                    
        return cov_matrix

    def calculate_irp_forward_rate(self, target_curr: str, base_curr: str, days_due: int) -> float:
        sovereign_yields = {"INR": 0.0675, "USD": 0.0525, "EUR": 0.0375, "JPY": 0.0025, "GBP": 0.0500, "SGD": 0.0340}
        try:
            response = requests.get(f"{self.api_url}{target_curr}")
            data = response.json()
            rates_map = data.get("rates", {})
            spot_rate = rates_map.get(base_curr, 1.0)
            
            r_home = sovereign_yields.get(base_curr, 0.0500)
            r_foreign = sovereign_yields.get(target_curr, 0.0400)
            time_fraction = days_due / 365.0
            
            forward_rate = spot_rate * ((1.0 + (r_home * time_fraction)) / (1.0 + (r_foreign * time_fraction)))
            return float(forward_rate)
        except Exception:
            return 1.0

    def extract_corporate_exposure(self) -> pd.DataFrame:
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database missing at {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        query = "SELECT invoice_id, counterparty, country, currency, amount, due_days FROM corporate_invoices"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def transform_and_engineer_features(self, base_currency: str = "INR") -> dict:
        rates = self.fetch_live_market_rates(base_currency)
        exposure_df = self.extract_corporate_exposure()

        if not rates or exposure_df.empty:
            return {}

        conn = sqlite3.connect(self.db_path)
        hedges_df = pd.read_sql_query("SELECT * FROM executed_hedges", conn)
        conn.close()

        processed_records = []
        currencies_in_portfolio = []
        naked_weights_base = []
        undiversified_var_sum = 0.0
        total_portfolio_exposure = 0.0
        total_portfolio_unhedged = 0.0
        total_carry_costs = 0.0
        
        fallback_vols = {"EUR": 0.0650, "GBP": 0.0720, "USD": 0.0590, "JPY": 0.0940, "INR": 0.0780, "SGD": 0.0415}

        for _, row in exposure_df.iterrows():
            currency = row['currency']
            amount = row['amount']
            inv_id = row['invoice_id']
            days_due = int(row['due_days'])
            
            match = hedges_df[hedges_df['invoice_id'] == inv_id]
            hedged_amount_base = float(match.iloc[0]['hedged_amount_base']) if not match.empty else 0.0
            hedged_ratio = float(match.iloc[0]['hedged_ratio']) if not match.empty else 0.0
            
            forward_rate = self.calculate_irp_forward_rate(currency, base_currency, days_due)
            vol = fallback_vols.get(currency, 0.0850)
            
            if currency in rates and rates[currency] > 0:
                base_val = amount / rates[currency]
                base_hedged_val = hedged_amount_base
                base_unhedged_val = base_val - base_hedged_val
                
                spot_rate = 1.0 / rates[currency]
                carry_cost_impact = base_unhedged_val * ((forward_rate - spot_rate) / spot_rate)
            else:
                base_val = amount if currency == base_currency else 0.0
                base_hedged_val = hedged_amount_base
                base_unhedged_val = base_val - base_hedged_val
                carry_cost_impact = 0.0

            time_factor = np.sqrt(days_due / 365.0)
            standalone_var = base_unhedged_val * (vol * 1.645) * time_factor
            undiversified_var_sum += standalone_var

            total_portfolio_exposure += base_val
            total_portfolio_unhedged += base_unhedged_val
            total_carry_costs += carry_cost_impact
            
            currencies_in_portfolio.append(currency)
            naked_weights_base.append(base_unhedged_val)

            processed_records.append({
                "invoice_id": inv_id,
                "counterparty": row["counterparty"],
                "country": row["country"],
                "currency": currency,
                "amount": amount,
                "hedged_ratio": hedged_ratio,
                "base_hedged_value": round(base_hedged_val, 2),
                "base_value": round(base_val, 2),
                "base_unhedged_value": round(base_unhedged_val, 2),
                "volatility_index": vol,
                "forward_exchange_rate": round(forward_rate, 4),
                "carry_cost_valuation": round(carry_cost_impact, 2),
                "due_days": days_due,
                "standalone_var": round(standalone_var, 2)
            })

        cov_matrix = self.get_empirical_covariance_matrix(currencies_in_portfolio)
        W = np.array(naked_weights_base)
        
        if len(W) > 0 and np.sum(W) > 0:
            portfolio_variance = np.dot(W.T, np.dot(cov_matrix, W))
            portfolio_std_dev = np.sqrt(max(0.0, portfolio_variance))
            avg_days = int(exposure_df['due_days'].mean())
            diversified_portfolio_var = portfolio_std_dev * 1.645 * np.sqrt(avg_days / 365.0)
        else:
            diversified_portfolio_var = 0.0

        diversification_benefit = max(0.0, undiversified_var_sum - diversified_portfolio_var)

        return {
            "base_currency_flag": base_currency,
            "total_portfolio_exposure_base": round(total_portfolio_exposure, 2),
            "residual_unhedged_portfolio_var_base": round(diversified_portfolio_var, 2),
            "global_portfolio_estimated_bank_fees_base": round(abs(total_carry_costs), 2),
            "undiversified_var_sum": round(undiversified_var_sum, 2),
            "diversification_benefit": round(diversification_benefit, 2),
            "breakdown": processed_records
        }