import os
from dotenv import load_dotenv

# Initialize and pull variables from the local secure .env file automatically
load_dotenv()

class SystemSettings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEFAULT_LLM_MODEL = "gpt-4o" # Multi-modal powerhouse for structural reasoning
    
    # Global System Constraints
    MAX_UNHEDGED_VAR_THRESHOLD_USD = 500000.00  # If total portfolio VaR exceeds $500k, force alert state
    
    @classmethod
    def validate_environment(cls):
        """Ensures critical infrastructure secrets are actively present."""
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY == "your_actual_api_key_here":
            print("⚠️ WARNING: OPENAI_API_KEY is missing or unconfigured in your .env file. Agent nodes will fail.")
        else:
            print("✨ System configurations loaded securely.")

if __name__ == "__main__":
    SystemSettings.validate_environment()