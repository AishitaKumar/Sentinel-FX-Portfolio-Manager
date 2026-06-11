import os
import chromadb
from datetime import datetime

class SemanticMarketMemory:
    def __init__(self, db_dir="data/chroma_memory"):
        """Initializes the local vector database infrastructure."""
        os.makedirs(db_dir, exist_ok=True)
        # Initialize persistent client (saves data directly to the disk)
        self.client = chromadb.PersistentClient(path=db_dir)
        
        # We use Chroma's default embedding function for localized vector translation
        self.collection = self.client.get_or_create_collection(
            name="macro_economic_playbooks"
        )

    def seed_historical_events(self):
        """Seeds the vector DB with historical market anomalies to serve as RAG memory."""
        # Check if already seeded to prevent compounding duplication
        if self.collection.count() > 0:
            return

        print("Seeding Vector Database with historical macro regimes...")
        
        # Historical context blueprints for the AI to recall
        events = [
            {
                "id": "hist_001",
                "text": "Federal Reserve unexpected hawkish pivot during soaring inflation pressures. Interest rate differentials widened radically favoring USD outperformance over EUR and JPY.",
                "metadata": {"regime": "Hawkish Fed", "impact_currency": "USD", "severity": "High"}
            },
            {
                "id": "hist_002",
                "text": "Bank of Japan unexpected yield curve control expansion causing rapid short squeezing of JPY short positions. Massive localized volatility spikes across Asian trading desks.",
                "metadata": {"regime": "BOJ Intervention", "impact_currency": "JPY", "severity": "Extreme"}
            },
            {
                "id": "hist_003",
                "text": "European Central Bank executes emergency monetary easing policy due to economic cooling across manufacturing zones. Direct downward depreciation trendline forced onto EUR/USD pair.",
                "metadata": {"regime": "ECB Easing", "impact_currency": "EUR", "severity": "Medium"}
            }
        ]

        self.collection.add(
            documents=[e["text"] for e in events],
            metadatas=[e["metadata"] for e in events],
            ids=[e["id"] for e in events]
        )
        print(f"Vector database initialized with {self.collection.count()} reference states.")

    def search_similar_market_regimes(self, current_news_headline: str, limit=1) -> list:
        """Queries the vector database using vector similarity distance matching."""
        results = self.collection.query(
            query_texts=[current_news_headline],
            n_results=limit
        )
        
        # Format the query output back into a clean payload for our agents
        formatted_matches = []
        if results and results['documents']:
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                formatted_matches.append({
                    "historical_insight": doc,
                    "metadata_tags": meta
                })
        return formatted_matches

if __name__ == "__main__":
    print("Testing Semantic Market Memory Framework...")
    memory = SemanticMarketMemory()
    memory.seed_historical_events()
    
    # Let's test a mock query to prove the vector search can match contextually!
    test_headline = "Speculation mounts that Tokyo will step in to shore up the crashing yen currency."
    print(f"\nInput Live Headline: '{test_headline}'")
    
    matches = memory.search_similar_market_regimes(test_headline, limit=1)
    import json
    print("\nClosest Historical Vector Match Found:")
    print(json.dumps(matches, indent=4))