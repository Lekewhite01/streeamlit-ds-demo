import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import OneHotEncoder
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import os
import anthropic
import openai  # if you're using the OpenAI API
from dotenv import load_dotenv

load_dotenv()

openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = anthropic.Anthropic(api_key=os.getenv("CLAUDEAI_API_KEY"))

class DemographicRiskAssessor:
    def __init__(self, reference_df: pd.DataFrame, features: list, cluster_col: str = 'cluster'):
        self.df = reference_df.copy()
        self.features = features
        self.cluster_col = cluster_col
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self._prepare_encoder()

    def _prepare_encoder(self):
        self.encoder.fit(self.df[self.features].astype(str))

    def _encode(self, df_subset):
        return self.encoder.transform(df_subset[self.features].astype(str))

    def get_cluster_distributions(self):
        clusters = self.df[self.cluster_col].unique()
        distributions = {}

        for cluster in clusters:
            subset = self.df[self.df[self.cluster_col] == cluster]
            encoded = self._encode(subset)
            # Mean vector as probability distribution
            distributions[cluster] = np.mean(encoded, axis=0)

        return distributions

    def assess_new_user(self, new_user_dict):
        user_df = pd.DataFrame([new_user_dict])
        user_encoded = self.encoder.transform(user_df[self.features].astype(str))[0]

        cluster_distributions = self.get_cluster_distributions()
        scores = {
            cluster: jensenshannon(user_encoded, dist)
            for cluster, dist in cluster_distributions.items()
        }

        best_match = min(scores.items(), key=lambda x: x[1])
        return best_match  # returns (cluster, distance)

# reference_df = filtered_df.copy()
# # Define demographic fields from the onboarding doc & schema
demo_features = [
    'bvnData.gender', 'bvnData.nationality', 'bvnData.state_of_origin',
    'bvnData.lga_of_origin', 'bvnData.state_of_residence', 'bvnData.lga_of_residence',
    'occupation', 'experience', 'tier'
]

class Insight(BaseModel):
    category: str  # e.g., 'Behavioral', 'Demographic'
    summary: str   # A concise human-readable insight
    detail: str    # Expanded explanation of the insight
    metric_examples: List[str] = Field(default_factory=list)

class ClusterInsights(BaseModel):
    cluster_id: int
    narrative: str
    insights: List[Insight]


class ClusterProfiler:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def extract_cluster_profile(self, cluster_id: int) -> dict:
        subset = self.df[self.df['cluster'] == cluster_id]

        profile = {
            "cluster_id": cluster_id,
            "top_occupations": subset['occupation'].value_counts().head(5).to_dict(),
            "top_states": subset['bvnData.state_of_residence'].value_counts().head(5).to_dict(),
            "avg_transaction_volume": subset['amount'].mean(),
            "avg_balance": subset['preBalance'].mean(),
            "activity_by_hour": subset['hour'].value_counts().sort_index().to_dict(),
            "unusual_txn_flag_rate": subset['unusual_large_transaction'].mean(),
            "channel_usage": {
                "ATM": subset['channel_atm'].mean(),
                "POS": subset['channel_pos'].mean(),
                "Transfer": subset['channel_transfer'].mean()
            },
            "weekend_activity_rate": subset['is_weekend'].mean(),
            "avg_experience": subset['experience'].value_counts().head(3).to_dict(),
            "gender_dist": subset['bvnData.gender'].value_counts(normalize=True).round(2).to_dict(),
            "risk_score": subset['risk_score'].mean(),
        }
        return profile


class InsightPromptBuilder:
    @staticmethod
    def build_prompt(profile: dict) -> str:
        return f"""
You're a financial risk analyst AI assistant.

You have the following profile summary for cluster ID {profile['cluster_id']}:
- Top occupations: {profile['top_occupations']}
- Top states of residence: {profile['top_states']}
- Average transaction volume: ₦{profile['avg_transaction_volume']:.2f}
- Average account balance: ₦{profile['avg_balance']:.2f}
- Hourly activity distribution: {profile['activity_by_hour']}
- Rate of unusual transactions: {profile['unusual_txn_flag_rate']:.2%}
- Channel usage (ATM, POS, Transfer): {profile['channel_usage']}
- Weekend activity rate: {profile['weekend_activity_rate']:.2%}
- Top experience levels: {profile['avg_experience']}
- Gender distribution: {profile['gender_dist']}
- Average risk score: {profile['risk_score']:.2f}

Please generate:
1. A concise, narrative-style summary of this cluster's behavioral and demographic traits.
2. A list of 4 categorized insights ('Behavioral', 'Transactional', 'Demographic', 'Temporal'), each with:
   - A brief summary
   - A more detailed explanation
   - Examples of metrics that support it

Return the result as a standard, python-parseable JSON structure adhering to this schema:
{{
  "cluster_id": int,
  "narrative": str,
  "insights": [
    {{
      "category": str,
      "summary": str,
      "detail": str,
      "metric_examples": [str]
    }}
  ]
}}
Be insightful and engaging, not just dry facts. Highlight anything unusual, significant, or particularly risky or stable.
"""
    

def query_gpt_for_insights(prompt: str) -> ClusterInsights:
    response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2
    )
    json_text = response.choices[0].message.content
    # Your raw GPT response
    raw_output = json_text.strip()

    # Remove triple backticks and language specifier
    if raw_output.startswith("```json"):
        raw_output = raw_output[7:]  # remove '```json\n'
    if raw_output.endswith("```"):
        raw_output = raw_output[:-3]  # remove '```'

    # Strip again to remove any leftover whitespace or newlines
    cleaned_json = raw_output.strip()

    return ClusterInsights.parse_raw(cleaned_json)

def query_claude_for_insights(prompt: str) -> ClusterInsights:
    """
    Query Claude API for cluster insights.
    
    Args:
        prompt: The formatted prompt string
        api_key: Your Claude API key
        
    Returns:
        ClusterInsights: Parsed insights from Claude's response
    """
    # client = anthropic.Anthropic(api_key=os.getenv("CLAUDEAI_API_KEY"))
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",  # You can also use "claude-3-opus-20240229" for higher quality
        max_tokens=2000,
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    json_text = response.content[0].text
    
    # Clean up the response (remove markdown formatting if present)
    raw_output = json_text.strip()
    
    # Remove triple backticks and language specifier
    if raw_output.startswith("```json"):
        raw_output = raw_output[7:]  # remove '```json\n'
    if raw_output.endswith("```"):
        raw_output = raw_output[:-3]  # remove '```'

    # Strip again to remove any leftover whitespace or newlines
    cleaned_json = raw_output.strip()

    return ClusterInsights.parse_raw(cleaned_json)


