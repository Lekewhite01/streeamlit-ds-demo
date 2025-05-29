import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tabulate import tabulate
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from prettyprinter import pprint
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


def engineer_features(df):
    # Convert date to datetime and extract features
    df['transactionDate'] = pd.to_datetime(df['transactionDate'])
    df['hour'] = df['transactionDate'].dt.hour
    df['day_of_week'] = df['transactionDate'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Create transaction velocity features
    df['amount_balance_ratio'] = df['amount'] / (df['preBalance'] + 1)  # +1 to avoid division by zero
    
    # Create flags for unusual activity
    df['large_amount_flag'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
    df['unusual_hour_flag'] = ~df['hour'].between(6, 20).astype(int)
    
    # Create interaction features
    df['unusual_large_transaction'] = df['large_amount_flag'] & df['unusual_hour_flag']
    
    # Create channel aggregation
    df['channel'] = np.where(df['channel_atm'], 'atm', 
                           np.where(df['channel_pos'], 'pos', 
                                   np.where(df['channel_transfer'], 'transfer', 'other')))
    
    return df

def preprocess_data(df):
    # Select features for clustering
    cluster_features = [
        'amount', 'preBalance', 'hour', 'day_of_week', 'is_weekend',
        'amount_balance_ratio', 'large_amount_flag', 'unusual_hour_flag',
        'unusual_large_transaction', 'aggregate_30_amount_sum',
        'aggregate_30_count', 'recent_transactions'
    ]
    
    # Separate features
    X = df[cluster_features]
    
    # Define preprocessing pipeline
    numeric_features = [f for f in cluster_features if df[f].dtype != 'object']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)])
    
    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, preprocessor

def reduce_dimensions(X_processed):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
    plt.title('PCA of Transaction Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
    
    return X_pca

def kmeans_clustering(X_processed, df, optimal_k = 10):
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['kmeans_cluster'] = kmeans.fit_predict(X_processed)
    
    # Analyze clusters
    cluster_stats = df.groupby('kmeans_cluster').agg({
        'amount': ['mean', 'median', 'count'],
        'fraudulent': 'mean',
        'unusual_hour_flag': 'mean',
        'large_amount_flag': 'mean'
    })
    
    print("Cluster Statistics:")
    print(cluster_stats)
    
    return df, kmeans

def isolation_forest_anomaly(X_processed, df, pca_data):
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    anomalies = iso_forest.fit_predict(X_processed)
    df['is_anomaly_iso'] = np.where(anomalies == -1, 1, 0)
    
    print("Isolation Forest Anomalies:")
    print(df['is_anomaly_iso'].value_counts())
    
    # Compare with known fraudulent transactions
    if 'fraudulent' in df.columns:
        print("\nAnomaly vs Fraud Comparison:")
        print(pd.crosstab(df['fraudulent'], df['is_anomaly_iso']))
    
    # Visualize anomalies
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=df['is_anomaly_iso'], 
                    palette='viridis', alpha=0.7)
    plt.title('Isolation Forest Anomalies in PCA Space')
    plt.show()
    
    return df, iso_forest

def lof_anomaly(X_processed, df, pca_data):
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    anomalies = lof.fit_predict(X_processed)
    df['is_anomaly_lof'] = np.where(anomalies == -1, 1, 0)
    
    print("LOF Anomalies:")
    print(df['is_anomaly_lof'].value_counts())
    
    # Compare with known fraudulent transactions
    if 'fraudulent' in df.columns:
        print("\nAnomaly vs Fraud Comparison:")
        print(pd.crosstab(df['fraudulent'], df['is_anomaly_lof']))
    
    # Visualize anomalies
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=df['is_anomaly_lof'], 
                    palette='viridis', alpha=0.7)
    plt.title('LOF Anomalies in PCA Space')
    plt.show()
    
    return df, lof

def create_risk_score(df):
    # Normalize cluster risk (assuming higher cluster numbers are riskier)
    df['cluster_risk'] = df['kmeans_cluster'] / df['kmeans_cluster'].max()
    
    # Combine anomaly detection results
    df['combined_anomaly'] = (df['is_anomaly_iso'] + df['is_anomaly_lof']) / 2
    
    # Create composite risk score (0-100)
    df['risk_score'] = (
        0.4 * df['cluster_risk'] + 
        0.3 * df['combined_anomaly'] +
        0.2 * (df['large_amount_flag']) +
        0.1 * (df['unusual_hour_flag'])
    ) * 100
    
    # Cap at 100
    df['risk_score'] = df['risk_score'].clip(0, 100)
    
    # Analyze risk score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['risk_score'], bins=20, kde=True)
    plt.title('Risk Score Distribution')
    plt.show()
    
    # Compare with known fraud
    if 'fraudulent' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='fraudulent', y='risk_score', data=df)
        plt.title('Risk Score by Fraud Status')
        plt.show()
    
    return df

def generate_explanatory_report(df):
    # # Cluster profiles - with explicit column naming
    cluster_profiles = df.groupby('kmeans_cluster').agg({
        'amount': ['mean', 'median', 'max'],
        'hour': ['mean', ('odd_hours', lambda x: (x < 6).mean())],  # Named tuple for the lambda
        'is_weekend': 'mean',
        'fraudulent': 'mean',
        'risk_score': 'mean'
    }).round(2)
    
    # print("Cluster Profiles:")
    # print(cluster_profiles)
    
    # High-risk transactions
    high_risk = df[df['risk_score'] > 70]
    # print("\nHigh Risk Transactions Summary:")
    # print(high_risk.describe())
    
    # Create network graph of suspicious users
    suspicious_users = df[df['risk_score'] > 80]['user'].unique()
    suspicious_df = df[df['user'].isin(suspicious_users)]
    
    # Create edges between users and receivers/senders
    edges = []
    for _, row in suspicious_df.iterrows():
        if pd.notna(row['receiverAccountNumber']):
            edges.append((row['user'], row['receiverAccountNumber']))
        if pd.notna(row['senderAccountNumber']):
            edges.append((row['senderAccountNumber'], row['user']))
    
    
    # Narrative explanation - with fixed column reference
    print("\nNarrative Explanation:")
    print("""
    Our analysis identified {0} distinct transaction patterns through clustering. 
    Cluster {1} appears to be the highest risk, with {2}% of transactions marked as fraudulent.
    The risk scoring system flagged {3} transactions as high risk (score > 70).
    
    Key findings:
    - High-risk transactions tend to occur during {4}
    - The average amount of suspicious transactions is ₦{5:,.2f}
    - We've identified {6} potentially suspicious users in the network graph
    
    Recommendations:
    1. Review all transactions with risk score above 70
    2. Monitor users in the suspicious network graph
    3. Implement additional verification for transactions matching high-risk patterns
    """.format(
        len(df['kmeans_cluster'].unique()),
        cluster_profiles['fraudulent']['mean'].idxmax(),
        cluster_profiles['fraudulent']['mean'].max() * 100,
        len(high_risk),
        "odd hours" if cluster_profiles[('hour', 'odd_hours')].max() > 0.3 else "normal hours",  # Fixed reference
        high_risk['amount'].mean(),
        len(suspicious_users)
    ))
    # High-risk transactions
    high_risk = df[df['risk_score'] > 70]
    suspicious_users = df[df['risk_score'] > 80]['user'].unique()
    suspicious_df = df[df['user'].isin(suspicious_users)]
    
    # Create edges between users and receivers/senders
    edges = []
    for _, row in suspicious_df.iterrows():
        if pd.notna(row['receiverAccountNumber']):
            edges.append((row['user'], row['receiverAccountNumber']))
        if pd.notna(row['senderAccountNumber']):
            edges.append((row['senderAccountNumber'], row['user']))
    
    # ========== Enhanced Analysis Section ==========
    # Cluster pattern analysis
    high_risk_cluster = cluster_profiles['fraudulent']['mean'].idxmax()
    cluster_analysis = []
    
    for cluster in cluster_profiles.index:
        profile = {
            'Cluster': cluster,
            'Risk Level': 'High' if cluster == high_risk_cluster else 'Medium' if cluster_profiles.loc[cluster, ('fraudulent', 'mean')] > 0 else 'Low',
            'Typical Amount': f"₦{cluster_profiles.loc[cluster, ('amount', 'mean')]:,.2f}",
            'Odd Hour %': f"{cluster_profiles.loc[cluster, ('hour', 'odd_hours')]*100:.1f}%",
            'Weekend Activity': f"{cluster_profiles.loc[cluster, ('is_weekend', 'mean')]*100:.1f}%",
            'Fraud Rate': f"{cluster_profiles.loc[cluster, ('fraudulent', 'mean')]*100:.1f}%"
        }
        cluster_analysis.append(profile)
    
    # High-risk transaction patterns
    hr_patterns = {
        'Amount Threshold': f"₦{high_risk['amount'].quantile(0.25):,.0f}+",
        'Odd Hour Frequency': f"{(high_risk['hour'] < 6).mean()*100:.1f}%",
        'Top Transaction Types': high_risk['type'].value_counts().nlargest(3).to_dict(),
        'Common Narration Keywords': ', '.join(
            high_risk['narration'].str.lower().str.extract(r'(\w{6,})')[0]
            .value_counts().nlargest(3).index.tolist()
        )
    }
    
    # ========== Enhanced Reporting ==========
    print("\n" + "="*50)
    print("DETAILED TRANSACTION PATTERN ANALYSIS REPORT")
    print("="*50 + "\n")
    
    # 1. Cluster Profiles
    print("=== CLUSTER PROFILES ===")
    print(tabulate(cluster_analysis, headers="keys", tablefmt="grid"))
    
    # 2. High-Risk Patterns
    print("\n=== HIGH-RISK TRANSACTION PATTERNS ===")
    for k, v in hr_patterns.items():
        print(f"- {k}: {v}")
    
    # 3. Network Analysis
    print("\n=== NETWORK ANALYSIS ===")
    print(f"- {len(suspicious_users)} suspicious users identified")
    print(f"- {len(edges)} connections in suspicious network")
    
    # 4. Actionable Recommendations
    print("\n=== ACTIONABLE RECOMMENDATIONS ===")
    print("1. Immediate Actions:")
    print(f"   - Review all {len(high_risk)} high-risk transactions (>70 score)")
    print(f"   - Investigate cluster {high_risk_cluster} transactions")
    
    print("\n2. Monitoring Rules to Implement:")
    print("   - Auto-flag transactions matching ALL of:")
    print("     * Amount > ₦50,000")
    print("     * Occurs between 10PM-6AM")
    print("     * From users with 30-day aggregate > ₦1M")
    
    print("\n3. System Enhancements:")
    print("   - Implement real-time scoring for:")
    print("     * Odd-hour large transactions")
    print("     * Rapid succession high-value transfers")
    print("     * Weekend activity spikes")
    
    # # Network graph
    # G = nx.Graph()
    # G.add_edges_from(edges)
    # plt.figure(figsize=(12, 12))
    # nx.draw(G, with_labels=True, node_size=100, font_size=8)
    # plt.title('Suspicious User Network')
    # plt.show()
    
    # ========== Executive Summary ==========
    print("\n" + "="*50)
    print("EXECUTIVE SUMMARY")
    print("="*50)
    print(f"""
    Key Findings:
    - Highest risk cluster ({high_risk_cluster}) shows:
      • Average transaction: ₦{cluster_profiles.loc[high_risk_cluster, ('amount', 'mean')]:,.2f}
      • {cluster_profiles.loc[high_risk_cluster, ('hour', 'odd_hours')]*100:.1f}% odd-hour activity
      • {cluster_profiles.loc[high_risk_cluster, ('fraudulent', 'mean')]*100:.1f}% fraud rate
    
    - High-risk transactions:
      • Average amount: ₦{high_risk['amount'].mean():,.2f}
      • {hr_patterns['Odd Hour Frequency']} occur during odd hours
      • Top types: {', '.join([f"{k} ({v})" for k,v in hr_patterns['Top Transaction Types'].items()])}
    
    Recommended Actions:
    1. Prioritize investigation of {len(high_risk)} high-risk transactions
    2. Implement monitoring for odd-hour large transactions
    3. Review user connections in suspicious network
    """)

# def analyze_high_risk_patterns(df, cluster_num=None, risk_threshold=70):
#     """
#     Analyze high-risk patterns in a given cluster or across all clusters
    
#     Parameters:
#     - df: DataFrame containing transaction data
#     - cluster_num: Specific cluster to analyze (None analyzes all clusters)
#     - risk_threshold: Minimum risk score to consider as high-risk
#     """
#     # Filter for high-risk transactions in specified cluster(s)
#     if cluster_num is not None:
#         high_risk_df = df[(df['kmeans_cluster'] == cluster_num) & (df['risk_score'] > risk_threshold)]
#         cluster_label = f"Cluster {cluster_num}"
#     else:
#         high_risk_df = df[df['risk_score'] > risk_threshold]
#         cluster_label = "All Clusters"
    
#     if len(high_risk_df) == 0:
#         print(f"No high-risk transactions found in {cluster_label} (risk_score > {risk_threshold})")
#         return
    
#     # ========== Fraud Pattern Identification ==========
#     print("\n" + "="*50)
#     print(f"{cluster_label.upper()} HIGH-RISK FRAUD PATTERN ANALYSIS")
#     print("="*50 + "\n")
    
#     # 1. Temporal Patterns
#     hourly_patterns = high_risk_df.groupby('hour').size()
#     daily_patterns = high_risk_df.groupby('day_of_week').size()
    
#     # 2. Transaction Type Analysis
#     type_distribution = high_risk_df['type'].value_counts(normalize=True) * 100
    
#     # 3. Amount Patterns
#     amount_bins = pd.cut(high_risk_df['amount'], 
#                         bins=[0, 50000, 100000, 500000, float('inf')],
#                         labels=['<50k', '50-100k', '100-500k', '500k+'])
#     amount_distribution = amount_bins.value_counts(normalize=True) * 100
    
#     # 4. User Behavior Patterns
#     user_frequency = high_risk_df['user'].value_counts()
#     repeat_offenders = user_frequency[user_frequency > 1]
    
#     # ========== Enhanced Visualizations ==========
#     plt.figure(figsize=(18, 12))
    
#     # 1. Temporal Pattern Plot
#     plt.subplot(2, 2, 1)
#     hourly_patterns.plot(kind='line', marker='o', color='red')
#     plt.title(f'Hourly Distribution ({cluster_label})')
#     plt.xlabel('Hour of Day')
#     plt.ylabel('Transaction Count')
#     plt.axvspan(0, 6, color='red', alpha=0.1, label='High-Risk Hours')
#     plt.legend()
    
#     # 2. Transaction Type Plot
#     plt.subplot(2, 2, 2)
#     type_distribution.plot(kind='bar', color='orange')
#     plt.title('Transaction Type Distribution (%)')
#     plt.xlabel('Transaction Type')
#     plt.ylabel('Percentage')
#     plt.xticks(rotation=45)
    
#     # 3. Amount Distribution Plot
#     plt.subplot(2, 2, 3)
#     amount_distribution.plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
#     plt.title('Transaction Amount Distribution')
#     plt.ylabel('')
    
#     # 4. User Frequency Plot
#     plt.subplot(2, 2, 4)
#     if len(repeat_offenders) > 0:
#         repeat_offenders.plot(kind='bar', color='purple')
#     plt.title('Repeat Offenders (Users with >1 High-Risk Tx)')
#     plt.xlabel('User ID')
#     plt.ylabel('Transaction Count')
#     plt.xticks(rotation=45)
    
#     plt.tight_layout()
#     plt.show()
    
#     # ========== Pattern Analysis Report ==========
#     print("\n=== IDENTIFIED FRAUD PATTERNS ===")
    
#     # 1. Temporal Patterns
#     peak_hours = hourly_patterns.idxmax() if len(hourly_patterns) > 0 else None
#     print(f"\n1. Temporal Patterns:")
#     if peak_hours is not None:
#         print(f"- Peak risk hours: {peak_hours}:00-{peak_hours+1}:00")
#         odd_hour_percent = hourly_patterns[hourly_patterns.index < 6].sum()/len(high_risk_df)*100
#         print(f"- {odd_hour_percent:.1f}% occur during odd hours (12AM-6AM)")
    
#     # 2. Transaction Types
#     print("\n2. Transaction Type Patterns:")
#     for t, p in type_distribution.items():
#         print(f"- {t}: {p:.1f}% of high-risk transactions")
    
#     # 3. Amount Patterns
#     print("\n3. Amount Patterns:")
#     for a, p in amount_distribution.items():
#         print(f"- {a}: {p:.1f}% of cases")
    
#     # 4. User Patterns
#     print("\n4. User Behavior Patterns:")
#     if len(repeat_offenders) > 0:
#         print(f"- {len(repeat_offenders)} users with multiple high-risk transactions")
#         print(f"- Top offender: {repeat_offenders.index[0]} with {repeat_offenders.iloc[0]} transactions")
#     else:
#         print("- No repeat offenders found")
    
#     # 5. Network Patterns
#     print("\n5. Network Patterns:")
#     print(f"- {len(high_risk_df['receiverAccountNumber'].unique())} unique receivers")
#     print(f"- {len(high_risk_df['senderAccountNumber'].unique())} unique senders")
    
#     # ========== Fraud Type Classification ==========
#     print("\n=== POTENTIAL FRAUD TYPES ===")
    
#     fraud_detected = False
    
#     # 1. Money Laundering Indicators
#     ml_score = 0
#     ml_indicators = []
    
#     if amount_distribution.get('500k+', 0) > 20:
#         ml_score += 1
#         ml_indicators.append(f"- {amount_distribution['500k+']:.1f}% of transactions > ₦500k")
#     if type_distribution.get('transfer', 0) > 30:
#         ml_score += 1
#         ml_indicators.append(f"- {type_distribution['transfer']:.1f}% are transfer transactions")
#     if len(repeat_offenders) > 3:
#         ml_score += 1
#         ml_indicators.append(f"- {len(repeat_offenders)} users with repeated suspicious activity")
    
#     if ml_score >= 2:
#         print("\n🔴 Likely Money Laundering Patterns Detected:")
#         print("\n".join(ml_indicators))
#         fraud_detected = True
    
    
#     # 2. Account Takeover Indicators
#     ato_score = 0
#     ato_indicators = []
    
#     if type_distribution.get('withdrawal', 0) > 40:
#         ato_score += 1
#         ato_indicators.append(f"- {type_distribution['withdrawal']:.1f}% are withdrawals")
#     if (hourly_patterns.max() > hourly_patterns.mean() * 2):
#         ato_score += 1
#         peak_hour = hourly_patterns.idxmax()
#         ato_indicators.append(f"- Unusual activity spike at {peak_hour}:00")
#     if high_risk_df['is_weekend'].mean() > 0.5:
#         ato_score += 1
#         ato_indicators.append(f"- {high_risk_df['is_weekend'].mean()*100:.1f}% occur on weekends")
    
#     if ato_score >= 2:
#         print("\n🔴 Likely Account Takeover Patterns Detected:")
#         print("\n".join(ato_indicators))
#         fraud_detected = True
    
#     # 3. Synthetic Identity Fraud
#     syn_score = 0
#     syn_indicators = []
    
#     if len(repeat_offenders) > 5:
#         syn_score += 1
#         syn_indicators.append(f"- {len(repeat_offenders)} users with repeated activity")
#     if type_distribution.get('new_account', 0) > 15:
#         syn_score += 1
#         syn_indicators.append(f"- {type_distribution['new_account']:.1f}% are new account transactions")
#     if high_risk_df['preBalance'].median() < 1000:
#         syn_score += 1
#         syn_indicators.append(f"- Median account balance only ₦{high_risk_df['preBalance'].median():.2f}")
    
#     if syn_score >= 2:
#         print("\n🔴 Likely Synthetic Identity Patterns Detected:")
#         print("\n".join(syn_indicators))
#         fraud_detected = True
    
#     # 4. Transaction Laundering
#     tl_score = 0
#     tl_indicators = []
    
#     if type_distribution.get('payment', 0) > 50:
#         tl_score += 1
#         tl_indicators.append(f"- {type_distribution['payment']:.1f}% are payment transactions")
#     if amount_distribution.get('50-100k', 0) > 40:
#         tl_score += 1
#         tl_indicators.append(f"- {amount_distribution['50-100k']:.1f}% are ₦50k-₦100k transactions")
#     if len(high_risk_df['receiverAccountNumber'].unique()) < 5:
#         tl_score += 1
#         tl_indicators.append(f"- Concentrated on {len(high_risk_df['receiverAccountNumber'].unique())} receiver accounts")
    
#     if tl_score >= 2:
#         print("\n🔴 Likely Transaction Laundering Detected:")
#         print("\n".join(tl_indicators))
#         fraud_detected = True
    
#     # If no specific patterns detected
#     if not fraud_detected:
#         print("\n🟡 No strong patterns matching known fraud types detected.")
#         print("Review these potential indicators:")
#         print(f"- Most common transaction type: {type_distribution.idxmax()} ({type_distribution.max():.1f}%)")
#         print(f"- Peak activity hour: {hourly_patterns.idxmax()}:00")
#         print(f"- Most frequent amount range: {amount_distribution.idxmax()}")
#         print(f"- Top suspicious user: {repeat_offenders.index[0] if len(repeat_offenders) > 0 else 'N/A'}")
#     # ========== Enhanced Pattern Detection ==========
#     print("\n" + "="*50)
#     print("COMPREHENSIVE FRAUD PATTERN ANALYSIS")
#     print("="*50 + "\n")
    
#     # 1. Core Patterns (existing analysis)
#     hourly_patterns = high_risk_df.groupby('hour').size()
#     type_distribution = high_risk_df['type'].value_counts(normalize=True) * 100
#     amount_bins = pd.cut(high_risk_df['amount'], 
#                         bins=[0, 50000, 100000, 500000, float('inf')],
#                         labels=['<50k', '50-100k', '100-500k', '500k+'])
#     amount_distribution = amount_bins.value_counts(normalize=True) * 100
#     user_frequency = high_risk_df['user'].value_counts()
#     repeat_offenders = user_frequency[user_frequency > 1]
    
#     # ========== New Advanced Analyses ==========
    
#     # A. Geolocation Analysis
#     print("\n=== GEOLOCATION PATTERNS ===")
#     if 'transactionLocationLat' in df.columns:
#         # Calculate distance from account's usual location (simplified)
#         geo_cluster = high_risk_df[['transactionLocationLat', 'transactionLocationLong']].dropna()
#         if len(geo_cluster) > 0:
#             print(f"- Transactions span {len(geo_cluster)} unique locations")
            
#             # Plot locations
#             plt.figure(figsize=(10, 6))
#             plt.scatter(geo_cluster['transactionLocationLong'], 
#                        geo_cluster['transactionLocationLat'], 
#                        alpha=0.5, color='red')
#             plt.title('Geographic Distribution of High-Risk Transactions')
#             plt.xlabel('Longitude')
#             plt.ylabel('Latitude')
#             plt.grid()
#             plt.show()
            
#             # Suspicious location changes
#             user_locations = high_risk_df.groupby('user').agg({
#                 'transactionLocationLat': ['nunique', 'mean'],
#                 'transactionLocationLong': ['nunique', 'mean']
#             })
#             multi_location_users = user_locations[user_locations[('transactionLocationLat', 'nunique')] > 1]
#             if len(multi_location_users) > 0:
#                 print(f"- {len(multi_location_users)} users transacting from multiple locations")
#                 print(f"  Max locations per user: {user_locations[('transactionLocationLat', 'nunique')].max()}")
#         else:
#             print("- No location data available for high-risk transactions")
    
#     # B. Velocity Pattern Detection
#     print("\n=== TRANSACTION VELOCITY ===")
#     velocity_stats = high_risk_df.groupby('user').agg({
#         'transactionDate': ['count', lambda x: (x.max() - x.min()).total_seconds()/3600 if len(x) > 1 else 0]
#     })
#     velocity_stats.columns = ['tx_count', 'time_span_hours']
#     velocity_stats['tx_per_hour'] = velocity_stats['tx_count'] / velocity_stats['time_span_hours'].replace(0, 1)
    
#     high_velocity = velocity_stats[velocity_stats['tx_per_hour'] > 5]  # >5 tx/hour
#     if len(high_velocity) > 0:
#         print(f"- {len(high_velocity)} users with high transaction velocity (>5 tx/hr)")
#         print(f"  Max velocity: {velocity_stats['tx_per_hour'].max():.1f} tx/hour")
        
#         # Velocity visualization
#         plt.figure(figsize=(10, 4))
#         velocity_stats['tx_per_hour'].clip(0, 50).hist(bins=20)  # Cap at 50 for visualization
#         plt.title('Transaction Velocity Distribution (tx/hour)')
#         plt.xlabel('Transactions per hour')
#         plt.ylabel('User Count')
#         plt.show()
#     else:
#         print("- No extreme transaction velocity detected")
    
#     # C. Benford's Law Analysis
#     print("\n=== BENFORD'S LAW ANALYSIS ===")
#     def benfords_law_analysis(amounts):
#         # Get first digits (ignore 0 amounts)
#         first_digits = amounts[amounts > 0].astype(str).str[0].astype(int)
#         if len(first_digits) == 0:
#             return 0  # Return 0 if no valid amounts
        
#         # Calculate observed distribution
#         observed = first_digits.value_counts(normalize=True).sort_index()
#         expected = np.log10(1 + 1/np.arange(1, 10)) * 100
        
#         # Create index for plotting
#         digits = np.arange(1, 10)
        
#         plt.figure(figsize=(10, 4))
#         plt.bar(digits - 0.15, observed*100, width=0.3, label='Observed')
#         plt.bar(digits + 0.15, expected, width=0.3, label='Expected')
#         plt.title("Benford's Law Compliance Check")
#         plt.xlabel('First Digit')
#         plt.ylabel('Percentage')
#         plt.xticks(digits)
#         plt.legend()
#         plt.show()
        
#         # Chi-square test (align observed with expected)
#         observed_aligned = observed.reindex(digits, fill_value=0)
#         chi_square = ((observed_aligned*100 - expected)**2 / expected).sum() * len(amounts) / 100
#         return chi_square

#     if len(high_risk_df['amount']) > 0:
#         benford_chi = benfords_law_analysis(high_risk_df['amount'])
#         print(f"- Chi-square statistic: {benford_chi:.1f}")
#         if benford_chi > 20:  # Threshold for potential manipulation
#             print("  🚨 Significant deviation from Benford's Law - potential amount manipulation")
#         else:
#             print("  ✅ Amount distribution follows expected Benford's pattern")
#     else:
#         print("- No valid amounts for Benford's analysis")
    
#     # Add velocity checks to fraud types
#     if len(high_velocity) > 3:
#         print("\n🔴 Likely Bust-Out Fraud Detected:")
#         print(f"- {len(high_velocity)} users with rapid transaction bursts")
#         print(f"- Peak velocity: {velocity_stats['tx_per_hour'].max():.1f} tx/hour")
    
#     # Add geolocation checks
#     if 'transactionLocationLat' in df.columns and len(multi_location_users) > 2:
#         print("\n🔴 Likely Geographic Arbitrage Patterns:")
#         print(f"- {len(multi_location_users)} users operating across locations")
#         print(f"- Max locations per user: {user_locations[('transactionLocationLat', 'nunique')].max()}")
    
#     # Add velocity timeline plot
#     if len(high_risk_df) > 10:
#         plt.figure(figsize=(12, 4))
#         high_risk_df.set_index('transactionDate')['amount'].plot(
#             style='.', alpha=0.5, title='High-Risk Transactions Over Time')
#         plt.ylabel('Amount (₦)')
#         plt.show()


# def analyze_high_risk_patterns(df, cluster_num=None, risk_threshold=70):
#     """
#     Analyze high-risk patterns in a given cluster or across all clusters and return complete report
    
#     Parameters:
#     - df: DataFrame containing transaction data
#     - cluster_num: Specific cluster to analyze (None analyzes all clusters)
#     - risk_threshold: Minimum risk score to consider as high-risk
    
#     Returns:
#     - Complete report as a string
#     - List of matplotlib figures containing visualizations
#     """
#     # Initialize report components
#     report_parts = []
#     figures = []
    
#     # Filter for high-risk transactions in specified cluster(s)
#     if cluster_num is not None:
#         high_risk_df = df[(df['kmeans_cluster'] == cluster_num) & (df['risk_score'] >= risk_threshold)]
#         cluster_label = f"Cluster {cluster_num}"
#     else:
#         high_risk_df = df[df['risk_score'] >= risk_threshold]
#         cluster_label = "All Clusters"
    
#     if len(high_risk_df) == 0:
#         return f"No high-risk transactions found in {cluster_label} (risk_score > {risk_threshold})", None
    
#     # ====== Prepare Report Header ======
#     report_parts.append("\n" + "="*50)
#     report_parts.append(f"{cluster_label.upper()} HIGH-RISK FRAUD PATTERN ANALYSIS")
#     report_parts.append("="*50 + "\n")
    
#     # ====== Data Analysis ======
#     # 1. Temporal Patterns
#     hourly_patterns = high_risk_df.groupby('hour').size()
    
#     # 2. Transaction Type Analysis
#     type_distribution = high_risk_df['type'].value_counts(normalize=True) * 100
    
#     # 3. Amount Patterns
#     amount_bins = pd.cut(high_risk_df['amount'], 
#                         bins=[0, 50000, 100000, 500000, float('inf')],
#                         labels=['<50k', '50-100k', '100-500k', '500k+'])
#     amount_distribution = amount_bins.value_counts(normalize=True) * 100
    
#     # 4. User Behavior Patterns
#     user_frequency = high_risk_df['user'].value_counts()
#     repeat_offenders = user_frequency[user_frequency > 1]
    
#     # ====== Create Visualizations ======
#     # Figure 1: Hourly Distribution
#     fig1, ax1 = plt.subplots(figsize=(10, 5))
#     hourly_patterns.plot(kind='line', marker='o', color='red', ax=ax1)
#     ax1.set_title('Hourly Distribution of High-Risk Transactions')
#     ax1.set_xlabel('Hour of Day')
#     ax1.set_ylabel('Transaction Count')
#     ax1.axvspan(0, 6, color='red', alpha=0.1, label='High-Risk Hours')
#     ax1.legend()
#     figures.append(fig1)
    
#     # Figure 2: Transaction Type Distribution
#     fig2, ax2 = plt.subplots(figsize=(10, 5))
#     type_distribution.plot(kind='bar', color='orange', ax=ax2)
#     ax2.set_title('Transaction Type Distribution (%)')
#     ax2.set_xlabel('Transaction Type')
#     ax2.set_ylabel('Percentage')
#     ax2.tick_params(axis='x', rotation=45)
#     figures.append(fig2)
    
#     # Figure 3: Amount Distribution
#     fig3, ax3 = plt.subplots(figsize=(8, 8))
#     wedges, texts, autotexts = ax3.pie(
#         amount_distribution,
#         labels=amount_distribution.index,
#         autopct='%1.1f%%',
#         colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'],
#         startangle=90
#     )
#     ax3.set_title('Transaction Amount Distribution')
#     figures.append(fig3)
    
#     # Figure 4: Repeat Offenders
#     fig4, ax4 = plt.subplots(figsize=(10, 5))
#     if len(repeat_offenders) > 0:
#         repeat_offenders.plot(kind='bar', color='purple', ax=ax4)
#     ax4.set_title('Repeat Offenders (Users with >1 High-Risk Tx)')
#     ax4.set_xlabel('User ID')
#     ax4.set_ylabel('Transaction Count')
#     ax4.tick_params(axis='x', rotation=45)
#     figures.append(fig4)
    
#     # ====== Build Report Content ======
#     report_parts.append("\n=== IDENTIFIED FRAUD PATTERNS ===")
    
#     # 1. Temporal Patterns
#     peak_hours = hourly_patterns.idxmax() if len(hourly_patterns) > 0 else None
#     report_parts.append("\n1. Temporal Patterns:")
#     if peak_hours is not None:
#         report_parts.append(f"- Peak risk hours: {peak_hours}:00-{peak_hours+1}:00")
#         odd_hour_percent = hourly_patterns[hourly_patterns.index < 6].sum()/len(high_risk_df)*100
#         report_parts.append(f"- {odd_hour_percent:.1f}% occur during odd hours (12AM-6AM)")
    
#     # 2. Transaction Types
#     report_parts.append("\n2. Transaction Type Patterns:")
#     for t, p in type_distribution.items():
#         report_parts.append(f"- {t}: {p:.1f}% of high-risk transactions")
    
#     # 3. Amount Patterns
#     report_parts.append("\n3. Amount Patterns:")
#     for a, p in amount_distribution.items():
#         report_parts.append(f"- {a}: {p:.1f}% of cases")
    
#     # 4. User Patterns
#     report_parts.append("\n4. User Behavior Patterns:")
#     if len(repeat_offenders) > 0:
#         report_parts.append(f"- {len(repeat_offenders)} users with multiple high-risk transactions")
#         report_parts.append(f"- Top offender: {repeat_offenders.index[0]} with {repeat_offenders.iloc[0]} transactions")
#     else:
#         report_parts.append("- No repeat offenders found")
    
#     # 5. Network Patterns
#     report_parts.append("\n5. Network Patterns:")
#     report_parts.append(f"- {len(high_risk_df['receiverAccountNumber'].unique())} unique receivers")
#     report_parts.append(f"- {len(high_risk_df['senderAccountNumber'].unique())} unique senders")
    
#     # ====== Fraud Type Classification ======
#     report_parts.append("\n=== POTENTIAL FRAUD TYPES ===")
    
#     # [Include all your fraud type detection logic here]
#     # Just append to report_parts instead of printing
    
#     # ====== Final Report Assembly ======
#     full_report = "\n".join(report_parts)
#     return full_report, figures

def analyze_high_risk_patterns(df, cluster_num=None, risk_threshold=70):
    """
    Analyze high-risk patterns in a given cluster or across all clusters and return complete report
    
    Parameters:
    - df: DataFrame containing transaction data
    - cluster_num: Specific cluster to analyze (None analyzes all clusters)
    - risk_threshold: Minimum risk score to consider as high-risk
    
    Returns:
    - Complete report as a string
    - List of matplotlib figures containing visualizations
    """
    # Initialize report components
    report_parts = []
    figures = []
    
    # Filter for high-risk transactions in specified cluster(s)
    if cluster_num is not None:
        high_risk_df = df[(df['kmeans_cluster'] == cluster_num) & (df['risk_score'] > risk_threshold)]
        cluster_label = f"Cluster {cluster_num}"
    else:
        high_risk_df = df[df['risk_score'] > risk_threshold]
        cluster_label = "All Clusters"
    
    if len(high_risk_df) == 0:
        return f"No high-risk transactions found in {cluster_label} (risk_score > {risk_threshold})", None
    
    # ====== Prepare Report Header ======
    report_parts.append("\n" + "="*50)
    report_parts.append(f"{cluster_label.upper()} HIGH-RISK FRAUD PATTERN ANALYSIS")
    report_parts.append("="*50 + "\n")
    
    # ====== Data Analysis ======
    # 1. Temporal Patterns
    hourly_patterns = high_risk_df.groupby('hour').size()
    daily_patterns = high_risk_df.groupby('day_of_week').size()
    
    # 2. Transaction Type Analysis
    type_distribution = high_risk_df['type'].value_counts(normalize=True) * 100
    
    # 3. Amount Patterns
    amount_bins = pd.cut(high_risk_df['amount'], 
                        bins=[0, 50000, 100000, 500000, float('inf')],
                        labels=['<50k', '50-100k', '100-500k', '500k+'])
    amount_distribution = amount_bins.value_counts(normalize=True) * 100
    
    # 4. User Behavior Patterns
    user_frequency = high_risk_df['user'].value_counts()
    repeat_offenders = user_frequency[user_frequency > 1]
    
    # ====== Create Visualizations ======
    # Figure 1: Hourly Distribution
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    hourly_patterns.plot(kind='line', marker='o', color='red', ax=ax1)
    ax1.set_title('Hourly Distribution of High-Risk Transactions')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Transaction Count')
    ax1.axvspan(0, 6, color='red', alpha=0.1, label='High-Risk Hours')
    ax1.legend()
    figures.append(fig1)
    
    # Figure 2: Transaction Type Distribution
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    type_distribution.plot(kind='bar', color='orange', ax=ax2)
    ax2.set_title('Transaction Type Distribution (%)')
    ax2.set_xlabel('Transaction Type')
    ax2.set_ylabel('Percentage')
    ax2.tick_params(axis='x', rotation=45)
    figures.append(fig2)
    
    # Figure 3: Amount Distribution
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax3.pie(
        amount_distribution,
        labels=amount_distribution.index,
        autopct='%1.1f%%',
        colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'],
        startangle=90
    )
    ax3.set_title('Transaction Amount Distribution')
    figures.append(fig3)
    
    # Figure 4: Repeat Offenders
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    if len(repeat_offenders) > 0:
        repeat_offenders.plot(kind='bar', color='purple', ax=ax4)
    ax4.set_title('Repeat Offenders (Users with >1 High-Risk Tx)')
    ax4.set_xlabel('User ID')
    ax4.set_ylabel('Transaction Count')
    ax4.tick_params(axis='x', rotation=45)
    figures.append(fig4)
    
    # ====== Build Report Content ======
    report_parts.append("\n=== IDENTIFIED FRAUD PATTERNS ===")
    
    # 1. Temporal Patterns
    peak_hours = hourly_patterns.idxmax() if len(hourly_patterns) > 0 else None
    report_parts.append("\n1. Temporal Patterns:")
    if peak_hours is not None:
        report_parts.append(f"- Peak risk hours: {peak_hours}:00-{peak_hours+1}:00")
        odd_hour_percent = hourly_patterns[hourly_patterns.index < 6].sum()/len(high_risk_df)*100
        report_parts.append(f"- {odd_hour_percent:.1f}% occur during odd hours (12AM-6AM)")
    
    # 2. Transaction Types
    report_parts.append("\n2. Transaction Type Patterns:")
    for t, p in type_distribution.items():
        report_parts.append(f"- {t}: {p:.1f}% of high-risk transactions")
    
    # 3. Amount Patterns
    report_parts.append("\n3. Amount Patterns:")
    for a, p in amount_distribution.items():
        report_parts.append(f"- {a}: {p:.1f}% of cases")
    
    # 4. User Patterns
    report_parts.append("\n4. User Behavior Patterns:")
    if len(repeat_offenders) > 0:
        report_parts.append(f"- {len(repeat_offenders)} users with multiple high-risk transactions")
        report_parts.append(f"- Top offender: {repeat_offenders.index[0]} with {repeat_offenders.iloc[0]} transactions")
    else:
        report_parts.append("- No repeat offenders found")
    
    # 5. Network Patterns
    report_parts.append("\n5. Network Patterns:")
    report_parts.append(f"- {len(high_risk_df['receiverAccountNumber'].unique())} unique receivers")
    report_parts.append(f"- {len(high_risk_df['senderAccountNumber'].unique())} unique senders")
    
    # ====== Fraud Type Classification ======
    report_parts.append("\n=== POTENTIAL FRAUD TYPES ===")
    
    fraud_detected = False
    
    # 1. Money Laundering Indicators
    ml_score = 0
    ml_indicators = []
    
    if amount_distribution.get('500k+', 0) > 20:
        ml_score += 1
        ml_indicators.append(f"- {amount_distribution['500k+']:.1f}% of transactions > ₦500k")
    if type_distribution.get('transfer', 0) > 30:
        ml_score += 1
        ml_indicators.append(f"- {type_distribution['transfer']:.1f}% are transfer transactions")
    if len(repeat_offenders) > 3:
        ml_score += 1
        ml_indicators.append(f"- {len(repeat_offenders)} users with repeated suspicious activity")
    
    if ml_score >= 2:
        report_parts.append("\n🔴 Likely Money Laundering Patterns Detected:")
        report_parts.extend(ml_indicators)
        fraud_detected = True
    
    # 2. Account Takeover Indicators
    ato_score = 0
    ato_indicators = []
    
    if type_distribution.get('withdrawal', 0) > 40:
        ato_score += 1
        ato_indicators.append(f"- {type_distribution['withdrawal']:.1f}% are withdrawals")
    if (hourly_patterns.max() > hourly_patterns.mean() * 2):
        ato_score += 1
        peak_hour = hourly_patterns.idxmax()
        ato_indicators.append(f"- Unusual activity spike at {peak_hour}:00")
    if high_risk_df['is_weekend'].mean() > 0.5:
        ato_score += 1
        ato_indicators.append(f"- {high_risk_df['is_weekend'].mean()*100:.1f}% occur on weekends")
    
    if ato_score >= 2:
        report_parts.append("\n🔴 Likely Account Takeover Patterns Detected:")
        report_parts.extend(ato_indicators)
        fraud_detected = True
    
    # 3. Synthetic Identity Fraud
    syn_score = 0
    syn_indicators = []
    
    if len(repeat_offenders) > 5:
        syn_score += 1
        syn_indicators.append(f"- {len(repeat_offenders)} users with repeated activity")
    if type_distribution.get('new_account', 0) > 15:
        syn_score += 1
        syn_indicators.append(f"- {type_distribution['new_account']:.1f}% are new account transactions")
    if high_risk_df['preBalance'].median() < 1000:
        syn_score += 1
        syn_indicators.append(f"- Median account balance only ₦{high_risk_df['preBalance'].median():.2f}")
    
    if syn_score >= 2:
        report_parts.append("\n🔴 Likely Synthetic Identity Patterns Detected:")
        report_parts.extend(syn_indicators)
        fraud_detected = True
    
    # 4. Transaction Laundering
    tl_score = 0
    tl_indicators = []
    
    if type_distribution.get('payment', 0) > 50:
        tl_score += 1
        tl_indicators.append(f"- {type_distribution['payment']:.1f}% are payment transactions")
    if amount_distribution.get('50-100k', 0) > 40:
        tl_score += 1
        tl_indicators.append(f"- {amount_distribution['50-100k']:.1f}% are ₦50k-₦100k transactions")
    if len(high_risk_df['receiverAccountNumber'].unique()) < 5:
        tl_score += 1
        tl_indicators.append(f"- Concentrated on {len(high_risk_df['receiverAccountNumber'].unique())} receiver accounts")
    
    if tl_score >= 2:
        report_parts.append("\n🔴 Likely Transaction Laundering Detected:")
        report_parts.extend(tl_indicators)
        fraud_detected = True
    
    # If no specific patterns detected
    if not fraud_detected:
        report_parts.append("\n🟡 No strong patterns matching known fraud types detected.")
        report_parts.append("Review these potential indicators:")
        report_parts.append(f"- Most common transaction type: {type_distribution.idxmax()} ({type_distribution.max():.1f}%)")
        report_parts.append(f"- Peak activity hour: {hourly_patterns.idxmax()}:00")
        report_parts.append(f"- Most frequent amount range: {amount_distribution.idxmax()}")
        report_parts.append(f"- Top suspicious user: {repeat_offenders.index[0] if len(repeat_offenders) > 0 else 'N/A'}")
    
    # ====== Enhanced Pattern Detection ======
    report_parts.append("\n" + "="*50)
    report_parts.append("COMPREHENSIVE FRAUD PATTERN ANALYSIS")
    report_parts.append("="*50 + "\n")
    
    # A. Geolocation Analysis
    report_parts.append("\n=== GEOLOCATION PATTERNS ===")
    if 'transactionLocationLat' in df.columns:
        geo_cluster = high_risk_df[['transactionLocationLat', 'transactionLocationLong']].dropna()
        if len(geo_cluster) > 0:
            report_parts.append(f"- Transactions span {len(geo_cluster)} unique locations")
            
            # Create geolocation plot
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            ax5.scatter(geo_cluster['transactionLocationLong'], 
                       geo_cluster['transactionLocationLat'], 
                       alpha=0.5, color='red')
            ax5.set_title('Geographic Distribution of High-Risk Transactions')
            ax5.set_xlabel('Longitude')
            ax5.set_ylabel('Latitude')
            ax5.grid()
            figures.append(fig5)
            
            user_locations = high_risk_df.groupby('user').agg({
                'transactionLocationLat': ['nunique', 'mean'],
                'transactionLocationLong': ['nunique', 'mean']
            })
            multi_location_users = user_locations[user_locations[('transactionLocationLat', 'nunique')] > 1]
            if len(multi_location_users) > 0:
                report_parts.append(f"- {len(multi_location_users)} users transacting from multiple locations")
                report_parts.append(f"  Max locations per user: {user_locations[('transactionLocationLat', 'nunique')].max()}")
        else:
            report_parts.append("- No location data available for high-risk transactions")
    
    # B. Velocity Pattern Detection
    report_parts.append("\n=== TRANSACTION VELOCITY ===")
    velocity_stats = high_risk_df.groupby('user').agg({
        'transactionDate': ['count', lambda x: (x.max() - x.min()).total_seconds()/3600 if len(x) > 1 else 0]
    })
    velocity_stats.columns = ['tx_count', 'time_span_hours']
    velocity_stats['tx_per_hour'] = velocity_stats['tx_count'] / velocity_stats['time_span_hours'].replace(0, 1)
    
    high_velocity = velocity_stats[velocity_stats['tx_per_hour'] > 5]  # >5 tx/hour
    if len(high_velocity) > 0:
        report_parts.append(f"- {len(high_velocity)} users with high transaction velocity (>5 tx/hr)")
        report_parts.append(f"  Max velocity: {velocity_stats['tx_per_hour'].max():.1f} tx/hour")
        
        # Velocity visualization
        fig6, ax6 = plt.subplots(figsize=(10, 4))
        velocity_stats['tx_per_hour'].clip(0, 50).hist(bins=20, ax=ax6)
        ax6.set_title('Transaction Velocity Distribution (tx/hour)')
        ax6.set_xlabel('Transactions per hour')
        ax6.set_ylabel('User Count')
        figures.append(fig6)
    else:
        report_parts.append("- No extreme transaction velocity detected")

    # C. Benford's Law Analysis
    report_parts.append("\n=== BENFORD'S LAW ANALYSIS ===")
    def benfords_law_analysis(amounts):
        # Get first digits (ignore 0 amounts)
        first_digits = amounts[amounts > 0].astype(str).str[0].astype(int)
        if len(first_digits) == 0:
            return 0, None  # Return 0 and None figure if no valid amounts
        
        # Calculate observed distribution (ensure we have all digits 1-9)
        observed_counts = first_digits.value_counts()
        observed = pd.Series(0, index=np.arange(1, 10))  # Initialize with zeros for digits 1-9
        observed.update(observed_counts)  # Update with actual counts
        observed_pct = (observed / observed.sum()) * 100  # Convert to percentages
        
        # Expected distribution according to Benford's Law
        expected = np.log10(1 + 1/np.arange(1, 10)) * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 4))
        digits = np.arange(1, 10)
        ax.bar(digits - 0.15, observed_pct, width=0.3, label='Observed')
        ax.bar(digits + 0.15, expected, width=0.3, label='Expected')
        ax.set_title("Benford's Law Compliance Check")
        ax.set_xlabel('First Digit')
        ax.set_ylabel('Percentage')
        ax.set_xticks(digits)
        ax.legend()
        
        # Calculate chi-square statistic
        chi_square = ((observed_pct - expected)**2 / expected).sum() * len(amounts) / 100
        return chi_square, fig

    if len(high_risk_df['amount']) > 0:
        benford_chi, benford_fig = benfords_law_analysis(high_risk_df['amount'])
        if benford_fig is not None:
            figures.append(benford_fig)
            report_parts.append(f"- Chi-square statistic: {benford_chi:.1f}")
            if benford_chi > 20:  # Threshold for potential manipulation
                report_parts.append("  🚨 Significant deviation from Benford's Law - potential amount manipulation")
            else:
                report_parts.append("  ✅ Amount distribution follows expected Benford's pattern")
        else:
            report_parts.append("- No valid amounts for Benford's analysis")
    else:
        report_parts.append("- No valid amounts for Benford's analysis")

    
    # # C. Benford's Law Analysis
    # report_parts.append("\n=== BENFORD'S LAW ANALYSIS ===")
    # def benfords_law_analysis(amounts):
    #     first_digits = amounts[amounts > 0].astype(str).str[0].astype(int)
    #     if len(first_digits) == 0:
    #         return 0
        
    #     observed = first_digits.value_counts(normalize=True).sort_index()
    #     expected = np.log10(1 + 1/np.arange(1, 10)) * 100
    #     digits = np.arange(1, 10)
        
    #     fig7, ax7 = plt.subplots(figsize=(10, 4))
    #     ax7.bar(digits - 0.15, observed*100, width=0.3, label='Observed')
    #     ax7.bar(digits + 0.15, expected, width=0.3, label='Expected')
    #     ax7.set_title("Benford's Law Compliance Check")
    #     ax7.set_xlabel('First Digit')
    #     ax7.set_ylabel('Percentage')
    #     ax7.set_xticks(digits)
    #     ax7.legend()
    #     figures.append(fig7)
        
    #     observed_aligned = observed.reindex(digits, fill_value=0)
    #     chi_square = ((observed_aligned*100 - expected)**2 / expected).sum() * len(amounts) / 100
    #     return chi_square

    # if len(high_risk_df['amount']) > 0:
    #     benford_chi = benfords_law_analysis(high_risk_df['amount'])
    #     report_parts.append(f"- Chi-square statistic: {benford_chi:.1f}")
    #     if benford_chi > 20:
    #         report_parts.append("  🚨 Significant deviation from Benford's Law - potential amount manipulation")
    #     else:
    #         report_parts.append("  ✅ Amount distribution follows expected Benford's pattern")
    # else:
    #     report_parts.append("- No valid amounts for Benford's analysis")
    
    # # Add velocity checks to fraud types
    # if len(high_velocity) > 3:
    #     report_parts.append("\n🔴 Likely Bust-Out Fraud Detected:")
    #     report_parts.append(f"- {len(high_velocity)} users with rapid transaction bursts")
    #     report_parts.append(f"- Peak velocity: {velocity_stats['tx_per_hour'].max():.1f} tx/hour")
    
    # # Add geolocation checks
    # if 'transactionLocationLat' in df.columns and len(multi_location_users) > 2:
    #     report_parts.append("\n🔴 Likely Geographic Arbitrage Patterns:")
    #     report_parts.append(f"- {len(multi_location_users)} users operating across locations")
    #     report_parts.append(f"- Max locations per user: {user_locations[('transactionLocationLat', 'nunique')].max()}")
    
    # # Add velocity timeline plot
    # if len(high_risk_df) > 10:
    #     fig8, ax8 = plt.subplots(figsize=(12, 4))
    #     high_risk_df.set_index('transactionDate')['amount'].plot(
    #         style='.', alpha=0.5, ax=ax8, title='High-Risk Transactions Over Time')
    #     ax8.set_ylabel('Amount (₦)')
    #     figures.append(fig8)
    
    # # ====== Final Report Assembly ======
    # full_report = "\n".join(report_parts)
    # return full_report, figures