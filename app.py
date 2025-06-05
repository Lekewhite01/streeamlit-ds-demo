import streamlit as st
import pandas as pd
import json
import boto3
import time
import matplotlib.pyplot as plt
import io
from io import StringIO
from contextlib import redirect_stdout
from sklearn.cluster import KMeans
from fraud_utils import (
    engineer_features, preprocess_data, reduce_dimensions,
    kmeans_clustering, isolation_forest_anomaly, lof_anomaly,
    create_risk_score, generate_explanatory_report, analyze_high_risk_patterns
)
from onboarding import (ClusterProfiler, InsightPromptBuilder,  DemographicRiskAssessor, 
                        demo_features, query_gpt_for_insights, query_claude_for_insights)

# Configure Streamlit page
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🕵️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stFileUploader>div>div>div>div {color: #4CAF50;}
    .stSlider>div>div>div>div {background-color: #4CAF50;}
    .report-view-container {background-color: white; padding: 20px; border-radius: 10px;}
    </style>
    """, unsafe_allow_html=True)

# Initialize AWS clients (you'll need to set up your AWS credentials)
def get_aws_client(service_name):
    return boto3.client(
        service_name,
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets["AWS_REGION"]
    )

# Page selection in sidebar
page = st.sidebar.selectbox(
    "Select Analysis Mode",
    ["Transaction Clustering", "Real-time Screening", "Onboarding risk", "About"]
)

if page == "Transaction Clustering":

    # App title and description
    st.title("🕵️ Fraud Detection Dashboard")
    st.markdown("""
    This dashboard helps identify potentially fraudulent transactions using:
    - **Clustering analysis** to group similar transactions
    - **Anomaly detection** to flag unusual activity
    - **Risk scoring** to prioritize investigations
    """)

    # File uploader section
    st.header("1. Data Upload")
    uploaded_file = st.file_uploader("Upload transaction data (CSV format)", type=["csv"])

    if uploaded_file:
        # Load and process data
        with st.spinner('Loading and processing data...'):
            df = pd.read_csv(uploaded_file)
            df = engineer_features(df)
            X_processed, preprocessor = preprocess_data(df)
            X_pca = reduce_dimensions(X_processed)
        
        st.success("✅ Data loaded and processed successfully!")
        
        # Show sample data
        if st.checkbox("Show sample data"):
            st.dataframe(df.head())
        
        # Clustering configuration
        st.header("2. Clustering Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            optimal_k = st.slider(
                "Select number of clusters (K)",
                min_value=2,
                max_value=10,
                value=4,
                help="More clusters will identify finer patterns but may overfit"
            )
        
        with col2:
            contamination = st.slider(
                "Anomaly detection sensitivity",
                min_value=0.01,
                max_value=0.2,
                value=0.05,
                step=0.01,
                help="Higher values will flag more transactions as anomalies"
            )

        with col3:
            risk_threshold = st.slider(
                "Set risk threshold for filtering",
                min_value=50,
                max_value=100,
                value=70
            )
            # risk_threshold = st.slider("High-risk threshold", 50, 100, 70)
        
        # Run analysis
        if st.button("Run Fraud Analysis"):
            with st.spinner('Running clustering analysis...'):
                # df, kmeans = kmeans_clustering(X_processed, df, optimal_k)
                kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                df['kmeans_cluster'] = kmeans.fit_predict(X_processed)
            
            with st.spinner('Detecting anomalies...'):
                df, iso_forest = isolation_forest_anomaly(X_processed, df, X_pca)
                df, lof = lof_anomaly(X_processed, df, X_pca)
            
            with st.spinner('Calculating risk scores...'):
                df = create_risk_score(df)
            
            # Display PCA plot
            st.header("3. Data Visualization")
            st.subheader("Transaction Clusters (PCA Reduced)")
            fig = plt.figure(figsize=(10, 6))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['kmeans_cluster'], alpha=0.5, cmap='viridis')
            plt.title('Transaction Clusters in PCA Space')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.colorbar(label='Cluster')
            st.pyplot(fig)
            
            # Show risk score distribution
            st.subheader("Risk Score Distribution")
            fig2 = plt.figure(figsize=(10, 4))
            plt.hist(df['risk_score'], bins=20, color='blue', alpha=0.7)
            plt.title('Distribution of Risk Scores')
            plt.xlabel('Risk Score')
            plt.ylabel('Number of Transactions')
            st.pyplot(fig2)
            

            # Generate and display reports
            st.header("4. Analysis Reports")

            with st.expander("📊 Comprehensive Fraud Analysis Report", expanded=True):
                # Create a placeholder to capture printed output
                report_placeholder = st.empty()
                
                # Redirect stdout to capture the printed report
                
                f = io.StringIO()
                with redirect_stdout(f):
                    generate_explanatory_report(df)
                report_placeholder.text(f.getvalue())
            
            # Download results
            st.header("5. Download Results")
            st.download_button(
                label="Download Analyzed Data",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='fraud_analysis_results.csv',
                mime='text/csv'
            )
            
            # Show high-risk transactions
            st.header("6. High-Risk Transactions")
            risk_threshold = st.slider(
                "Set risk threshold for filtering",
                min_value=0,
                max_value=100,
                value=70
            )
            high_risk = df[df['risk_score'] >= risk_threshold]
            st.write(f"Found {len(high_risk)} transactions with risk score ≥ {risk_threshold}")
            st.dataframe(high_risk.sort_values('risk_score', ascending=False))
    else:
        st.info("👆 Please upload a CSV file to begin analysis")

        
elif page == "Real-time Screening":
    st.title("Real-time Transaction Screening")
    
    # User input form
    with st.form("screening_form"):
        col1, col2 = st.columns(2)
        with col1:
            # transaction_id = st.text_input("Transaction ID")
            product_type = st.selectbox(
                "Product Type",
                ["transactionScreening", "transactionMonitoring"]
            )
        with col2:
            user_id = st.text_input("User ID")
            business = st.selectbox(
                "Business",
                ["Shago", "Other"]
            )
        
        submitted = st.form_submit_button("Screen Transaction")
    
    if submitted:
        # Prepare payload for Lambda
        payload = {
            "id": [user_id],
            "product": product_type,
            "business": business
        }
        
        with st.spinner('Screening transaction...'):
            # Invoke Lambda function
            lambda_client = get_aws_client("lambda")
            response = lambda_client.invoke(
                FunctionName=st.secrets["LAMBDA_FUNCTION_NAME"],
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            # Parse response
            result = json.loads(response['Payload'].read().decode())
            # st.write(result)
            # Display results
            st.subheader("Screening Results")
            
            if result["result"][0] == "1":  # Flagged transaction
                st.error("🚨 Transaction flagged for review")
            else:
                st.success("✅ Transaction appears normal")
            
            st.metric("Risk Score", f"{float(result['prob'][0])*100:.2f}%")
            
            if "explanation" in result and result["explanation"]:
                with st.expander("Explanation"):
                    st.write(result["explanation"])
            
            # Show cluster information if available
            if "cluster_id" in result:
                st.subheader("Transaction Cluster Profile")
                st.json(result["cluster_profiles"])

elif page == "Onboarding risk":
    st.write("Upload your clustered dataset and analyze a specific cluster using AI.")

    with st.sidebar:
        st.subheader("Upload CSV")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        reference_df = pd.read_csv(uploaded_file)
        st.success("CSV uploaded successfully!")

        with st.form("new_user_form"):
            st.subheader("New User Profile")
            gender = st.selectbox("Gender", ["Male", "Female"])
            nationality = st.text_input("Nationality", "Nigerian")
            state_of_origin = st.text_input("State of Origin", "Lagos")
            lga_of_origin = st.text_input("LGA of Origin", "Ikeja")
            state_of_residence = st.text_input("State of Residence", "Lagos")
            lga_of_residence = st.text_input("LGA of Residence", "Eti-Osa")
            occupation = st.text_input("Occupation", "Trader")
            experience = st.selectbox("Experience", ["individual", "signature"])
            tier = st.selectbox("Tier", ["Tier 1", "Tier 2", "Tier 3"])
            submit_button = st.form_submit_button("Analyze")

        if submit_button:
            new_user = {
                'bvnData.gender': gender,
                'bvnData.nationality': nationality,
                'bvnData.state_of_origin': state_of_origin,
                'bvnData.lga_of_origin': lga_of_origin,
                'bvnData.state_of_residence': state_of_residence,
                'bvnData.lga_of_residence': lga_of_residence,
                'occupation': occupation,
                'experience': experience,
                'tier': tier
            }

            assessor = DemographicRiskAssessor(reference_df, demo_features)
            best_cluster, jsd_score = assessor.assess_new_user(new_user)
            cluster_risk = reference_df[reference_df['cluster'] == best_cluster]['cluster_risk'].mean()
            st.info(
                f"**Closest Cluster**: {best_cluster}\n"
                f"**Jensen-Shannon Score**: {jsd_score:.4f}\n"
                f"**Estimated Cluster Risk**: {cluster_risk:.4f}"
            )

            st.session_state["best_cluster"] = best_cluster

        # Streamlit form
        with st.form("cluster_form"):
            # cluster_id = st.number_input("Enter Cluster ID to Analyze", placeholder=st.session_state.get("best_cluster", None), min_value=0, step=1)
            cluster_id = st.text_input("Enter Cluster ID to Analyze", placeholder=st.session_state.get("best_cluster", None))
            submit_button_ai = st.form_submit_button("Generate Insight")

        if submit_button_ai:
            try:
                # Extract cluster profile and generate AI explanation
                profiler = ClusterProfiler(reference_df)
                cluster_profile = profiler.extract_cluster_profile(cluster_id=st.session_state.get("best_cluster", None))

                prompt = InsightPromptBuilder.build_prompt(cluster_profile)
                # insight_report = query_gpt_for_insights(prompt)
                insight_report = query_claude_for_insights(prompt)

                st.subheader("AI Narrative")
                st.write(insight_report.narrative)

                st.subheader("AI Insights")
                for insight in insight_report.insights:
                    st.markdown(f"**{insight.category} Insight: {insight.summary}**")
                    st.markdown(insight.detail)
                    st.markdown("_Metrics: " + ", ".join(insight.metric_examples) + "_")

            except Exception as e:
                st.error(f"Error generating insight: {e}")
    else:
        st.info("Please upload a CSV file to get started.")

elif page == "About":
    st.title("About This Application")
    st.write("""
    This fraud detection system combines two approaches:
    
    1. **Transaction Clustering**: Analyzes historical transaction patterns to identify suspicious clusters
    2. **Real-time Screening**: Evaluates individual transactions using machine learning models
    
    The system helps financial institutions detect potential fraud through:
    - Anomaly detection
    - Behavioral pattern analysis
    - Real-time risk scoring
    """)
    st.image("fraud_detection_flow.png", caption="System Architecture")

# Add some styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stAlert {
        padding: 20px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)