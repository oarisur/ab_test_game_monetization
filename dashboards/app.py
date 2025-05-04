import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- 1. Data Loading ---
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("data/simulated_player_data.csv")
        return data
    except FileNotFoundError:
        st.error("Error: The file 'data/simulated_player_data.csv' was not found. Please ensure it exists in the correct directory.")
        return None

# --- 2. Key Metrics Calculation ---
def calculate_key_metrics(df):
    metrics = df.groupby('group').agg(
        ARPU=('revenue', 'mean'),
        Median_Revenue=('revenue', 'median'),
        Revenue_StdDev=('revenue', 'std'),
        NFT_Purchase_Rate=('nft_purchased', 'mean'),
        Mean_XP=('xp', 'mean'),
        Median_XP=('xp', 'median'),
        Mean_Crashes=('crashes', 'mean'),
        Mean_Stunts=('stunts', 'mean'),
        Mean_Tokens_Spent=('tokens_spent', 'mean'),
        User_Count=('user_id', 'count')).reset_index()
    metrics['NFT_Purchase_Rate'] *= 100
    return metrics.set_index('group')

# --- 3. Visualizations ---
def plot_revenue_distribution(df, selected_groups):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df[df['group'].isin(selected_groups)], x='group', y='revenue', palette='Set2', ax=ax)
    ax.set_title("Revenue Distribution by Group")
    ax.set_xlabel("Group")
    ax.set_ylabel("Revenue")
    return fig

def plot_revenue_kde(df, selected_groups):
    fig, ax = plt.subplots(figsize=(10, 5))
    if 'A' in selected_groups:
        sns.kdeplot(df[df.group == 'A']['revenue'], label='Group A', fill=True, ax=ax)
    if 'B' in selected_groups:
        sns.kdeplot(df[df.group == 'B']['revenue'], label='Group B', fill=True, ax=ax)
    ax.set_title("Revenue KDE Plot")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("Density")
    ax.legend()
    return fig

def plot_token_usage(df, selected_groups):
    fig, ax = plt.subplots(figsize=(10, 5))
    if 'A' in selected_groups:
        sns.kdeplot(df[df.group == 'A']['tokens_spent'], label='Tokens A', fill=True, ax=ax)
    if 'B' in selected_groups:
        sns.kdeplot(df[df.group == 'B']['tokens_spent'], label='Tokens B', fill=True, ax=ax)
    ax.set_title("Token Spending Distribution")
    ax.set_xlabel("Tokens Spent")
    ax.set_ylabel("Density")
    ax.legend()
    return fig

def plot_nft_purchase_rate(metrics, selected_groups):
    metrics_filtered = metrics.reset_index()
    metrics_filtered = metrics_filtered[metrics_filtered['group'].isin(selected_groups)]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='group', y='NFT_Purchase_Rate', data=metrics_filtered, palette='Set2', ax=ax)
    ax.set_title("NFT Purchase Rate by Group")
    ax.set_ylabel("Purchase Rate (%)")
    ax.set_xlabel("Group")
    ax.set_ylim(0, metrics['NFT_Purchase_Rate'].max() * 1.1)
    return fig

def plot_xp_distribution(df, selected_groups):
    fig, ax = plt.subplots(figsize=(10, 5))
    for group in selected_groups:
        sns.histplot(df[df.group == group], x='xp', kde=True, element='step', label=f'Group {group}', ax=ax)
    ax.set_title("XP Distribution by Group")
    ax.set_xlabel("XP")
    ax.set_ylabel("Frequency")
    ax.legend()
    return fig

def plot_event_impact(df, selected_groups):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for group in selected_groups:
        sns.scatterplot(data=df[df.group == group], x='crashes', y='revenue', label=f'Group {group}', alpha=0.6, ax=axes[0])
        sns.scatterplot(data=df[df.group == group], x='stunts', y='revenue', label=f'Group {group}', alpha=0.6, ax=axes[1])
    axes[0].set_title("Crash Count vs Revenue")
    axes[0].set_xlabel("Crash Count")
    axes[0].set_ylabel("Revenue")
    axes[1].set_title("Stunts vs Revenue")
    axes[1].set_xlabel("Stunts")
    axes[1].set_ylabel("Revenue")
    axes[0].legend()
    axes[1].legend()
    return fig

# --- 4. Frequentist A/B Test ---
def perform_frequentist_test(df, alpha):
    group_A_revenue = df[df.group == 'A']['revenue']
    group_B_revenue = df[df.group == 'B']['revenue']
    if len(group_A_revenue) < 2 or len(group_B_revenue) < 2:
        return None, None
    t_stat, p_val = stats.ttest_ind(group_B_revenue, group_A_revenue, equal_var=False)
    return t_stat, p_val

# --- 5. Bayesian Estimation ---
def perform_bayesian_estimation(df, n_samples=10000):
    rev_A = df[df.group == 'A']['revenue']
    rev_B = df[df.group == 'B']['revenue']
    if len(rev_A) == 0 or len(rev_B) == 0:
        return None, None
    posterior_A = np.random.normal(rev_A.mean(), rev_A.std() / np.sqrt(len(rev_A)), n_samples)
    posterior_B = np.random.normal(rev_B.mean(), rev_B.std() / np.sqrt(len(rev_B)), n_samples)
    prob_B_superior = (posterior_B > posterior_A).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(posterior_A, bins=50, alpha=0.5, label='Posterior A')
    ax.hist(posterior_B, bins=50, alpha=0.5, label='Posterior B')
    ax.axvline(np.mean(posterior_A), color='blue', linestyle='--', label=f'Mean A: {np.mean(posterior_A):.2f}')
    ax.axvline(np.mean(posterior_B), color='orange', linestyle='--', label=f'Mean B: {np.mean(posterior_B):.2f}')
    ax.set_title("Posterior Distributions of Mean Revenue")
    ax.set_xlabel("Mean Revenue")
    ax.set_ylabel("Frequency")
    ax.legend()

    return prob_B_superior, fig

# --- 6. Summary & Actionable Insights ---
def display_insights(df, p_val, nft_purchase_rate_diff, alpha):
    group_A_revenue_mean = df[df.group == 'A']['revenue'].mean()
    group_B_revenue_mean = df[df.group == 'B']['revenue'].mean()
    lift = (group_B_revenue_mean - group_A_revenue_mean) / group_A_revenue_mean

    st.subheader("ARPU Analysis")
    col1, col2 = st.columns(2)
    col1.metric(label="ARPU (Group A)", value=f"{group_A_revenue_mean:.2f}")
    col2.metric(label="ARPU (Group B)", value=f"{group_B_revenue_mean:.2f}", delta=f"{lift:.2%}")

    st.subheader(f"Frequentist A/B Test Result (Significance Level: {alpha})")
    if p_val is not None:
        st.write(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.6f}")
        if p_val < alpha:
            st.success(f"Statistically significant improvement in revenue for Group B at alpha = {alpha}.")
            st.info("Recommendation: Rollout the new feature (Group B).")
        else:
            st.warning(f"No statistically significant difference in revenue found at alpha = {alpha}.")
            st.info("Recommendation: Consider further testing or revert to the original version.")
    else:
        st.warning("Not enough data to perform Frequentist A/B Test.")

    st.subheader("Blockchain Engagement Insight")
    if nft_purchase_rate_diff is not None:
        if nft_purchase_rate_diff > 0:
            st.success(f"NFT Purchase Rate in Group B is {nft_purchase_rate_diff:.2f}% higher than Group A.")
            st.info("Recommendation: Consider expanding Web3 features based on this positive signal.")
        else:
            st.warning(f"NFT Purchase Rate in Group B is not higher than Group A (difference: {nft_purchase_rate_diff:.2f}%).")
            st.info("Recommendation: Investigate UX or education gaps for blockchain features.")
    else:
        st.warning("NFT Purchase Rate data is unavailable.")

# --- Main Streamlit App ---
st.title("Blockchain Game A/B Test Analysis")
st.markdown("""
### Project Overview and Business Goal
In this analysis, we will examine the impact of a new blockchain-based game feature on player behavior and monetization. Our goal is to analyze how the feature affects revenue, 
engagement with blockchain features (such as NFT purchases), and in-game progression.

### Causal Inference Assumptions
To ensure the validity of the A/B test, we assume that the random assignment of players to either Group A (control) or Group B (test) is unbiased and that no confounding variables 
affect the results. In practice, we would check for randomization balance by comparing baseline characteristics of both groups, and apply statistical controls if necessary. Any 
significant differences between groups at baseline should be addressed through further validation steps.
""")

# Load the data
data = load_data()

# Proceed only if the data was loaded successfully
if data is not None:
    # Sidebar for user controls
    st.sidebar.header("Analysis Controls")
    selected_groups_viz = st.sidebar.multiselect("Select Groups for Plots", ['A', 'B'], default=['A', 'B'])
    n_bayesian_samples = st.sidebar.slider("Bayesian Samples", min_value=1000, max_value=100000, value=10000, step=1000)
    alpha_level = st.sidebar.slider("Significance Level (Alpha)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
    display_metrics = st.sidebar.multiselect("Select Metrics to Display", ['ARPU', 'Median_Revenue', 'Revenue_StdDev', 'NFT_Purchase_Rate', 'Mean_XP', 'Median_XP', 'Mean_Crashes', 
    'Mean_Stunts', 'Mean_Tokens_Spent', 'User_Count'], default=['User_Count', 'Mean_XP', 'ARPU', 'NFT_Purchase_Rate'])

    st.sidebar.subheader("Data Filtering (Optional)")
    filter_column = st.sidebar.selectbox("Filter by Column", data.columns.tolist() + [None])
    filter_value = None
    if filter_column:
        unique_values = data[filter_column].unique().tolist()
        filter_value = st.sidebar.multiselect(f"Select values for {filter_column}", unique_values, default=unique_values)
        data_filtered = data[data[filter_column].isin(filter_value)]
    else:
        data_filtered = data

    if data_filtered.empty:
        st.warning("No data matching the filter criteria.")
    else:
        metrics = calculate_key_metrics(data_filtered)

        st.subheader("Key Performance Indicators")
        cols = st.columns(len(display_metrics))
        for i, metric_name in enumerate(display_metrics):
            if metric_name in metrics.columns:
                cols[i].metric(label=f"{metric_name} (A)", value=f"{metrics.loc['A', metric_name]:.2f}" if pd.api.types.is_float_dtype(metrics[metric_name]) else metrics.loc['A', metric_name])
                cols[i].metric(label=f"{metric_name} (B)", value=f"{metrics.loc['B', metric_name]:.2f}" if pd.api.types.is_float_dtype(metrics[metric_name]) else metrics.loc['B', metric_name])

        st.subheader("Data Visualizations")

        st.pyplot(plot_revenue_distribution(data_filtered, selected_groups_viz))
        st.pyplot(plot_revenue_kde(data_filtered, selected_groups_viz))
        st.pyplot(plot_token_usage(data_filtered, selected_groups_viz))
        st.pyplot(plot_nft_purchase_rate(metrics, selected_groups_viz))
        st.pyplot(plot_xp_distribution(data_filtered, selected_groups_viz))
        st.pyplot(plot_event_impact(data_filtered, selected_groups_viz))

        st.subheader("Frequentist A/B Test: Revenue")
        t_stat, p_val = perform_frequentist_test(data_filtered, alpha_level)
        if p_val is not None:
            st.write(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.6f}")
        else:
            st.warning("Not enough data in selected groups to perform Frequentist A/B Test.")

        st.subheader("Bayesian Estimation: Revenue")
        prob_superior, bayesian_plot = perform_bayesian_estimation(data_filtered, n_bayesian_samples)
        if bayesian_plot is not None:
            st.pyplot(bayesian_plot)
            st.write(f"Probability that Group B has higher mean revenue than Group A: {prob_superior:.3f}")
        else:
            st.warning("Not enough data in selected groups to perform Bayesian Estimation.")

        st.subheader("Summary and Actionable Insights")
        if 'NFT_Purchase_Rate' in metrics.columns:
            nft_purchase_rate_diff = metrics.loc['B', 'NFT_Purchase_Rate'] - metrics.loc['A', 'NFT_Purchase_Rate']
        else:
            nft_purchase_rate_diff = None
        display_insights(data_filtered, p_val, nft_purchase_rate_diff, alpha_level)

st.markdown("""
### Scalability Considerations
* As the dataset grows, optimizing data processing and visualization performance will be essential. We can improve scalability by using pagination for visualizations or caching 
expensive computations, especially in large datasets. 
- Additionally, we could leverage Apache Spark for more complex simulations.
""")

st.markdown("""
### Further Analysis Options
- **Segmentation Analysis:** Explore how the new feature impacts different player segments (e.g., by spending habits, play frequency).
- **Long-Term Effects:** Analyze data over a longer period to understand the sustained impact of the feature.
- **Cost-Benefit Analysis:** Evaluate the development and maintenance costs of the new feature against the observed benefits.
""")