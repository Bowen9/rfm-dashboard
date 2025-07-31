"""
rfm_dashboard_streamlit.py
==========================

This Streamlit application provides an interactive dashboard for
customer segmentation using Recency‑Frequency‑Monetary (RFM)
analysis.  It allows you to upload customer profile and
transaction CSV files, select the number of clusters for KMeans
segmentation, view summary statistics for each cluster and
visualise the distributions of transaction frequency and
monetary value.  Users can filter the enriched dataset by
cluster and download the results as a CSV for further analysis.

Usage
-----
To run this dashboard locally you will need to install
Streamlit (e.g. `pip install streamlit`) in your Python
environment.  Then run:

```
streamlit run rfm_dashboard_streamlit.py
```

Upon launch, the app will prompt for the two CSV files and
provide controls for cluster count and cluster filtering.
"""

import io
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def parse_dates(customer_df: pd.DataFrame, transactions_df: pd.DataFrame) -> None:
    """Parse REGISTERED_ON, DOB and APPROVAL_DATETIME columns in place."""
    # Parse customer registration datetime
    if 'REGISTERED_ON' in customer_df.columns:
        try:
            customer_df['REGISTERED_ON'] = pd.to_datetime(
                customer_df['REGISTERED_ON'],
                format='%d-%b-%y %I.%M.%S.%f %p',
                errors='coerce'
            )
        except Exception:
            customer_df['REGISTERED_ON'] = pd.to_datetime(
                customer_df['REGISTERED_ON'], errors='coerce'
            )
    # Parse DOB
    if 'DOB' in customer_df.columns:
        try:
            customer_df['DOB'] = pd.to_datetime(
                customer_df['DOB'], format='%m-%d-%Y', errors='coerce'
            )
        except Exception:
            customer_df['DOB'] = pd.to_datetime(customer_df['DOB'], errors='coerce')
    # Parse transaction approval datetime
    if 'APPROVAL_DATETIME' in transactions_df.columns:
        transactions_df['APPROVAL_DATETIME'] = pd.to_datetime(
            transactions_df['APPROVAL_DATETIME'], errors='coerce'
        )


def compute_age(customer_df: pd.DataFrame, reference_date: str = '2025-06-16') -> None:
    """Compute an AGE column based on the DOB column."""
    ref = pd.Timestamp(reference_date)
    customer_df['AGE'] = (ref - customer_df['DOB']).dt.days // 365


def compute_rfm(transactions_df: pd.DataFrame, reference_date: str = '2025-06-16') -> pd.DataFrame:
    """Compute RFM metrics per account."""
    ref = pd.Timestamp(reference_date)
    rfm = transactions_df.groupby('ACCOUNT_NUMBER').agg(
        RECENCY=('APPROVAL_DATETIME', lambda x: (ref - x.max()).days),
        FREQUENCY=('TXNID', 'count'),
        MONETARY=('TXN_AMOUNT', 'sum')
    ).reset_index()
    return rfm


def compute_additional_features(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Compute behavioural features beyond basic RFM for each account.

    Features include counts of 'Money In' and 'Money Out' transactions,
    the ratio of money-in to total money-in + money-out, and channel
    mix ratios for each channel (e.g. APP, Third Party, USSD, WEB).

    Parameters
    ----------
    transactions_df : DataFrame
        Transactions data with ACCOUNT_NUMBER, TRANSACTION_NAME,
        CHANAL, and TXNID columns.

    Returns
    -------
    DataFrame
        A DataFrame indexed by ACCOUNT_NUMBER containing the
        additional behavioural features.
    """
    # Count Money In and Money Out transactions
    money_in = transactions_df.groupby('ACCOUNT_NUMBER')['TRANSACTION_NAME'].apply(lambda x: (x == 'Money In').sum())
    money_in = money_in.rename('money_in_count')
    money_out = transactions_df.groupby('ACCOUNT_NUMBER')['TRANSACTION_NAME'].apply(lambda x: (x == 'Money Out').sum())
    money_out = money_out.rename('money_out_count')
    total_basic = money_in + money_out
    money_in_ratio = money_in.div(total_basic.replace(0, np.nan))
    money_in_ratio = money_in_ratio.rename('money_in_ratio')
    # Channel mix counts and ratios
    channel_counts = transactions_df.pivot_table(index='ACCOUNT_NUMBER', columns='CHANAL', values='TXNID', aggfunc='count', fill_value=0)
    channel_totals = channel_counts.sum(axis=1)
    channel_ratios = channel_counts.div(channel_totals.replace(0, np.nan), axis=0)
    channel_ratios = channel_ratios.add_suffix('_ratio')
    # Combine features
    features = pd.concat([money_in, money_out, money_in_ratio, channel_ratios], axis=1)
    return features.reset_index()


def cluster_customers(rfm: pd.DataFrame, n_clusters: int, feature_cols: list[str]) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """Standardise selected feature columns and apply KMeans clustering.

    Parameters
    ----------
    rfm : DataFrame
        DataFrame containing the feature columns to be used for clustering.
    n_clusters : int
        The number of clusters to form.
    feature_cols : list of str
        Names of columns in rfm to be used for clustering.

    Returns
    -------
    rfm : DataFrame
        The input DataFrame with an added CLUSTER column.
    model : KMeans
        The fitted KMeans model.
    scaler : StandardScaler
        The fitted scaler used for feature standardisation.
    """
    features = rfm[feature_cols].fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['CLUSTER'] = model.fit_predict(scaled)
    return rfm, model, scaler


def compute_cluster_quality(scaled_features: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Compute clustering quality metrics for the given labels.

    Returns the silhouette score, Davies–Bouldin index (lower is better)
    and Calinski–Harabasz index (higher is better).  If there are
    fewer than two clusters or insufficient samples, returns NaNs.

    Parameters
    ----------
    scaled_features : array-like
        Standardised feature matrix used for clustering.
    labels : array-like
        Cluster labels assigned to each sample.

    Returns
    -------
    tuple of floats
        (silhouette_score, davies_bouldin_index, calinski_harabasz_index)
    """
    labels = np.asarray(labels).ravel()
    unique_clusters = np.unique(labels)
    if len(unique_clusters) < 2 or scaled_features.shape[0] < 2:
        return float('nan'), float('nan'), float('nan')
    # Compute Silhouette score
    try:
        sil = silhouette_score(scaled_features, labels)
    except Exception:
        sil = float('nan')
    # Compute Davies–Bouldin index
    try:
        dbi = davies_bouldin_score(scaled_features, labels)
    except Exception:
        dbi = float('nan')
    # Compute Calinski–Harabasz index
    try:
        ch = calinski_harabasz_score(scaled_features, labels)
    except Exception:
        ch = float('nan')
    return sil, dbi, ch


def merge_profiles(customer_df: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    """Merge RFM and cluster labels back to customer data."""
    df = customer_df.copy()
    df['ACCOUNT_NUMBER'] = df['ACCOUNT_NUMBER'].astype(str)
    rfm['ACCOUNT_NUMBER'] = rfm['ACCOUNT_NUMBER'].astype(str)
    return df.merge(rfm, on='ACCOUNT_NUMBER', how='left')


def compute_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise count, average age, pct female, avg frequency and monetary for each cluster."""
    # Additional metrics: avg_txn_value (MONETARY / FREQUENCY) and money_in_ratio if present
    def pct_fem(s):
        return (s.str.lower() == 'female').mean() if not s.isnull().all() else np.nan
    summary = df.groupby('CLUSTER').agg(
        count=('ACCOUNT_NUMBER', 'count'),
        avg_age=('AGE', 'mean'),
        pct_female=('SEX', pct_fem),
        avg_frequency=('FREQUENCY', 'mean'),
        avg_monetary=('MONETARY', 'mean'),
        avg_txn_value=('avg_txn_value', 'mean'),
        money_in_ratio=('money_in_ratio', 'mean')
    ).round(2)
    return summary


def main() -> None:
    st.set_page_config(page_title='Customer Segmentation Dashboard', layout='wide')
    st.title('Customer Segmentation Dashboard (RFM + KMeans)')

    st.markdown(
        "Upload your **Customer Profile** and **Transactions** CSV files below. "
        "After loading, select the number of clusters and explore the resulting customer segments."
    )

    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        customer_file = st.file_uploader('Customer Profile CSV', type=['csv'])
    with col2:
        transactions_file = st.file_uploader('Transactions CSV', type=['csv'])

    # Choose number of clusters
    n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=5, step=1)
    # Toggle for behavioural features
    use_extra = st.sidebar.checkbox('Use behavioural features (advanced)', value=False)

    if customer_file is not None and transactions_file is not None:
        # Read uploaded files into DataFrames
        customer_df = pd.read_csv(customer_file)
        transactions_df = pd.read_csv(transactions_file)
        # Drop empty Date column if present
        if 'Date' in transactions_df.columns:
            transactions_df = transactions_df.drop(columns=['Date'])
        # Parse dates
        parse_dates(customer_df, transactions_df)
        # Compute age
        compute_age(customer_df)
        # Compute RFM metrics
        rfm = compute_rfm(transactions_df)
        # Compute additional behavioural features and merge
        extra = compute_additional_features(transactions_df)
        rfm = rfm.merge(extra, on='ACCOUNT_NUMBER', how='left')
        # Compute average transaction value per account
        rfm['avg_txn_value'] = rfm['MONETARY'] / rfm['FREQUENCY'].replace(0, np.nan)
        # Determine feature columns for clustering
        base_cols = ['RECENCY', 'FREQUENCY', 'MONETARY']
        feature_cols = base_cols.copy()
        if use_extra:
            # Use all numeric columns except identifier and cluster
            extra_cols = [c for c in rfm.columns if c not in ['ACCOUNT_NUMBER'] + base_cols + ['CLUSTER']]
            feature_cols.extend(extra_cols)
        # Cluster customers using selected features
        rfm, model, scaler = cluster_customers(rfm, n_clusters=n_clusters, feature_cols=feature_cols)
        # Compute cluster quality metrics on scaled features
        try:
            scaled_features = scaler.transform(rfm[feature_cols].fillna(0))
            sil_score, dbi_score, ch_score = compute_cluster_quality(scaled_features, rfm['CLUSTER'])
        except Exception:
            sil_score, dbi_score, ch_score = float('nan'), float('nan'), float('nan')
        # Merge results back to customers
        enriched = merge_profiles(customer_df, rfm)
        # Cluster summary
        summary_df = compute_cluster_summary(enriched)

        # Sidebar cluster filter
        clusters = sorted(enriched['CLUSTER'].dropna().unique().tolist())
        selected_clusters = st.sidebar.multiselect('Filter Clusters', options=clusters, default=clusters)

        # Age filter (if AGE column exists)
        if 'AGE' in enriched.columns and not enriched['AGE'].dropna().empty:
            age_min = int(enriched['AGE'].dropna().min())
            age_max = int(enriched['AGE'].dropna().max())
            age_range = st.sidebar.slider('Age Range', min_value=age_min, max_value=age_max, value=(age_min, age_max))
        else:
            age_range = (0, 120)

        # Gender filter
        genders = sorted(enriched['SEX'].dropna().str.title().unique().tolist())
        selected_genders = st.sidebar.multiselect('Filter Gender', options=genders, default=genders)

        # Account type filter
        acct_types = sorted(enriched['ACCOUNT_TYPE'].dropna().unique().tolist())
        selected_acct_types = st.sidebar.multiselect('Filter Account Type', options=acct_types, default=acct_types)

        # Apply filters
        filtered = enriched[
            (enriched['CLUSTER'].isin(selected_clusters)) &
            (enriched['AGE'].between(age_range[0], age_range[1], inclusive='both')) &
            (enriched['SEX'].str.title().isin(selected_genders)) &
            (enriched['ACCOUNT_TYPE'].isin(selected_acct_types))
        ]

        # Display cluster quality metrics in sidebar
        st.sidebar.markdown('---')
        st.sidebar.write('**Cluster Quality Metrics**')
        st.sidebar.metric('Silhouette Score', f"{sil_score:.3f}" if sil_score == sil_score else 'N/A')
        st.sidebar.metric('Davies–Bouldin Index', f"{dbi_score:.3f}" if dbi_score == dbi_score else 'N/A')
        st.sidebar.metric('Calinski–Harabasz Index', f"{ch_score:.3f}" if ch_score == ch_score else 'N/A (higher is better)')

        # Display summary
        st.subheader('Cluster Profile Summary')
        st.dataframe(summary_df)

        # Display filtered customer table (limited rows)
        st.subheader('Enriched Customer Data (filtered)')
        st.dataframe(filtered.head(100))

        # Plot distributions
        st.subheader('Distribution Plots')
        col_freq, col_mon = st.columns(2)
        with col_freq:
            st.write('**Transaction Frequency (log scale)**')
            freq_hist_fig = io.BytesIO()
            import matplotlib.pyplot as plt
            plt.figure(figsize=(4,3))
            plt.hist(filtered['FREQUENCY'].dropna(), bins=50)
            plt.xscale('log')
            plt.xlabel('Frequency (log scale)')
            plt.ylabel('Number of Customers')
            plt.tight_layout()
            plt.savefig(freq_hist_fig, format='png')
            plt.close()
            st.image(freq_hist_fig)
        with col_mon:
            st.write('**Monetary Value (log scale)**')
            mon_hist_fig = io.BytesIO()
            plt.figure(figsize=(4,3))
            plt.hist(filtered['MONETARY'].dropna(), bins=50)
            plt.xscale('log')
            plt.xlabel('Monetary (log scale)')
            plt.ylabel('Number of Customers')
            plt.tight_layout()
            plt.savefig(mon_hist_fig, format='png')
            plt.close()
            st.image(mon_hist_fig)

        # Scatter plot frequency vs monetary by cluster
        st.subheader('Frequency vs Monetary Scatter')
        scatter_fig = io.BytesIO()
        plt.figure(figsize=(5,4))
        for c in selected_clusters:
            sub = filtered[filtered['CLUSTER'] == c]
            plt.scatter(
                sub['FREQUENCY'], sub['MONETARY'],
                label=f'Cluster {c}', s=10, alpha=0.6
            )
        plt.xlabel('Frequency')
        plt.ylabel('Monetary')
        plt.legend()
        plt.tight_layout()
        plt.savefig(scatter_fig, format='png')
        plt.close()
        st.image(scatter_fig)

        # Download enriched data
        st.subheader('Download Enriched Data')
        csv = enriched.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='Download CSV',
            data=csv,
            file_name='enriched_customer_data.csv',
            mime='text/csv'
        )

    else:
        st.info('Please upload both CSV files to begin.')


if __name__ == '__main__':
    main()