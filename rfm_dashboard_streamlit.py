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
from sklearn.metrics import silhouette_score, davies_bouldin_score


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


def cluster_customers(rfm: pd.DataFrame, n_clusters: int) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """Standardise RFM metrics and apply KMeans clustering."""
    features = rfm[['RECENCY', 'FREQUENCY', 'MONETARY']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['CLUSTER'] = model.fit_predict(scaled)
    return rfm, model, scaler


def compute_cluster_quality(scaled_features: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Compute clustering quality metrics for the given labels.

    Returns the silhouette score and Davies–Bouldin index.  If there
    are fewer than 2 clusters or not enough samples, returns (nan,
    nan).
    """
    if labels.ndim > 1:
        labels = labels.ravel()
    # At least 2 clusters and each cluster must have at least one sample
    unique_clusters = np.unique(labels)
    if len(unique_clusters) < 2 or len(scaled_features) < 2:
        return float('nan'), float('nan')
    try:
        sil = silhouette_score(scaled_features, labels)
    except Exception:
        sil = float('nan')
    try:
        dbi = davies_bouldin_score(scaled_features, labels)
    except Exception:
        dbi = float('nan')
    return sil, dbi


def merge_profiles(customer_df: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    """Merge RFM and cluster labels back to customer data."""
    df = customer_df.copy()
    df['ACCOUNT_NUMBER'] = df['ACCOUNT_NUMBER'].astype(str)
    rfm['ACCOUNT_NUMBER'] = rfm['ACCOUNT_NUMBER'].astype(str)
    return df.merge(rfm, on='ACCOUNT_NUMBER', how='left')


def compute_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise count, average age, pct female, avg frequency and monetary for each cluster."""
    summary = df.groupby('CLUSTER').agg(
        count=('ACCOUNT_NUMBER', 'count'),
        avg_age=('AGE', 'mean'),
        pct_female=('SEX', lambda x: (x.str.lower() == 'female').mean()),
        avg_frequency=('FREQUENCY', 'mean'),
        avg_monetary=('MONETARY', 'mean')
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
        # Cluster customers
        rfm, model, scaler = cluster_customers(rfm, n_clusters=n_clusters)
        # Compute cluster quality metrics on scaled RFM features
        scaled_features = scaler.transform(rfm[['RECENCY', 'FREQUENCY', 'MONETARY']])
        sil_score, dbi_score = compute_cluster_quality(scaled_features, rfm['CLUSTER'])
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