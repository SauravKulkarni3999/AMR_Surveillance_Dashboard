def plot_top_5_pathogens_with_yoy(df, group_label="Adult", show_plot=True, plot_change=False):
    """
    Plot top 5 pathogens by frequency over time, with optional Year-over-Year % change.

    Parameters:
        df (pd.DataFrame): Must contain 'Pathogen', 'Year', and 'Frequency Count'.
        group_label (str): Title for the plot.
        show_plot (bool): Whether to display the plot.
        plot_change (bool): If True, plots YoY % change instead of raw counts.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Step 1: Top 5 pathogens overall
    top_pathogens = (
        df.groupby('Pathogen')['Frequency Count'].sum()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

    df_filtered = df[df['Pathogen'].isin(top_pathogens)].copy()

    # Step 2: Calculate % change
    if plot_change:
        df_filtered.sort_values(by=['Pathogen', 'Year'], inplace=True)
        df_filtered['YoY % Change'] = (
            df_filtered.groupby('Pathogen')['Frequency Count']
            .pct_change() * 100
        )

    # Step 3: Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_change:
        sns.lineplot(data=df_filtered, x='Year', y='YoY % Change', hue='Pathogen', marker='o', ax=ax)
        ax.set_ylabel("Year-over-Year % Change")
        ax.set_title(f"YoY Change in Resistance Frequency (Top 5 Pathogens – {group_label})")
    else:
        sns.lineplot(data=df_filtered, x='Year', y='Frequency Count', hue='Pathogen', marker='o', ax=ax)
        ax.set_ylabel("Frequency Count")
        ax.set_title(f"Top 5 Resistant Pathogens ({group_label})")

    ax.set_xlabel("Year")
    ax.legend(title="Pathogen")
    ax.grid(True)
    plt.tight_layout()

    if show_plot:
        plt.show()
    else:
        return fig, ax

import matplotlib.pyplot as plt
import seaborn as sns

def plot_yoy_change_side_by_side(adult_df, pediatric_df):
    """
    Plots YoY % change in resistance frequency for top 5 pathogens in adults and pediatrics side-by-side.
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for i, (df, label, ax) in enumerate(zip([adult_df, pediatric_df], ['Adult', 'Pediatric'], axs)):
        # Get top 5 pathogens by total frequency
        top_pathogens = (
            df.groupby('Pathogen')['Frequency Count']
            .sum().sort_values(ascending=False).head(5).index
        )
        df_filtered = df[df['Pathogen'].isin(top_pathogens)].copy()
        df_filtered.sort_values(by=['Pathogen', 'Year'], inplace=True)

        # Calculate YoY % change
        df_filtered['YoY % Change'] = (
            df_filtered.groupby('Pathogen')['Frequency Count']
            .pct_change() * 100
        )

        # Plot
        sns.lineplot(data=df_filtered, x='Year', y='YoY % Change', hue='Pathogen', marker='o', ax=ax)
        ax.set_title(f"YoY % Change in Resistance (Top 5 Pathogens – {label})")
        ax.set_xlabel("Year")
        ax.grid(True)
        if i == 0:
            ax.set_ylabel("Year-over-Year % Change")
        else:
            ax.set_ylabel("")
        ax.legend(title="Pathogen")

    plt.tight_layout()
    plt.show()
    return fig


import matplotlib.pyplot as plt

def plot_cumulative_and_share_side_by_side(adult_df, pediatric_df):
    """
    Creates a 2x2 grid of plots:
    - Top row: Cumulative resistance burden (adult & pediatric)
    - Bottom row: Pathogen resistance share over time (adult & pediatric)
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    for i, (df, label) in enumerate(zip([adult_df, pediatric_df], ['Adult', 'Pediatric'])):
        # Step 1: Top 5 pathogens overall
        top_pathogens = df.groupby('Pathogen')['Frequency Count'].sum().nlargest(5).index
        df_filtered = df[df['Pathogen'].isin(top_pathogens)]

        # --- Top Row: Cumulative Burden ---
        cumulative = df_filtered.groupby('Pathogen')['Frequency Count'].sum().sort_values(ascending=False)
        axs[0, i].bar(cumulative.index, cumulative.values, color='skyblue')
        axs[0, i].set_title(f"Cumulative Resistance Burden – {label}")
        axs[0, i].set_ylabel("Total Frequency (2018–2021)")
        axs[0, i].set_xlabel("Pathogen")
        axs[0, i].grid(axis='y')

        # --- Bottom Row: Percent Share Area Plot ---
        area_df = df_filtered.groupby(['Year', 'Pathogen'])['Frequency Count'].sum().reset_index()
        area_df['Total'] = area_df.groupby('Year')['Frequency Count'].transform('sum')
        area_df['Percent Share'] = area_df['Frequency Count'] / area_df['Total'] * 100
        area_pivot = area_df.pivot(index='Year', columns='Pathogen', values='Percent Share').fillna(0)

        area_pivot.plot.area(ax=axs[1, i], stacked=True, cmap='tab10')
        axs[1, i].set_title(f"Pathogen Resistance Share Over Time – {label}")
        axs[1, i].set_ylabel("Percentage of Total Resistance")
        axs[1, i].set_xlabel("Year")
        axs[1, i].legend(title="Pathogen", bbox_to_anchor=(1.05, 1), loc='upper left')
        axs[1, i].grid(True)

    plt.tight_layout()
    plt.show()
    return fig

import matplotlib.pyplot as plt
import pandas as pd

def plot_pathogen_volatility(df, group_label="Adult", show_plot=True):
    """
    Calculates and plots standard deviation of YoY % change in resistance
    frequency to identify the most volatile pathogens.

    Parameters:
        df (pd.DataFrame): Must contain 'Pathogen', 'Year', and 'Frequency Count'.
        group_label (str): 'Adult' or 'Pediatric' (used for titles).
        show_plot (bool): If True, shows the plot. If False, returns fig for Streamlit.

    Returns:
        fig if show_plot is False.
    """
    # Step 1: Calculate YoY % change for each pathogen
    df_sorted = df.sort_values(by=['Pathogen', 'Year']).copy()
    df_sorted['YoY % Change'] = df_sorted.groupby('Pathogen')['Frequency Count'].pct_change() * 100

    # Step 2: Compute standard deviation of YoY % Change
    volatility_scores = (
        df_sorted.groupby('Pathogen')['YoY % Change']
        .std()
        .dropna()
        .sort_values(ascending=False)
        .head(5)
    )

    # Step 3: Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(volatility_scores.index, volatility_scores.values, color='salmon')
    ax.set_title(f"Top 5 Most Volatile Pathogens ({group_label})")
    ax.set_ylabel("Std. Dev. of YoY % Change")
    ax.set_xlabel("Pathogen")
    ax.grid(axis='y')

    plt.tight_layout()

    if show_plot:
        plt.show()
    else:
        return fig


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr

def plot_amu_vs_resistance(amu_df, resistance_df):
    """
    Plots the correlation between Cephalosporin usage and Klebsiella pneumoniae resistance.

    Parameters:
        amu_df (pd.DataFrame): Must contain 'Year', 'ATC_Level_4', 'DDD per 1000 inhabitants per day'
        resistance_df (pd.DataFrame): Must contain 'Year', 'Pathogen', 'Frequency Count'

    Returns:
        fig (matplotlib.figure.Figure): Scatterplot with regression line and correlation annotation.
    """

    # Step 1: Filter AMU for cephalosporins (from ATC Level 4 classification)
    cephalosporin_df = amu_df[amu_df['ATC4Name'].str.contains("cephalo", case=False)]
    amu_grouped = cephalosporin_df.groupby('Year')['DDD'].sum().reset_index()
    amu_grouped.columns = ['Year', 'Cephalosporin_Usage']

    # Step 2: Filter Resistance data for K. pneumoniae
    kpn_df = resistance_df[resistance_df['Pathogen'].str.lower().str.contains("klebsiella pneumoniae")]
    res_grouped = kpn_df.groupby('Year')['Frequency Count'].sum().reset_index()
    res_grouped.columns = ['Year', 'KPN_Resistance']

    # Step 3: Merge datasets
    merged_df = pd.merge(amu_grouped, res_grouped, on='Year', how='inner')

    # Step 4: Calculate correlation
    r, _ = pearsonr(merged_df['Cephalosporin_Usage'], merged_df['KPN_Resistance'])

    # Step 5: Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=merged_df, x='Cephalosporin_Usage', y='KPN_Resistance', ax=ax, scatter_kws={'s': 60})
    ax.set_title("Cephalosporin Usage vs Klebsiella pneumoniae Resistance")
    ax.set_xlabel("Cephalosporin Usage (DDD per 1000 inhabitants/day)")
    ax.set_ylabel("K. pneumoniae Resistance Frequency")
    ax.grid(True)
    ax.annotate(f"Pearson r = {r:.2f}", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12)

    plt.tight_layout()
    return fig


import pandas as pd
import matplotlib.pyplot as plt

def plot_amu_resistance_timeseries(amu_df, resistance_df):
    """
    Creates a dual-axis time series plot of Cephalosporin usage and 
    Klebsiella pneumoniae resistance.

    Parameters:
        amu_df (pd.DataFrame): Antimicrobial usage data with 'ATC code', 'Year', and 'DDD per 1000 inhabitants per day'.
        resistance_df (pd.DataFrame): Resistance data with 'Pathogen', 'Year', and 'Frequency Count'.

    Returns:
        matplotlib.figure.Figure: The plot figure.
    """

    # Step 1: Filter AMU data for cephalosporins
    ceph_df = amu_df[amu_df['ATC4Name'].str.contains('J01')]
    amu_grouped = ceph_df.groupby('Year')['DDD'].sum().reset_index()
    amu_grouped.columns = ['Year', 'Cephalosporin_Usage']

    # Step 2: Filter resistance data for K. pneumoniae
    kpn_df = resistance_df[resistance_df['Pathogen'].str.lower().str.contains('klebsiella pneumoniae')]
    res_grouped = kpn_df.groupby('Year')['Frequency Count'].sum().reset_index()
    res_grouped.columns = ['Year', 'KPN_Resistance']

    # Step 3: Merge both on Year
    merged_df = pd.merge(amu_grouped, res_grouped, on='Year', how='inner')

    # Step 4: Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(merged_df['Year'], merged_df['Cephalosporin_Usage'], color='blue', marker='o', label='Cephalosporin Usage')
    ax2.plot(merged_df['Year'], merged_df['KPN_Resistance'], color='red', marker='s', label='KPN Resistance')

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Cephalosporin Usage (DDD/1000/day)', color='blue')
    ax2.set_ylabel('K. pneumoniae Resistance Frequency', color='red')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    ax1.set_title("Cephalosporin Usage vs K. pneumoniae Resistance Over Time")
    ax1.grid(True)

    fig.tight_layout()
    return fig


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def plot_custom_amu_vs_resistance(amu_df, resistance_df, atc_code, pathogen_name):
    """
    Plots the correlation between a selected antibiotic (by ATC code)
    and resistance frequency of a selected pathogen over time.

    Parameters:
        amu_df (pd.DataFrame): Must contain 'ATC code', 'Year', 'DDD per 1000 inhabitants per day'
        resistance_df (pd.DataFrame): Must contain 'Pathogen', 'Year', 'Frequency Count'
        atc_code (str): Selected ATC code (e.g., 'J01DD')
        pathogen_name (str): Selected pathogen (e.g., 'Escherichia coli')

    Returns:
        matplotlib.figure.Figure: The scatterplot with correlation annotation.
    """
    # Step 1: Filter AMU data
    amu_filtered = amu_df[amu_df['ATC4Name'] == atc_code]
    amu_grouped = amu_filtered.groupby('Year')['DDD'].sum().reset_index()
    amu_grouped.columns = ['Year', 'AMU_Value']

    # Step 2: Filter resistance data
    res_filtered = resistance_df[resistance_df['Pathogen'].str.lower() == pathogen_name.lower()]
    res_grouped = res_filtered.groupby('Year')['Frequency Count'].sum().reset_index()
    res_grouped.columns = ['Year', 'Resistance_Value']

    # Step 3: Merge
    merged_df = pd.merge(amu_grouped, res_grouped, on='Year', how='inner')

    if merged_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data available for this drug-pathogen pair.', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.axis('off')
        return fig

    # Step 4: Calculate correlation
    r, _ = pearsonr(merged_df['AMU_Value'], merged_df['Resistance_Value'])

    # Step 5: Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=merged_df, x='AMU_Value', y='Resistance_Value', ax=ax, scatter_kws={'s': 60})
    ax.set_title(f"{atc_code} Usage vs {pathogen_name} Resistance")
    ax.set_xlabel("Antibiotic Usage (DDD/1000 inhabitants/day)")
    ax.set_ylabel("Resistance Frequency")
    ax.annotate(f"Pearson r = {r:.2f}", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12)
    ax.grid(True)

    plt.tight_layout()
    return fig


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_mechanism_clusters(df_cluster):
    """
    Visualizes clustering of AMR genes using PCA-reduced features.
    Parameters:
        df_cluster (pd.DataFrame): Preprocessed data with columns:
            ['PCA1', 'PCA2', 'Cluster', 'Resistance Mechanism', 'Drug Class']
    Returns:
        Matplotlib figure object.
    """

    # Determine dominant mechanism and drug class per cluster
    cluster_labels = df_cluster.groupby('Cluster')[['Resistance Mechanism', 'Drug Class']] \
        .agg(lambda x: x.value_counts().index[0]).reset_index()

    # Merge cluster-level labels
    df_cluster = df_cluster.merge(cluster_labels, on='Cluster', suffixes=('', '_Label'))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df_cluster,
        x='PCA1', y='PCA2',
        hue='Cluster',
        style='Resistance Mechanism_Label',
        palette='tab10',
        alpha=0.8,
        s=80
    )

    for _, row in cluster_labels.iterrows():
        subset = df_cluster[df_cluster['Cluster'] == row['Cluster']]
        cx = subset['PCA1'].mean()
        cy = subset['PCA2'].mean()
        label = f"{row['Resistance Mechanism']} ({row['Drug Class']})"
        ax.text(cx, cy, label, fontsize=9, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

    ax.set_title("Mechanism-Based Clustering of AMR Genes")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.grid(True)
    ax.legend(title='Cluster')
    plt.tight_layout()
    return fig


import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(df):
    """
    Plots top feature importances from a Random Forest model.

    Parameters:
        df (pd.DataFrame): Must contain columns ['Feature', 'Importance']

    Returns:
        matplotlib.figure.Figure
    """
    # Sort and get top 15 features
    top_features = df.sort_values(by='Importance', ascending=False).head(15)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_features, x='Importance', y='Feature', palette='viridis', ax=ax)

    ax.set_title("Top Feature Importances (Random Forest)")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    ax.grid(True)

    plt.tight_layout()
    return fig


