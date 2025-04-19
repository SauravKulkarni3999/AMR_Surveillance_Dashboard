import streamlit as st
import matplotlib.pyplot as plt
from resistance_visuals import (
    plot_top_5_pathogens_with_yoy,
    plot_yoy_change_side_by_side,
    plot_cumulative_and_share_side_by_side,
    plot_pathogen_volatility,
    plot_amu_vs_resistance,
    plot_amu_resistance_timeseries,
    plot_custom_amu_vs_resistance,
    plot_mechanism_clusters,
    plot_feature_importance
)
from forecasting import generate_forecast_plot

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AMR Surveillance Dashboard",
    layout="wide"
)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📊 Dashboard Navigation")
selection = st.sidebar.radio("Go to", [
    "Project Overview",
    "Resistance Trends",
    "Forecasting Resistance",
    "AMU vs Resistance",
    "Mechanism Clustering",
    "ML Feature Importance"
])

# --- PROJECT OVERVIEW PAGE ---
if selection == "Project Overview":
    st.title("🧬 AMR Surveillance Dashboard")
    st.markdown("""
    This dashboard presents a comprehensive surveillance of antimicrobial resistance (AMR) using publicly available datasets from:
    - **CDC** and **WHO GLASS** for pathogen and antibiotic usage data
    - **CARD** for genetic resistance mechanisms

    ### Goals
    - Visualize trends in resistance by pathogen and population
    - Compare antibiotic usage with resistance emergence
    - Identify key resistance mechanisms through ML and clustering
    """)

# --- RESISTANCE TRENDS PAGE ---
elif selection == "Resistance Trends":
    st.title("📈 Resistance Trends (Adult & Pediatric)")
    st.markdown("Visualize resistance frequency trends, percent changes, and proportional shifts for the most common pathogens in adult and pediatric populations.")

    try:
        from data_loader import adult_yearly_df, pediatric_yearly_df

        # Plot 1: Raw Frequency Trends
        st.subheader("Raw Frequency Trends – Top 5 Pathogens")
        fig1, _ = plot_top_5_pathogens_with_yoy(adult_yearly_df, group_label="Adult", show_plot=False)
        st.pyplot(fig1)

        fig2, _ = plot_top_5_pathogens_with_yoy(pediatric_yearly_df, group_label="Pediatric", show_plot=False)
        st.pyplot(fig2)

        # Plot 2: YoY % Change (Side-by-Side)
        st.subheader("Year-over-Year % Change")
        fig_yoy = plot_yoy_change_side_by_side(adult_yearly_df, pediatric_yearly_df)
        st.pyplot(fig_yoy)

        # Plot 3: Cumulative Burden + Share Over Time
        st.subheader("Cumulative Burden and Share of Resistance Over Time")
        fig_combo = plot_cumulative_and_share_side_by_side(adult_yearly_df, pediatric_yearly_df)
        st.pyplot(fig_combo)

        # Plot 4: Volatility in YoY Change
        st.subheader("Volatility in Pathogen Resistance Trends")
        fig_volatility_adult = plot_pathogen_volatility(adult_yearly_df, group_label="Adult", show_plot=False)
        st.pyplot(fig_volatility_adult)

        fig_volatility_ped = plot_pathogen_volatility(pediatric_yearly_df, group_label="Pediatric", show_plot=False)
        st.pyplot(fig_volatility_ped)

        st.markdown("""
        ### 🧠 Interpretation: Resistance Trends
        
        - **Top 5 Pathogens**: Highlights the most frequently reported pathogens over time in both adults and pediatrics.
        - **Year-over-Year Change**: Identifies increases or decreases in resistance, helping to flag emerging threats or progress from interventions.
        - **Cumulative & Proportional Burden**: Shows which pathogens contribute most to the resistance load over time.
        - **Volatility Analysis**: Flags unstable resistance patterns that may require attention or deeper investigation.
        """)


    except Exception as e:
        st.error(f"Error loading data or rendering plot: {e}")

# --- FORECASTING RESISTANCE PAGE ---
elif selection == "Forecasting Resistance":
    st.title("🔮 Forecasting Resistance")
    st.markdown("Use time series forecasting to project resistance trends for selected pathogens.")

    try:
        from data_loader import adult_yearly_df, pediatric_yearly_df

        population_group = st.selectbox("Select Population Group", ["Adult", "Pediatric"])
        df = adult_yearly_df if population_group == "Adult" else pediatric_yearly_df

        pathogen_options = df['Pathogen'].unique()
        selected_pathogen = st.selectbox("Select a Pathogen", sorted(pathogen_options))

        fig_forecast, fig_components, mse = generate_forecast_plot(df, selected_pathogen, population_group)
        
        st.subheader("Resistance Forecast")
        st.pyplot(fig_forecast)
        
        st.subheader("Forecast Accuracy")
        st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}")
        
        st.subheader("Forecast Components (Trend / Seasonality)")
        st.plotly_chart(fig_components)
        st.markdown("""
        ### 🧠 Interpretation: Forecasting Resistance
        
        - **Forecast Plot**: Projects future resistance frequency for selected pathogens, helping anticipate healthcare burdens.
        - **Component Plot**: Breaks down the forecast into **trend** and **seasonality**, revealing long-term vs short-term patterns.
        - **MSE Metric**: Provides a numeric evaluation of forecast accuracy. Lower values indicate better performance.
        """)


    except Exception as e:
        st.error(f"Error generating forecast: {e}")


# --- AMU VS RESISTANCE PAGE ---
elif selection == "AMU vs Resistance":
    st.title("💊 AMU vs Resistance Correlation")
    st.markdown("Explore how antimicrobial usage trends relate to the development of resistance in specific pathogens.")

    try:
        from data_loader import amu_data, adult_yearly_df

        # Static example: Cephalosporins vs. K. pneumoniae
        st.subheader("Cephalosporin Usage vs Klebsiella pneumoniae Resistance")
        fig_corr = plot_amu_vs_resistance(amu_data, adult_yearly_df)
        st.pyplot(fig_corr)
        
        st.subheader("📈 Usage and Resistance Over Time")
        fig_trend = plot_amu_resistance_timeseries(amu_data, adult_yearly_df)
        st.pyplot(fig_trend)
        
        st.subheader("🔍 Explore Other Drug-Pathogen Correlations")
        drug_options = amu_data['ATC4Name'].dropna().unique().tolist()
        pathogen_options = adult_yearly_df['Pathogen'].dropna().unique().tolist()
    
        selected_drug = st.selectbox("Select Antibiotic Class (ATC code)", sorted(drug_options))
        selected_pathogen = st.selectbox("Select Pathogen", sorted(pathogen_options))
    
        fig_custom = plot_custom_amu_vs_resistance(amu_data, adult_yearly_df, selected_drug, selected_pathogen)
        st.pyplot(fig_custom)

        st.markdown("""
        ### 🧠 Interpretation: AMU vs Resistance
        
        - **Correlation Plot**: Quantifies how antibiotic usage aligns with resistance patterns. A higher Pearson r suggests potential overuse or misuse.
        - **Time Series Overlay**: Visualizes usage and resistance side-by-side to detect trend alignment or lags.
        - **Interactive Pairwise Comparison**: Lets users explore specific drug-pathogen pairs to draw more localized insights.
        """)


    except Exception as e:
        st.error(f"Error rendering AMU vs Resistance plot: {e}")


# --- MECHANISM CLUSTERING PAGE ---
elif selection == "Mechanism Clustering":
    st.title("🧪 Mechanism-Based Clustering")
    st.markdown("This section visualizes antimicrobial resistance gene clusters using unsupervised learning on CARD features.")

    try:
        from data_loader import df_cluster
        fig_mech = plot_mechanism_clusters(df_cluster)
        st.pyplot(fig_mech)

        st.markdown("""
        ### 🧠 Interpretation: Mechanism Clustering
        
        - **PCA Clustering Plot**: Groups AMR genes based on shared biological properties.
        - **Cluster Labels**: Summarize dominant resistance mechanisms and associated drug classes for each group.
        - **Takeaway**: Genes with similar resistance behavior may require similar diagnostic or therapeutic approaches.
        """)


        
    except Exception as e:
        st.error(f"Error rendering Mechanism Clustering plot: {e}")


# --- ML FEATURE IMPORTANCE PAGE ---
elif selection == "ML Feature Importance":
    st.title("🤖 ML Feature Importance: Resistance Mechanisms")
    st.markdown("Explore which genetic or functional features are most important in predicting resistance gene clusters using a Random Forest classifier.")

    try:
        from data_loader import feature_importance_df
        fig_feat = plot_feature_importance(feature_importance_df)
        st.pyplot(fig_feat)

        st.markdown("""
        ### 🧠 Interpretation: ML Feature Importance
        
        - **Top Features**: Indicates which biological features (e.g., gene family, resistance mechanism) most influence cluster prediction.
        - **Random Forest Model**: A powerful yet interpretable model used to assess feature contributions.
        - **Use Case**: Helps focus attention on key attributes that drive resistance behavior at the genomic level.
        """)

    except Exception as e:
        st.error(f"Error rendering ML Feature Importance plot: {e}")