from prophet import Prophet
from prophet.plot import plot_components_plotly
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def generate_forecast_plot(df, selected_pathogen, group_label):
    """
    Forecast resistance trend for a selected pathogen using Prophet.

    Parameters:
        df (pd.DataFrame): Source dataframe containing 'Pathogen', 'Year', 'Frequency Count'.
        selected_pathogen (str): Pathogen to forecast.
        group_label (str): 'Adult' or 'Pediatric'.

    Returns:
        fig (matplotlib.figure.Figure): Matplotlib figure with forecast plot.
    """
    # Step 1: Filter and prepare data
    df_pathogen = df[df['Pathogen'] == selected_pathogen].groupby('Year')['Frequency Count'].sum().reset_index()
    df_pathogen.columns = ['ds', 'y']  # Prophet requires these column names

    # Step 2: Fit Prophet model
    model = Prophet(yearly_seasonality=True, interval_width=0.80)
    model.fit(df_pathogen)

    # Step 3: Create future dataframe and forecast
    future = model.make_future_dataframe(periods=5, freq='Y')
    forecast = model.predict(future)

    # Step 4: Plot results
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_pathogen['ds'], df_pathogen['y'], label='Historical', marker='o')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='green')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='green', alpha=0.2)
    ax.set_title(f"Forecasted Resistance for {selected_pathogen} ({group_label})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Frequency Count")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

     # Return both forecast figure and component plot
    component_fig = plot_components_plotly(model, forecast)

    # Align predictions with actuals
    actual_len = len(df_pathogen)
    actuals = df_pathogen['y']
    predicted = forecast['yhat'][:actual_len]
    
    # Calculate forecast error
    mse = mean_squared_error(actuals, predicted)


    return fig, component_fig, mse

