import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

# --- 1. CORE RESERVOIR PHYSICS MODELS ---

def hyperbolic_decline(t, qi, Di, b):
    """Arps Hyperbolic equation for Boundary Dominated Flow."""
    return qi / ((np.abs(1 + b * Di * t)) ** (1 / b))

def calculate_hyperbolic_eur(qi, Di, b, q_limit, cum_prod_historic):
    """Calculates Estimated Ultimate Recovery (EUR) using Arps."""
    remaining_reserves = (qi**b / (Di * (1 - b))) * (qi**(1 - b) - q_limit**(1 - b))
    return cum_prod_historic + remaining_reserves

def calculate_water_metrics(df):
    """Calculates WOR and Cumulative Oil for diagnostic plotting."""
    df = df[df['oil_rate'] > 0].copy()
    df['WOR'] = df['water_rate'] / df['oil_rate']
    df['water_cut'] = df['water_rate'] / (df['oil_rate'] + df['water_rate'])
    df['Np'] = (df['oil_rate'] * 10).cumsum()  # Assuming 10-day reporting intervals
    return df

def predict_eur_by_wor(df_filtered, economic_water_cut=0.95):
    """Fits log(WOR) vs Np to project EUR based on water handling limits."""
    df_fit = df_filtered[df_filtered['WOR'] > 0.1]
    x = df_fit['Np'].values
    y = np.log(df_fit['WOR'].values)
    
    slope, intercept, _, _, _ = linregress(x, y)
    target_wor = economic_water_cut / (1 - economic_water_cut)
    eur_estimated = (np.log(target_wor) - intercept) / slope
    return slope, intercept, eur_estimated

# --- 2. DATA ACQUISITION & EXECUTION ---

# Simulated historical field production data
data = {
    'days_online': np.arange(1, 201, 10),  # Changed from 210 to 201 to give exactly 20 elements
    'oil_rate':   [1000, 950, 880, 810, 750, 690, 630, 580, 530, 490, 450, 410, 380, 350, 320, 290, 270, 250, 230, 210],
    'water_rate': [50,   120, 210, 330, 440, 560, 680, 790, 890, 980, 1060, 1130, 1200, 1260, 1310, 1350, 1390, 1420, 1450, 1480]
}
df_well = pd.DataFrame(data)

# Process engineering metrics
df_processed = calculate_water_metrics(df_well)

# Fit Decline Curve (Arps Hyperbolic)
popt, _ = curve_fit(
    hyperbolic_decline, 
    df_processed['days_online'], 
    df_processed['oil_rate'], 
    bounds=([100, 0.0001, 0.01], [5000, 0.1, 0.99])
)
qi_fit, Di_fit, b_fit = popt

# Fit Water Sweep Line
limit_cut = 0.95
slope, intercept, wor_eur = predict_eur_by_wor(df_processed, economic_water_cut=limit_cut)

# --- 3. GENERATE PROFESSIONAL PLOTS ---

plt.figure(figsize=(14, 6))

# Plot 1: Standard Decline Curve
plt.subplot(1, 2, 1)
plt.plot(df_processed['days_online'], df_processed['oil_rate'], color='green', marker='o', label='Historical Oil Rate')
plt.plot(df_processed['days_online'], df_processed['water_rate'], color='blue', marker='s', label='Historical Water Rate')
plt.title('Production Rate History', fontsize=12, fontweight='bold')
plt.xlabel('Time Online (Days)')
plt.ylabel('Fluid Production Rate (STB/d)')
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc='upper right')

# Plot 2: WOR vs Cumulative Oil
plt.subplot(1, 2, 2)
plt.semilogy(df_processed['Np'], df_processed['WOR'], color='darkred', marker='o', linestyle='', label='Historical WOR Data')

np_forecast = np.linspace(df_processed['Np'].min(), wor_eur, 50)
wor_forecast = np.exp(slope * np_forecast + intercept)
plt.semilogy(np_forecast, wor_forecast, color='red', linestyle='--', linewidth=2, label='Linear Sweep Forecast')

target_wor = limit_cut / (1 - limit_cut)
plt.axhline(y=target_wor, color='black', linestyle=':', label=f'Economic Limit ({limit_cut*100:.0f}% Water Cut)')
plt.axvline(x=wor_eur, color='gray', linestyle='-.', label=f'Predicted EUR: {wor_eur:,.0f} STB')

plt.title('WOR vs. Cumulative Oil Production', fontsize=12, fontweight='bold')
plt.xlabel('Cumulative Oil Production, Np (STB)')
plt.ylabel('Water-Oil Ratio, WOR (STB/STB) - Log Scale')
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(loc='lower left')

plt.tight_layout()

# --- 4. AUTOMATICALLY CREATE DIRECTORIES AND SAVE ---
# This safely ensures folders exist before saving to prevent crashes
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/production_diagnostic_plot.png', dpi=300)
print("Success: Plot saved automatically inside the 'outputs' folder!")

# Print text report to console
print("\n--- RESERVOIR EVALUATION REPORT ---")
print(f"Initial Production Rate (qi): {qi_fit:.1f} STB/d")
print(f"Arps Decline Exponent (b): {b_fit:.2f}")
print(f"Predicted EUR at {limit_cut*100:.0f}% Water Cut Limit: {wor_eur:,.0f} STB")

plt.show()