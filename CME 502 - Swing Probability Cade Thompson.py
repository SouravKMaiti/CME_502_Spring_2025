import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pybaseball import statcast
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import plotly.offline as pyo
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Data fetching with caching
def cache_statcast_data(batter_id, start, end, filename="statcast_cache.csv"):
    try:
        if os.path.exists(filename):
            logger.info(f"Loading cached data from {filename}")
            data = pd.read_csv(filename)
        else:
            logger.info(f"Fetching data with statcast() for {start} to {end}")
            data = statcast(start, end)
            data.to_csv(filename, index=False)
        data = data[data['batter'] == batter_id]
        logger.info(f"Filtered data for batter {batter_id}: {len(data)} rows")
        return data
    except Exception as e:
        logger.error(f"Error in cache_statcast_data: {e}")
        raise

def get_batter_data(batter_id=592450, start='2023-04-01', end='2022-06-01'):
    try:
        data = cache_statcast_data(batter_id, start, end)
        if data.empty:
            logger.warning("No data for batter. Trying broader date range: 2023-03-01 to 2023-07-01")
            data = cache_statcast_data(batter_id, '2023-03-01', '2023-07-01')
        
        # Select relevant columns
        data = data[['release_speed', 'release_spin_rate', 'plate_x', 'plate_z', 'description', 'pitch_type']].copy()
        logger.info(f"Selected columns, data shape: {data.shape}")
        
        # Handle missing values
        data = data.dropna()
        logger.info(f"After dropna, data shape: {data.shape}")
        
        # Label swings (1) vs. takes (0)
        swing_events = ['hit_into_play', 'foul', 'swinging_strike', 'swinging_strike_blocked']
        data['swing'] = data['description'].apply(lambda x: 1 if x in swing_events else 0)
        logger.info(f"Swing column created, swing rate: {data['swing'].mean():.3f}")
        
        # Filter to reasonable pitch locations
        data = data[(data['plate_x'].between(-1.5, 1.5)) & (data['plate_z'].between(1, 4))]
        logger.info(f"After location filter, data shape: {data.shape}")
        
        if len(data) < 100:
            raise ValueError(f"Not enough data for analysis: {len(data)} pitches")
        
        return data
    except Exception as e:
        logger.error(f"Error in get_batter_data: {e}")
        raise

# Logistic regression model
def train_swing_model(data):
    try:
        X = data[['release_speed', 'release_spin_rate', 'plate_x', 'plate_z']].values
        y = data['swing'].values
        logger.info(f"Training model with {len(X)} samples")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train logistic regression
        model = LogisticRegression()
        model.fit(X_scaled, y)
        
        # Get feature importance
        importance = np.abs(model.coef_[0])
        feature_names = ['Velocity', 'Spin Rate', 'Plate X', 'Plate Z']
        logger.info(f"Model trained, feature importance: {importance}")
        
        return model, scaler, importance, feature_names
    except Exception as e:
        logger.error(f"Error in train_swing_model: {e}")
        raise

# Swing probability and interpolation
def compute_swing_probability(data, model, scaler, feature='release_speed'):
    try:
        X = data[['release_speed', 'release_spin_rate', 'plate_x', 'plate_z']].values
        X_scaled = scaler.transform(X)
        
        # Predict probabilities
        probs = model.predict_proba(X_scaled)[:, 1]
        logger.info(f"Predicted probabilities, mean: {probs.mean():.3f}")
        
        # Handle duplicates by averaging probabilities for each unique feature value
        df = pd.DataFrame({'x': data[feature], 'prob': probs})
        df_agg = df.groupby('x').agg({'prob': 'mean'}).reset_index()
        x_vals = df_agg['x'].values
        probs = df_agg['prob'].values
        
        # Sort by x_vals
        sort_idx = np.argsort(x_vals)
        x_vals = x_vals[sort_idx]
        probs = probs[sort_idx]
        
        # Apply a larger rolling average to smooth probabilities
        window_size = max(5, len(probs) // 10)  # Increased to 10% of data points
        probs_smooth = pd.Series(probs).rolling(window=window_size, center=True, min_periods=1).mean().values
        
        logger.info(f"Unique x values after aggregation: {len(x_vals)}")
        if len(x_vals) < 2:
            raise ValueError(f"Too few unique values for interpolation: {len(x_vals)}")
        
        # Interpolate with linear method for smoother curve
        interp_func = interp1d(x_vals, probs_smooth, kind='linear', fill_value='extrapolate')
        x_dense = np.linspace(min(x_vals), max(x_vals), 500)
        probs_dense = interp_func(x_dense)
        
        # Compute derivative and smooth it with a larger window
        dy_dx = np.gradient(probs_dense, x_dense)
        dy_dx_smooth = savgol_filter(dy_dx, window_length=51, polyorder=2)  # Increased window length
        
        logger.info(f"Interpolated probabilities, range: {probs_dense.min():.3f} to {probs_dense.max():.3f}")
        return x_vals, probs_smooth, x_dense, probs_dense, dy_dx_smooth
    except Exception as e:
        logger.error(f"Error in compute_swing_probability: {e}")
        raise

# 2D swing probability heatmap
# I don't think the heatmap is working at all, The outcomes don'e make any sense. I think it may be an issue with the data
def compute_swing_heatmap(data, model, scaler):
    try:
        x_grid = np.linspace(-1.5, 1.5, 50)
        z_grid = np.linspace(1, 4, 50)
        X, Z = np.meshgrid(x_grid, z_grid)
        
        # Use mean velocity, spin rate, and location
        mean_speed = data['release_speed'].mean()
        mean_spin = data['release_spin_rate'].mean()
        mean_plate_x = data['plate_x'].mean()
        mean_plate_z = data['plate_z'].mean()
        logger.info(f"Mean speed: {mean_speed:.2f}, mean spin: {mean_spin:.2f}, mean plate_x: {mean_plate_x:.2f}, mean plate_z: {mean_plate_z:.2f}")
        
        # Compute swing probabilities
        probs = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                input_data = np.array([[mean_speed, mean_spin, X[i, j], Z[i, j]]])
                input_scaled = scaler.transform(input_data)
                probs[i, j] = model.predict_proba(input_scaled)[:, 1]
        
        logger.info(f"Heatmap computed, probability range: {probs.min():.3f} to {probs.max():.3f}")
        return x_grid, z_grid, probs
    except Exception as e:
        logger.error(f"Error in compute_swing_heatmap: {e}")
        raise

# Expected swings via integration
def compute_expected_swings(data, model, scaler):
    try:
        x_grid = np.linspace(-1.5, 1.5, 50)
        z_grid = np.linspace(1, 4, 50)
        _, _, probs = compute_swing_heatmap(data, model, scaler)
        
        # Integrate over plate_x for each plate_z
        expected_swings = np.array([simpson(y=probs[i, :], x=x_grid) for i in range(len(z_grid))])
        total_expected = simpson(y=expected_swings, x=z_grid)
        
        # Normalize by the area of the grid to get probability per pitch
        x_min, x_max = x_grid[0], x_grid[-1]
        z_min, z_max = z_grid[0], z_grid[-1]
        area = (x_max - x_min) * (z_max - z_min)
        total_expected /= area
        
        logger.info(f"Expected swings (normalized): {total_expected:.3f}")
        return total_expected
    except Exception as e:
        logger.error(f"Error in compute_expected_swings: {e}")
        raise

# Plotting
def plot_results(x_vals, probs_smooth, x_dense, probs_dense, dy_dx, feature, importance, feature_names, x_grid, z_grid, probs, optimal_speed=None):
    try:
        # Swing probability curve with raw data points
        plt.figure(figsize=(8, 5))
        plt.scatter(x_vals, probs_smooth, color='gray', alpha=0.5, label='Raw Data (Averaged)', s=30)
        plt.plot(x_dense, probs_dense, label='Swing Probability (Smoothed)', color='blue')
        plt.plot(x_dense, dy_dx, label='Derivative (Sensitivity)', color='red', linestyle='--')
        if optimal_speed:
            plt.axvline(x=optimal_speed, color='green', linestyle=':', label=f'Optimal Speed ({optimal_speed:.1f} mph)')
        plt.xlabel(feature.replace('_', ' ').title())
        plt.ylabel('Probability / Sensitivity')
        plt.title(f'Swing Probability vs. {feature.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        logger.info("Plotted swing probability curve")
        
        # Feature importance
        plt.figure(figsize=(8, 5))
        plt.bar(feature_names, importance, color='green')
        plt.xlabel('Features')
        plt.ylabel('Importance (Abs. Coefficient)')
        plt.title('Feature Importance for Swing Decision')
        plt.tight_layout()
        plt.show()
        logger.info("Plotted feature importance")
        
        # Interactive heatmap with strike zone overlay
        fig = go.Figure(data=go.Heatmap(
            x=x_grid, y=z_grid, z=probs,
            colorscale='Viridis', zmin=0, zmax=1
        ))
        fig.update_layout(
            title='Swing Probability by Pitch Location',
            xaxis_title='Plate X (ft)',
            yaxis_title='Plate Z (ft)',
            template='plotly_white',
            shapes=[
                dict(
                    type='rect',
                    x0=-0.83, x1=0.83, y0=1.5, y1=3.5,
                    line=dict(color='white', width=2),
                    fillcolor=None
                )
            ]
        )
        pyo.plot(fig, filename='swing_heatmap.html', auto_open=False)
        logger.info("Interactive heatmap saved as 'swing_heatmap.html'")
    except Exception as e:
        logger.error(f"Error in plot_results: {e}")
        raise

# Plot Swing probability by pitch type
def plot_swing_by_pitch_type(data, model, scaler):
    try:
        # Predict probabilities for all pitches
        X = data[['release_speed', 'release_spin_rate', 'plate_x', 'plate_z']].values
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[:, 1]
        
        # Combine with pitch type
        df = pd.DataFrame({'pitch_type': data['pitch_type'], 'prob': probs})
        
        # Group by pitch type and compute mean swing probability
        swing_by_pitch = df.groupby('pitch_type').agg({'prob': 'mean'}).sort_values('prob', ascending=False)
        
        # Plot
        plt.figure(figsize=(8, 5))
        swing_by_pitch['prob'].plot(kind='bar', color='purple')
        plt.xlabel('Pitch Type')
        plt.ylabel('Average Swing Probability')
        plt.title('Average Swing Probability by Pitch Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        logger.info("Plotted swing probability by pitch type")
    except Exception as e:
        logger.error(f"Error in plot_swing_by_pitch_type: {e}")
        raise

# Optimize for minimum swing probability
def optimize_min_swing(data, model, scaler, x_dense, probs_dense):
    try:
        # Define objective function (negative swing probability to maximize takes)
        def objective(speed):
            idx = np.argmin(np.abs(x_dense - speed))
            return probs_dense[idx]  # Minimize probability directly
        
        # Optimize over the range of x_dense
        result = minimize(objective, x0=x_dense.mean(), bounds=[(min(x_dense), max(x_dense))])
        optimal_speed = result.x[0]
        logger.info(f"Optimal speed for minimum swings: {optimal_speed:.1f} mph")
        return optimal_speed
    except Exception as e:
        logger.error(f"Error in optimize_min_swing: {e}")
        return None

# Running the Code
if __name__ == "__main__":
    batter_id = 592450  # Aaron Judge
    try:
        logger.info(f"Starting analysis for batter ID {batter_id}")
        # Get data
        data = get_batter_data(batter_id)
        logger.info(f"Analyzing {len(data)} pitches")
        
        # Train model
        model, scaler, importance, feature_names = train_swing_model(data)
        
        # Compute swing probability for velocity
        x_vals, probs_smooth, x_dense, probs_dense, dy_dx = compute_swing_probability(data, model, scaler, feature='release_speed')
        
        # Optimize for minimum swing probability
        optimal_speed = optimize_min_swing(data, model, scaler, x_dense, probs_dense)
        
        # Compute 2D heatmap
        x_grid, z_grid, probs = compute_swing_heatmap(data, model, scaler)
        
        # Compute expected swings
        expected_swings = compute_expected_swings(data, model, scaler)
        
        # Plot results
        plot_results(x_vals, probs_smooth, x_dense, probs_dense, dy_dx, 'release_speed', importance, feature_names, x_grid, z_grid, probs, optimal_speed)
        
        # New plot: Swing probability by pitch type
        plot_swing_by_pitch_type(data, model, scaler)
        
        # Print summary
        print("\nSwing Decision Summary:")
        print(f"Top Feature: {feature_names[np.argmax(importance)]} (Importance: {importance.max():.3f})")
        print(f"Expected Swings per Pitch (Integrated): {expected_swings:.3f}")
        print(f"Swing Rate: {data['swing'].mean():.3f} ({100*data['swing'].mean():.1f}%)")
        if optimal_speed:
            print(f"Optimal Pitch Speed to Minimize Swings: {optimal_speed:.1f} mph")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"Error: {e}")
        
        