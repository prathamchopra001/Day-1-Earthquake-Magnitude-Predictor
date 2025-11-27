import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, Any, List


def create_magnitude_gauge(
    prediction: float,
    ci_lower: float,
    ci_upper: float,
    title: str = "Predicted Magnitude"
) -> go.Figure:
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 3], 'color': '#90EE90'},      # Light green
                {'range': [3, 4], 'color': '#FFFFE0'},      # Light yellow
                {'range': [4, 5], 'color': '#FFD700'},      # Gold
                {'range': [5, 6], 'color': '#FFA500'},      # Orange
                {'range': [6, 7], 'color': '#FF6347'},      # Tomato
                {'range': [7, 10], 'color': '#DC143C'},     # Crimson
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prediction
            }
        }
    ))
    
    # Add CI annotation
    fig.add_annotation(
        x=0.5, y=-0.15,
        text=f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]",
        showarrow=False,
        font=dict(size=14),
        xref="paper",
        yref="paper"
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=50, b=50)
    )
    
    return fig


def create_uncertainty_distribution(
    mean: float,
    std: float,
    n_points: int = 200
) -> go.Figure:
    
    # Generate distribution
    x_min = max(0, mean - 4 * std)
    x_max = mean + 4 * std
    x = np.linspace(x_min, x_max, n_points)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    
    # Create figure
    fig = go.Figure()
    
    # Add 95% CI shaded region
    ci_lower = mean - 1.96 * std
    ci_upper = mean + 1.96 * std
    
    x_ci = x[(x >= ci_lower) & (x <= ci_upper)]
    y_ci = y[(x >= ci_lower) & (x <= ci_upper)]
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([[x_ci[0]], x_ci, [x_ci[-1]]]),
        y=np.concatenate([[0], y_ci, [0]]),
        fill='toself',
        fillcolor='rgba(255, 193, 7, 0.4)',
        line=dict(color='rgba(255, 193, 7, 0)'),
        name='95% CI',
        hoverinfo='skip'
    ))
    
    # Add main curve
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color='#1f77b4', width=2),
        name='Probability Density',
        hovertemplate='Magnitude: %{x:.2f}<br>Density: %{y:.4f}<extra></extra>'
    ))
    
    # Add mean line
    fig.add_vline(
        x=mean,
        line_dash="dash",
        line_color="green",
        annotation_text=f"μ = {mean:.2f}",
        annotation_position="top"
    )
    
    # Add CI boundary lines
    fig.add_vline(x=ci_lower, line_dash="dot", line_color="orange")
    fig.add_vline(x=ci_upper, line_dash="dot", line_color="orange")
    
    fig.update_layout(
        title="Prediction Probability Distribution",
        xaxis_title="Magnitude",
        yaxis_title="Probability Density",
        height=300,
        margin=dict(l=50, r=30, t=50, b=50),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig


def create_regional_stats_chart(context: Dict[str, Any]) -> go.Figure:
    
    metrics = [
        ('Events (7d)', context.get('rolling_count_7d', 0)),
        ('Events (30d)', context.get('rolling_count_30d', 0)),
    ]
    
    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=['#1f77b4', '#ff7f0e'],
            text=values,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Regional Seismic Activity",
        yaxis_title="Event Count",
        height=250,
        margin=dict(l=50, r=30, t=50, b=50),
        showlegend=False
    )
    
    return fig


def create_error_band_chart(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    max_points: int = 100
) -> go.Figure:
    
    # Subsample if too many points
    if len(y_true) > max_points:
        idx = np.random.choice(len(y_true), max_points, replace=False)
        idx = np.sort(idx)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
        y_std = y_std[idx]
    
    x = np.arange(len(y_true))
    
    fig = go.Figure()
    
    # Add error bands
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([y_pred + 1.96*y_std, (y_pred - 1.96*y_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI'
    ))
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=x, y=y_pred,
        mode='lines',
        line=dict(color='#1f77b4'),
        name='Predicted'
    ))
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=x, y=y_true,
        mode='markers',
        marker=dict(color='red', size=6),
        name='Actual'
    ))
    
    fig.update_layout(
        title="Predictions vs Actual Values",
        xaxis_title="Sample Index",
        yaxis_title="Magnitude",
        height=400,
        margin=dict(l=50, r=30, t=50, b=50)
    )
    
    return fig


def create_calibration_chart(
    expected: List[float],
    actual: List[float]
) -> go.Figure:
    
    fig = go.Figure()
    
    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Perfect Calibration'
    ))
    
    # Actual calibration
    fig.add_trace(go.Scatter(
        x=expected, y=actual,
        mode='lines+markers',
        line=dict(color='#1f77b4'),
        marker=dict(size=10),
        name='Model Calibration'
    ))
    
    # Fill between
    fig.add_trace(go.Scatter(
        x=expected + expected[::-1],
        y=actual + expected[::-1],
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Calibration Gap',
        showlegend=False
    ))
    
    fig.update_layout(
        title="Model Calibration",
        xaxis_title="Expected Coverage",
        yaxis_title="Actual Coverage",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=350,
        margin=dict(l=50, r=30, t=50, b=50)
    )
    
    return fig


def get_magnitude_severity(magnitude: float) -> Dict[str, Any]:
    
    if magnitude < 3:
        return {
            'level': 'Minor',
            'color': '#28a745',
            'icon': '🟢',
            'description': 'Generally not felt, but recorded by seismographs'
        }
    elif magnitude < 4:
        return {
            'level': 'Light',
            'color': '#ffc107',
            'icon': '🟡',
            'description': 'Often felt, but rarely causes damage'
        }
    elif magnitude < 5:
        return {
            'level': 'Moderate',
            'color': '#fd7e14',
            'icon': '🟠',
            'description': 'Can cause minor damage to poorly constructed buildings'
        }
    elif magnitude < 6:
        return {
            'level': 'Strong',
            'color': '#dc3545',
            'icon': '🔴',
            'description': 'Can cause significant damage in populated areas'
        }
    elif magnitude < 7:
        return {
            'level': 'Major',
            'color': '#721c24',
            'icon': '⭕',
            'description': 'Can cause serious damage over large areas'
        }
    else:
        return {
            'level': 'Great',
            'color': '#1a1a1a',
            'icon': '⚫',
            'description': 'Can cause devastating damage across very large areas'
        }