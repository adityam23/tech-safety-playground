"""Plotting helpers for feature analysis and steering demos."""

from __future__ import annotations

import plotly.graph_objects as go


def plot_loss_curves(
    steps: list[int],
    recon_loss: list[float],
    l1_loss: list[float],
    title: str = "SAE Training Loss",
) -> go.Figure:
    """Plot reconstruction and L1 loss curves."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=recon_loss, name="Reconstruction (MSE)"))
    fig.add_trace(go.Scatter(x=steps, y=l1_loss, name="L1"))
    fig.update_layout(title=title, xaxis_title="Step", yaxis_title="Loss")
    return fig


def plot_feature_density(
    feature_ids: list[int],
    activation_freq: list[float],
    title: str = "Feature Activation Density",
) -> go.Figure:
    """Bar plot of feature activation frequencies."""
    fig = go.Figure(go.Bar(x=feature_ids, y=activation_freq))
    fig.update_layout(
        title=title,
        xaxis_title="Feature ID",
        yaxis_title="Activation Frequency",
    )
    return fig
