import math

import plotly.graph_objects as go
import streamlit as st

from ui.theming import PLOTLY_CONFIG, apply_plot_theme


def _format_value(value: float | None, suffix: str | None) -> str:
    invalid = isinstance(value, float) and (math.isnan(value) or math.isinf(value))
    if value is None or invalid:
        return "—"
    if suffix == "%":
        return f"{value:.1%}"
    if suffix:
        return f"{value:,.2f}{suffix}"
    return f"{value:,.2f}"


def kpi_card(label: str, value: float | None, suffix: str = "%") -> None:
    display_value = _format_value(value, suffix)
    st.markdown(
        f"""
        <div class="mt-kpi">
          <div class="mt-kpi-label">{label}</div>
          <div class="mt-kpi-value">{display_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_plotly(fig: go.Figure) -> None:
    apply_plot_theme(fig)
    st.plotly_chart(
        fig,
        config=PLOTLY_CONFIG,
    )


def section_divider(label: str) -> None:
    st.markdown(
        "<div style='color:var(--mt-muted);text-transform:uppercase;"
        "font-size:0.75rem;letter-spacing:0.25em;margin:24px 0 8px;'>"
        f"{label}</div>",
        unsafe_allow_html=True,
    )
