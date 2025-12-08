from __future__ import annotations

from typing import Final

import plotly.graph_objects as go
import streamlit as st

NAVY_DARK: Final[dict[str, str]] = {
    "background": "#050b18",
    "panel": "#0f172a",
    "surface": "#111c32",
    "border": "rgba(148,163,184,0.25)",
    "text": "#f1f5f9",
    "muted": "#93a7c1",
    "accent": "#38bdf8",
}
FONT_FAMILY: Final[str] = "'Inter', 'Segoe UI', sans-serif"


def configure_plotly(theme: str | None = None) -> dict[str, object]:
    if theme is not None:
        st.session_state["_mt_theme"] = theme
    return {
        "displaylogo": False,
        "responsive": True,
        "modeBarButtonsToRemove": ["select2d", "lasso2d", "zoomIn", "zoomOut"],
        "scrollZoom": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "macrotones_chart",
        },
    }


PLOTLY_CONFIG: Final[dict[str, object]] = configure_plotly()


def inject_global_styles() -> None:
    if not st.session_state.get("_mt_page_configured"):
        st.set_page_config(page_title="MacroTone", layout="wide")
        st.session_state["_mt_page_configured"] = True
    st.markdown(
        f"""
        <style>
          :root {{
            --mt-bg: {NAVY_DARK["background"]};
            --mt-panel: {NAVY_DARK["panel"]};
            --mt-surface: {NAVY_DARK["surface"]};
            --mt-border: {NAVY_DARK["border"]};
            --mt-text: {NAVY_DARK["text"]};
            --mt-muted: {NAVY_DARK["muted"]};
            --mt-accent: {NAVY_DARK["accent"]};
            --mt-font: {FONT_FAMILY};
          }}
          html, body {{
            background:
              radial-gradient(
                circle at 20% 20%,
                rgba(31,64,104,0.45),
                rgba(5,11,24,0.95)
              ),
              linear-gradient(
                120deg,
                rgba(15,23,42,0.95),
                rgba(3,7,18,0.98)
              );
            color: var(--mt-text) !important;
            font-family: var(--mt-font);
          }}
          [data-testid="stAppViewContainer"] {{
            background:
              radial-gradient(
                circle at 10% 15%,
                rgba(59,130,246,0.08),
                rgba(5,11,24,0.95)
              ),
              linear-gradient(
                200deg,
                rgba(15,23,42,0.98),
                rgba(3,7,18,0.98)
              );
          }}
          [data-testid="stSidebar"] {{
            background: var(--mt-panel) !important;
            border-right: 1px solid rgba(255,255,255,0.05);
          }}
          h1, h2, h3, h4, h5, h6,
          .stMarkdown, .stText, .stCaption {{
            color: var(--mt-text) !important;
            font-family: var(--mt-font);
          }}
          .mt-kpi {{
            background: linear-gradient(
              135deg,
              rgba(33,45,74,0.9),
              rgba(8,20,43,0.9)
            );
            border-radius: 18px;
            padding: 18px 20px;
            border: 1px solid var(--mt-border);
            box-shadow: 0 20px 35px rgba(2, 6, 23, 0.55);
            min-height: 120px;
          }}
          .mt-kpi-label {{
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--mt-muted);
            margin-bottom: 6px;
          }}
          .mt-kpi-value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--mt-text);
          }}
          [data-testid="block-container"] {{
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            background: linear-gradient(145deg, rgba(8,16,32,0.94), rgba(6,12,24,0.98));
            border-left: 1px solid rgba(255,255,255,0.02);
            border-right: 1px solid rgba(0,0,0,0.3);
            box-shadow: inset 0 0 60px rgba(0,0,0,0.35);
          }}
          [data-baseweb="tab-list"] button {{
            font-family: var(--mt-font) !important;
            font-weight: 600;
            color: rgba(226,232,240,0.75);
            border-bottom: 2px solid transparent;
          }}
          [data-baseweb="tab-list"] button[aria-selected="true"] {{
            color: var(--mt-text) !important;
            border-bottom: 2px solid var(--mt-accent);
            background: linear-gradient(
              120deg,
              rgba(37,99,235,0.16),
              rgba(56,189,248,0.12)
            );
          }}
          .stButton>button, .stDownloadButton>button {{
            font-family: var(--mt-font);
            font-weight: 600;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.15);
            background: linear-gradient(
              135deg,
              rgba(37,99,235,0.9),
              rgba(14,165,233,0.8)
            );
            color: #f8fafc;
            box-shadow: 0 15px 35px rgba(3, 7, 18, 0.4);
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_plot_theme(fig: go.Figure) -> go.Figure:
    theme = st.session_state.get("_mt_theme", "Dark")
    if theme == "Light":
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font={"color": "#0f172a", "family": FONT_FAMILY},
            legend={
                "bgcolor": "rgba(255,255,255,0.8)",
                "bordercolor": "rgba(15,23,42,0.1)",
                "font": {"color": "#0f172a"},
            },
            margin={"l": 40, "r": 25, "t": 60, "b": 40},
        )
        grid_color = "rgba(15,23,42,0.08)"
    else:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": NAVY_DARK["text"], "family": FONT_FAMILY},
            legend={
                "bgcolor": "rgba(15,23,42,0.8)",
                "bordercolor": NAVY_DARK["border"],
                "font": {"color": NAVY_DARK["text"]},
            },
            margin={"l": 40, "r": 25, "t": 60, "b": 40},
        )
        grid_color = "rgba(148,163,184,0.18)"
    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid_color,
        zeroline=False,
        linecolor=grid_color,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        zeroline=False,
        linecolor=grid_color,
    )
    return fig


__all__ = [
    "NAVY_DARK",
    "PLOTLY_CONFIG",
    "apply_plot_theme",
    "configure_plotly",
    "inject_global_styles",
]
