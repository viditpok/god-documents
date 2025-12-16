import plotly.graph_objects as go
from ui.components import show_plotly
from ui.theming import PLOTLY_CONFIG


def test_show_plotly_uses_global_config(monkeypatch):
    captured: dict[str, object] = {}

    def fake_plotly_chart(fig, config: dict) -> None:
        captured["fig"] = fig
        captured["config"] = config

    def fake_apply_theme(fig: go.Figure) -> go.Figure:
        return fig

    monkeypatch.setattr("ui.components.st.plotly_chart", fake_plotly_chart)
    monkeypatch.setattr("ui.components.apply_plot_theme", fake_apply_theme)

    fig = go.Figure()
    show_plotly(fig)

    assert captured["fig"] is fig
    assert captured["config"] is PLOTLY_CONFIG
