"""Plotly chart helpers with one consistent visual system."""

from __future__ import annotations

import plotly.graph_objects as go

INK = "#13233F"
SLATE = "#51607A"
RULE = "#DCE2E9"
ACCENT = "#0B6E6E"
ADVERSE = "#B4442C"
CATEGORICAL = ["#0B6E6E", "#13233F", "#7C8BA1", "#C9962B", "#B4442C", "#5B8DB8", "#9DB8A8"]
BUCKET_COLORS = {"Current": "#9DB8A8", "1-29 DPD": "#E2C27A", "30-59 DPD": "#C9962B",
                 "60-89 DPD": "#C46A3C", "90+ DPD": "#B4442C", "Charged Off": "#6E2A1C",
                 "Closed": "#C3CBD6"}
SEQUENTIAL = [[0, "#F2F7F7"], [0.5, "#5FA3A0"], [1, "#0B4F50"]]
ADVERSE_SEQ = [[0, "#FBF3EF"], [0.5, "#D98A6C"], [1, "#7A2A18"]]


def style(fig: go.Figure, title: str, height: int = 330, x_title: str | None = None,
          y_title: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "y": 1, "yanchor": "top",
               "yref": "container", "pad": {"t": 8},
               "font": {"size": 14, "color": INK}},
        height=height + (40 if legend else 0),
        margin={"l": 8, "r": 8, "t": 44, "b": (66 if x_title else 48) if legend else 8},
        paper_bgcolor="white", plot_bgcolor="white",
        font={"family": "Public Sans, system-ui, sans-serif", "size": 12, "color": SLATE},
        colorway=CATEGORICAL, showlegend=legend,
        legend={"orientation": "h", "yanchor": "top", "y": -0.24 if x_title else -0.12,
                "x": 0,
                "font": {"size": 11}, "title": None},
        hoverlabel={"font": {"family": "Public Sans, sans-serif"}},
    )
    fig.update_xaxes(title=x_title, gridcolor=RULE, linecolor=RULE, zeroline=False,
                     title_font={"size": 11})
    fig.update_yaxes(title=y_title, gridcolor=RULE, linecolor=RULE, zeroline=False,
                     title_font={"size": 11})
    return fig
