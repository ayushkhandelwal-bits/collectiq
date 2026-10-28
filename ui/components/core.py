"""Shared UI components and formatting for CollectIQ."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

STYLE_DIR = Path(__file__).resolve().parents[1] / "styles"


# ----------------------------------------------------------------- formatting
def _indian_group(n: int) -> str:
    s = str(abs(n))
    if len(s) <= 3:
        out = s
    else:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        out = ",".join(parts) + "," + tail
    return ("-" if n < 0 else "") + out


def inr(x: float | None, short: bool = False) -> str:
    """Format INR with Indian digit grouping; ``short`` uses lakh / crore units."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if short:
        a = abs(x)
        if a >= 1e7:
            return f"₹{x / 1e7:,.2f} Cr"
        if a >= 1e5:
            return f"₹{x / 1e5:,.2f} L"
    return "₹" + _indian_group(int(round(x)))


def pct(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:.{digits}f}%"


def num(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return _indian_group(int(round(x)))


def fmt_date(s: Any) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "—"
    try:
        return pd.Timestamp(s).strftime("%d %b %Y")
    except Exception:
        return str(s)


# ----------------------------------------------------------------- components
def inject_css() -> None:
    css = (STYLE_DIR / "theme.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f'<p class="ciq-sub">{html.escape(subtitle)}</p>', unsafe_allow_html=True)


def provenance_band(items: list[tuple[str, str]], synthetic: bool = True) -> None:
    """Thin band stating data mode and context for the page."""
    tag = ('<span><span class="ciq-tag syn">Synthetic data</span></span>' if synthetic
           else '<span><span class="ciq-tag ok">Public data</span></span>')
    cells = "".join(f"<span>{html.escape(k)} <b>{html.escape(v)}</b></span>" for k, v in items)
    st.markdown(f'<div class="ciq-band">{tag}{cells}</div>', unsafe_allow_html=True)


def kpi_strip(kpis: list[dict[str, Any]]) -> None:
    """Ledger-style KPI cells. Each dict: label, value, detail, tone, estimate, help."""
    cells = []
    for k in kpis:
        cls = "ciq-kpi est" if k.get("estimate") else "ciq-kpi"
        tone = k.get("tone", "")
        title = html.escape(k.get("help", ""), quote=True)
        detail = html.escape(k.get("detail", ""))
        cells.append(
            f'<div class="{cls}" title="{title}"><div class="l">{html.escape(k["label"])}</div>'
            f'<div class="v">{html.escape(str(k["value"]))}</div>'
            f'<div class="d {tone}">{detail}</div></div>')
    st.markdown(f'<div class="ciq-kpis" style="--n:{len(kpis)}">{"".join(cells)}</div>',
                unsafe_allow_html=True)


def note(text: str, adverse: bool = False) -> None:
    cls = "ciq-note adverse" if adverse else "ciq-note"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def empty_state(text: str) -> None:
    st.markdown(f'<div class="ciq-empty">{html.escape(text)}</div>', unsafe_allow_html=True)


def uncertainty_warning(extra: str = "") -> None:
    note("Model estimates on synthetic data. Expected values are not guaranteed amounts and "
         "do not establish that an action causes a payment. " + extra)


def tag(text: str, kind: str = "neutral") -> str:
    return f'<span class="ciq-tag {kind}">{html.escape(text)}</span>'


def insights(rows: Iterable[tuple[str, str]]) -> None:
    out = "".join(f'<div class="ciq-insight"><div class="k">{html.escape(k)}</div>{v}</div>'
                  for k, v in rows)
    st.markdown(out, unsafe_allow_html=True)


def methodology(title: str, body: str) -> None:
    with st.expander(title):
        st.markdown(body)


def download_csv(df: pd.DataFrame, label: str, file_name: str, key: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=file_name,
                       mime="text/csv", key=key, disabled=df.empty)


def footer(version: str, data_line: str) -> None:
    st.markdown(f'<div class="ciq-footer">CollectIQ {html.escape(version)} — decision-support '
                f'prototype, not an operationally validated lending system. '
                f'{html.escape(data_line)}</div>', unsafe_allow_html=True)


MONEY_COL = st.column_config.NumberColumn
