"""Global portfolio filters (sidebar) shared by pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

PRODUCTS = ["Personal Loan", "Two-Wheeler Loan", "Consumer Durable Loan", "Business Loan",
            "Used Car Loan"]
GRADES = ["A", "B", "C", "D", "E"]


def sidebar_filters() -> None:
    st.sidebar.markdown("**Portfolio filters**")
    st.sidebar.multiselect("Loan product", PRODUCTS, key="f_products",
                           placeholder="All products")
    st.sidebar.multiselect("Risk grade", GRADES, key="f_grades", placeholder="All grades")


def selected() -> tuple[list[str], list[str]]:
    return (st.session_state.get("f_products") or [], st.session_state.get("f_grades") or [])


def apply(df: pd.DataFrame, product_col: str = "loan_product",
          grade_col: str = "risk_grade") -> pd.DataFrame:
    prods, grades = selected()
    out = df
    if prods and product_col in out.columns:
        out = out[out[product_col].isin(prods)]
    if grades and grade_col in out.columns:
        out = out[out[grade_col].isin(grades)]
    return out


def describe() -> str:
    prods, grades = selected()
    parts = []
    parts.append(", ".join(prods) if prods else "all products")
    parts.append("grades " + ", ".join(grades) if grades else "all grades")
    return "; ".join(parts)
