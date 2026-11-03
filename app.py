"""
CleanData - a free web tool to clean datasets (CSV / JSON / TXT / Excel / pasted text)

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
Then open http://localhost:8501
"""

import io
import json

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CleanData - Free Data Cleaner", page_icon="🧹", layout="wide")

NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "missing"}


# --------------------------------------------------------------------------
# 1. LOADING DATA
# --------------------------------------------------------------------------
def parse_json_text(raw: str) -> pd.DataFrame:
    data = json.loads(raw)
    if isinstance(data, list):
        return pd.json_normalize(data)
    if isinstance(data, dict):
        # dict of lists -> columns ; otherwise look for a list inside the dict
        if all(isinstance(v, list) for v in data.values()):
            return pd.DataFrame(data)
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return pd.json_normalize(v)
        return pd.json_normalize(data)
    raise ValueError("Unsupported JSON structure")


def parse_text(raw: str) -> pd.DataFrame:
    raw = raw.strip()
    if raw.startswith("[") or raw.startswith("{"):
        try:
            return parse_json_text(raw)
        except Exception:
            pass
    # delimiter auto-detection (comma, tab, semicolon, pipe)
    return pd.read_csv(io.StringIO(raw), sep=None, engine="python")


def load_data(uploaded, pasted: str):
    if uploaded is not None:
        name = uploaded.name.lower()
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded)
        raw = uploaded.getvalue().decode("utf-8", errors="replace")
        if name.endswith(".json"):
            return parse_json_text(raw)
        return parse_text(raw)  # csv, tsv, txt
    if pasted and pasted.strip():
        return parse_text(pasted)
    return None


# --------------------------------------------------------------------------
# 2. CLEANING ENGINE
# --------------------------------------------------------------------------
def standardize_missing(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Trim whitespace and convert placeholders like 'NA', 'null', '?' to real NaN."""
    df = df.copy()
    before = int(df.isna().sum().sum())
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].astype("string").str.strip()
        df[col] = s.mask(s.str.lower().isin(NULL_TOKENS), pd.NA).astype(object)
    after = int(df.isna().sum().sum())
    log.append(f"Trimmed whitespace; converted placeholders to nulls (+{after - before} nulls found)")
    return df


def coerce_numeric(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Convert text columns that are mostly numbers into numeric columns."""
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        converted = pd.to_numeric(non_null.astype(str).str.replace(",", "", regex=False), errors="coerce")
        if converted.notna().mean() >= 0.8:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
            log.append(f"Column '{col}' converted to numeric")
    return df


def remove_duplicates(df, subset, keep, log):
    before = len(df)
    df = df.drop_duplicates(subset=subset or None, keep=keep)
    log.append(f"Removed {before - len(df)} duplicate rows")
    return df


def handle_missing(df, mode, num_strategy, cat_strategy, num_const, cat_const, col_threshold, log):
    df = df.copy()

    if mode == "Drop rows with any null":
        before = len(df)
        df = df.dropna()
        log.append(f"Dropped {before - len(df)} rows containing nulls")
        return df

    # drop columns that are mostly empty
    if col_threshold < 100:
        pct = df.isna().mean() * 100
        drop_cols = pct[pct > col_threshold].index.tolist()
        if drop_cols:
            df = df.drop(columns=drop_cols)
            log.append(f"Dropped columns with >{col_threshold}% nulls: {drop_cols}")

    if mode == "Fill nulls":
        for col in df.columns:
            n_null = int(df[col].isna().sum())
            if n_null == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                if num_strategy == "Mean (average)":
                    val = df[col].mean()
                elif num_strategy == "Median":
                    val = df[col].median()
                elif num_strategy == "Mode (most frequent)":
                    m = df[col].mode()
                    val = m.iloc[0] if not m.empty else 0
                elif num_strategy == "Constant":
                    val = num_const
                elif num_strategy == "Forward fill":
                    df[col] = df[col].ffill().bfill(); log.append(f"'{col}': forward-filled {n_null} nulls"); continue
                else:  # Backward fill
                    df[col] = df[col].bfill().ffill(); log.append(f"'{col}': backward-filled {n_null} nulls"); continue
                df[col] = df[col].fillna(val)
                log.append(f"'{col}': filled {n_null} nulls with {num_strategy.split(' ')[0].lower()} ({val})")
            else:
                if cat_strategy == "Mode (most frequent)":
                    m = df[col].mode()
                    val = m.iloc[0] if not m.empty else "Unknown"
                elif cat_strategy == "Constant":
                    val = cat_const
                elif cat_strategy == "Forward fill":
                    df[col] = df[col].ffill().bfill(); log.append(f"'{col}': forward-filled {n_null} nulls"); continue
                else:
                    df[col] = df[col].bfill().ffill(); log.append(f"'{col}': backward-filled {n_null} nulls"); continue
                df[col] = df[col].fillna(val)
                log.append(f"'{col}': filled {n_null} nulls with '{val}'")
    return df


def remove_outliers_iqr(df, factor, log):
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return df
    mask = pd.Series(True, index=df.index)
    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        mask &= df[col].between(q1 - factor * iqr, q3 + factor * iqr) | df[col].isna()
    removed = int((~mask).sum())
    log.append(f"Removed {removed} outlier rows (IQR x {factor})")
    return df[mask]


def profile(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nulls": df.isna().sum(),
        "null %": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(),
    })


# --------------------------------------------------------------------------
# 3. UI
# --------------------------------------------------------------------------
st.title("🧹 CleanData - Free Data Cleaning Tool")
st.caption("Upload or paste your CSV / JSON / TXT / Excel data, pick your cleaning steps, download reliable data.")

with st.sidebar:
    st.header("1. Input")
    uploaded = st.file_uploader("Upload file", type=["csv", "json", "txt", "tsv", "xlsx", "xls"])
    pasted = st.text_area("...or paste data (CSV / JSON)", height=150)

try:
    df_raw = load_data(uploaded, pasted)
except Exception as e:
    st.error(f"Could not read the data: {e}")
    st.stop()

if df_raw is None:
    st.info("👈 Upload a file or paste data in the sidebar to begin.")
    st.stop()

with st.sidebar:
    st.header("2. Cleaning options")
    opt_standardize = st.checkbox("Trim spaces & treat 'NA', 'null', '?' as missing", True)
    opt_numeric = st.checkbox("Auto-convert numeric-looking text columns", True)

    opt_dupes = st.checkbox("Remove duplicate rows", True)
    dupe_subset, dupe_keep = None, "first"
    if opt_dupes:
        dupe_subset = st.multiselect("Check duplicates only on columns (empty = all)", df_raw.columns.tolist())
        dupe_keep = st.radio("Keep", ["first", "last"], horizontal=True)

    st.subheader("Missing values")
    miss_mode = st.radio("Method", ["Leave as is", "Fill nulls", "Drop rows with any null"])
    num_strategy = cat_strategy = "Mode (most frequent)"
    num_const, cat_const, col_threshold = 0.0, "Unknown", 100
    if miss_mode != "Leave as is":
        col_threshold = st.slider("Drop columns with more than X% nulls", 10, 100, 100,
                                  help="100 = never drop columns")
    if miss_mode == "Fill nulls":
        num_strategy = st.selectbox("Numeric columns", ["Mean (average)", "Median", "Mode (most frequent)",
                                                        "Constant", "Forward fill", "Backward fill"])
        if num_strategy == "Constant":
            num_const = st.number_input("Numeric constant", value=0.0)
        cat_strategy = st.selectbox("Text columns", ["Mode (most frequent)", "Constant",
                                                     "Forward fill", "Backward fill"])
        if cat_strategy == "Constant":
            cat_const = st.text_input("Text constant", "Unknown")

    opt_outliers = st.checkbox("Remove numeric outliers (IQR)", False)
    iqr_factor = st.slider("IQR factor", 1.0, 3.0, 1.5, 0.5) if opt_outliers else 1.5

# ---- run pipeline
log: list = []
df = df_raw.copy()
if opt_standardize:
    df = standardize_missing(df, log)
if opt_numeric:
    df = coerce_numeric(df, log)
if opt_dupes:
    df = remove_duplicates(df, dupe_subset, dupe_keep, log)
if miss_mode != "Leave as is":
    df = handle_missing(df, miss_mode, num_strategy, cat_strategy, num_const, cat_const, col_threshold, log)
if opt_outliers:
    df = remove_outliers_iqr(df, iqr_factor, log)
df = df.reset_index(drop=True)

# ---- report
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", len(df), delta=len(df) - len(df_raw))
c2.metric("Columns", df.shape[1], delta=df.shape[1] - df_raw.shape[1])
c3.metric("Null cells", int(df.isna().sum().sum()), delta=int(df.isna().sum().sum() - df_raw.isna().sum().sum()))
c4.metric("Duplicate rows", int(df.duplicated().sum()), delta=int(df.duplicated().sum() - df_raw.duplicated().sum()))

tab1, tab2, tab3, tab4 = st.tabs(["✅ Cleaned data", "📄 Original data", "📊 Column profile", "📝 Cleaning log"])
with tab1:
    st.dataframe(df, use_container_width=True)
with tab2:
    st.dataframe(df_raw, use_container_width=True)
with tab3:
    left, right = st.columns(2)
    left.write("**Before**"); left.dataframe(profile(df_raw), use_container_width=True)
    right.write("**After**"); right.dataframe(profile(df), use_container_width=True)
with tab4:
    for line in log or ["No operations applied."]:
        st.write("•", line)

# ---- download
st.subheader("⬇️ Download cleaned data")
d1, d2, d3 = st.columns(3)
d1.download_button("CSV", df.to_csv(index=False).encode("utf-8"), "cleaned_data.csv", "text/csv")
d2.download_button("JSON", df.to_json(orient="records", indent=2).encode("utf-8"),
                   "cleaned_data.json", "application/json")
try:
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    d3.download_button("Excel", buf.getvalue(), "cleaned_data.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
except Exception:
    d3.caption("Install openpyxl for Excel export")
