"""
Interactive Accelerometer Signal Explorer — WISDM v1.1 (PT1)
Run with:
    pip install streamlit plotly pandas numpy
    streamlit run dashboard.py
"""
import os
import tarfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="WISDM HAR Signal Explorer (PT1)", layout="wide")

ACTIVITIES_ORDER = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]
WINDOW_SIZE = 51
FS = 20.0
TIME_AXIS = np.arange(WINDOW_SIZE) / FS


@st.cache_data
def load_wisdm_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath) and os.path.exists("WISDM_ar_latest.tar.gz"):
        with tarfile.open("WISDM_ar_latest.tar.gz", "r:gz") as tar:
            for m in tar.getmembers():
                if m.name.endswith("WISDM_ar_v1.1_raw.txt"):
                    with open(filepath, "wb") as f_out:
                        f_out.write(tar.extractfile(m).read())
                    break

    records = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for seg in line.strip().split(";"):
                seg = seg.strip()
                if not seg:
                    continue
                parts = [p.strip() for p in seg.split(",")]
                if len(parts) == 7 and parts[6] == "":
                    parts = parts[:6]
                if len(parts) != 6 or "" in parts:
                    continue
                try:
                    u, act, ts = int(parts[0]), parts[1], int(parts[2])
                    x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                    if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                        records.append((u, act, ts, x, y, z))
                except ValueError:
                    continue

    df = pd.DataFrame(records, columns=["user_id", "activity", "timestamp", "x", "y", "z"])
    df["mag"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    return df


def extract_window(df: pd.DataFrame, user_id: int, activity: str, offset: int):
    sub = df[(df["user_id"] == user_id) & (df["activity"] == activity)].reset_index(drop=True)
    if len(sub) < WINDOW_SIZE:
        return None
    start = min(offset, max(0, len(sub) - WINDOW_SIZE))
    return sub.iloc[start : start + WINDOW_SIZE]


def window_stats(label: str, w: pd.DataFrame) -> dict:
    return {
        "Series": label,
        "X Mean": w["x"].mean(),
        "X Std": w["x"].std(),
        "Y Mean": w["y"].mean(),
        "Y Std": w["y"].std(),
        "Z Mean": w["z"].mean(),
        "Z Std": w["z"].std(),
        "Mag Mean": w["mag"].mean(),
        "Mag Std": w["mag"].std(),
        "Mag Range": w["mag"].max() - w["mag"].min(),
    }


# --- Sidebar Controls ---
st.sidebar.title("WISDM Explorer Controls")
data_path = st.sidebar.text_input("Raw Data File Path:", value="WISDM_ar_v1.1_raw.txt")

try:
    df = load_wisdm_data(data_path)
except FileNotFoundError:
    st.error(
        f"File `{data_path}` not found. Place `WISDM_ar_v1.1_raw.txt` or "
        "`WISDM_ar_latest.tar.gz` in the project directory."
    )
    st.stop()

lock_y = st.sidebar.checkbox("Lock Shared Y-Axis (Scientifically Accurate)", value=True)
show_mag = st.sidebar.checkbox("Overlay Vector Magnitude (|a|)", value=False)
offset = st.sidebar.slider("Window Start Offset (samples)", 0, 2500, 150, step=26)

st.title("Interactive Accelerometer Signal Explorer — WISDM v1.1")
st.caption("Machine Learning PT1 | Window Length: 51 samples (2.55 seconds at 20 Hz nominal rate)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Valid Samples", f"{len(df):,}")
c2.metric("Participants", f"{df['user_id'].nunique()}")
c3.metric("Activities", f"{df['activity'].nunique()}")
c4.metric("Nominal Sampling Rate", "20 Hz")

tab1, tab2, tab3, tab4 = st.tabs([
    "Compare Activities (Single User)",
    "Inter-Subject Variability",
    "Confusable Activity Pairs",
    "Dataset Overview",
])

# --- Tab 1: Compare Activities ---
with tab1:
    user_act_mat = pd.crosstab(df["user_id"], df["activity"])
    complete_users = user_act_mat[(user_act_mat >= 200).all(axis=1)].index.tolist()
    default_u = complete_users[0] if complete_users else int(df["user_id"].iloc[0])

    col_a, col_b = st.columns([1, 3])
    with col_a:
        all_users_sorted = sorted(df["user_id"].unique())
        sel_user = st.selectbox(
            "Participant (User ID):",
            all_users_sorted,
            index=all_users_sorted.index(default_u),
        )
    with col_b:
        sel_acts = st.multiselect("Activities to Compare:", ACTIVITIES_ORDER, default=ACTIVITIES_ORDER)

    windows = [(act, extract_window(df, sel_user, act, offset)) for act in sel_acts]
    windows = [(a, w) for a, w in windows if w is not None]

    if not windows:
        st.warning("Not enough samples for the selected user/activities.")
    else:
        cols = min(3, len(windows))
        rows = int(np.ceil(len(windows) / cols))
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"{a} (User {sel_user})" for a, _ in windows],
            shared_yaxes=lock_y,
        )
        cols_check = ["x", "y", "z", "mag"] if show_mag else ["x", "y", "z"]
        all_vals = np.concatenate([w[cols_check].values.ravel() for _, w in windows])
        yrange = [float(np.floor(all_vals.min()) - 1), float(np.ceil(all_vals.max()) + 1)]

        stats_list = []
        for idx, (act, w) in enumerate(windows):
            r, c = divmod(idx, cols)
            show_leg = idx == 0
            fig.add_trace(
                go.Scatter(x=TIME_AXIS, y=w["x"], name="X", line=dict(color="#ef4444"), showlegend=show_leg),
                row=r + 1,
                col=c + 1,
            )
            fig.add_trace(
                go.Scatter(x=TIME_AXIS, y=w["y"], name="Y", line=dict(color="#22c55e"), showlegend=show_leg),
                row=r + 1,
                col=c + 1,
            )
            fig.add_trace(
                go.Scatter(x=TIME_AXIS, y=w["z"], name="Z", line=dict(color="#3b82f6"), showlegend=show_leg),
                row=r + 1,
                col=c + 1,
            )
            if show_mag:
                fig.add_trace(
                    go.Scatter(
                        x=TIME_AXIS,
                        y=w["mag"],
                        name="|a|",
                        line=dict(color="#111827", dash="dot"),
                        showlegend=show_leg,
                    ),
                    row=r + 1,
                    col=c + 1,
                )
            if lock_y:
                fig.update_yaxes(range=yrange, row=r + 1, col=c + 1)
            fig.update_xaxes(title_text="Time (s)", row=r + 1, col=c + 1)
            fig.update_yaxes(title_text="Acc (m/s²)", row=r + 1, col=c + 1)
            stats_list.append(window_stats(act, w))

        fig.update_layout(height=360 * rows, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame(stats_list).round(3), use_container_width=True)

# --- Tab 2: Inter-Subject Variability ---
with tab2:
    c_act, c_us = st.columns([1, 3])
    with c_act:
        sel_act = st.selectbox("Activity:", ACTIVITIES_ORDER, index=0)
    avail_users = sorted(df[df["activity"] == sel_act]["user_id"].unique())
    with c_us:
        sel_users = st.multiselect("Participants (select at least 3):", avail_users, default=avail_users[:4])

    u_windows = [(f"User {u}", extract_window(df, u, sel_act, offset)) for u in sel_users]
    u_windows = [(lbl, w) for lbl, w in u_windows if w is not None]

    if u_windows:
        cols = min(2, len(u_windows))
        rows = int(np.ceil(len(u_windows) / cols))
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"{sel_act} — {lbl}" for lbl, _ in u_windows],
            shared_yaxes=lock_y,
        )
        all_vals = np.concatenate([w[["x", "y", "z"]].values.ravel() for _, w in u_windows])
        yrange = [float(np.floor(all_vals.min()) - 1), float(np.ceil(all_vals.max()) + 1)]

        stats_list = []
        for idx, (lbl, w) in enumerate(u_windows):
            r, c = divmod(idx, cols)
            show_leg = idx == 0
            fig.add_trace(
                go.Scatter(x=TIME_AXIS, y=w["x"], name="X", line=dict(color="#ef4444"), showlegend=show_leg),
                row=r + 1,
                col=c + 1,
            )
            fig.add_trace(
                go.Scatter(x=TIME_AXIS, y=w["y"], name="Y", line=dict(color="#22c55e"), showlegend=show_leg),
                row=r + 1,
                col=c + 1,
            )
            fig.add_trace(
                go.Scatter(x=TIME_AXIS, y=w["z"], name="Z", line=dict(color="#3b82f6"), showlegend=show_leg),
                row=r + 1,
                col=c + 1,
            )
            if lock_y:
                fig.update_yaxes(range=yrange, row=r + 1, col=c + 1)
            fig.update_xaxes(title_text="Time (s)", row=r + 1, col=c + 1)
            fig.update_yaxes(title_text="Acc (m/s²)", row=r + 1, col=c + 1)
            stats_list.append(window_stats(lbl, w))

        fig.update_layout(height=350 * rows, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame(stats_list).round(3), use_container_width=True)

# --- Tab 3: Pairwise Confusable Activities ---
with tab3:
    pair_choice = st.radio(
        "Select Confusable Pair:",
        ["Sitting vs Standing", "Walking vs Upstairs"],
        horizontal=True,
    )
    a1, a2 = pair_choice.split(" vs ")
    valid_pair_users = sorted(
        set(df[df["activity"] == a1]["user_id"].unique())
        & set(df[df["activity"] == a2]["user_id"].unique())
    )
    pair_u = st.selectbox("Common Participant:", valid_pair_users)
    w1 = extract_window(df, pair_u, a1, offset)
    w2 = extract_window(df, pair_u, a2, offset)
    if w1 is not None and w2 is not None:
        all_vals = np.concatenate([w1[["x", "y", "z"]].values.ravel(), w2[["x", "y", "z"]].values.ravel()])
        yrange = [float(np.floor(all_vals.min()) - 1), float(np.ceil(all_vals.max()) + 1)]
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[f"{a1} (User {pair_u})", f"{a2} (User {pair_u})"],
            shared_yaxes=lock_y,
        )
        for idx, (lbl, w) in enumerate([(a1, w1), (a2, w2)]):
            show_leg = idx == 0
            fig.add_trace(
                go.Scatter(x=TIME_AXIS, y=w["x"], name="X", line=dict(color="#ef4444"), showlegend=show_leg),
                row=1,
                col=idx + 1,
            )
            fig.add_trace(
                go.Scatter(x=TIME_AXIS, y=w["y"], name="Y", line=dict(color="#22c55e"), showlegend=show_leg),
                row=1,
                col=idx + 1,
            )
            fig.add_trace(
                go.Scatter(x=TIME_AXIS, y=w["z"], name="Z", line=dict(color="#3b82f6"), showlegend=show_leg),
                row=1,
                col=idx + 1,
            )
            if lock_y:
                fig.update_yaxes(range=yrange, row=1, col=idx + 1)
            fig.update_xaxes(title_text="Time (s)", row=1, col=idx + 1)
            fig.update_yaxes(title_text="Acc (m/s²)", row=1, col=idx + 1)
        fig.update_layout(height=420, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame([window_stats(a1, w1), window_stats(a2, w2)]).round(3), use_container_width=True)

# --- Tab 4: Dataset Overview ---
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        act_cnt = df["activity"].value_counts().reindex(ACTIVITIES_ORDER)
        fig1 = go.Figure(go.Bar(x=act_cnt.index, y=act_cnt.values, marker_color="#0ea5e9"))
        fig1.update_layout(
            title="Activity Distribution",
            xaxis_title="Activity",
            yaxis_title="Samples",
            template="plotly_white",
        )
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        usr_cnt = df["user_id"].value_counts().sort_index()
        fig2 = go.Figure(go.Bar(x=usr_cnt.index.astype(str), y=usr_cnt.values, marker_color="#6366f1"))
        fig2.update_layout(
            title="Samples per Participant",
            xaxis_title="User ID",
            yaxis_title="Samples",
            template="plotly_white",
        )
        st.plotly_chart(fig2, use_container_width=True)
