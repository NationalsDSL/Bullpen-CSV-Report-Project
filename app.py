import os
import re
import tempfile
import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ModuleNotFoundError:
    go = None
    HAS_PLOTLY = False

st.set_page_config(page_title="Pitching Report Generator", layout="wide")

st.title("Pitching Report Generator")
st.caption("Load all CSV files from a folder, then select a player/date file.")

DEFAULT_CSV_FOLDER = os.getenv("BULLPEN_CSV_FOLDER", "Bullpens")


def extract_player_and_date(sample_df: pd.DataFrame, filename: str) -> tuple[str, str, pd.Timestamp]:
    player_name = "Unknown"
    date_value = "Unknown"
    date_sort = pd.NaT

    if "PlayerName" in sample_df.columns:
        players = sample_df["PlayerName"].dropna().astype(str).str.strip()
        if not players.empty and players.iloc[0] != "":
            player_name = players.iloc[0]

    if "Date" in sample_df.columns:
        dates = sample_df["Date"].dropna().astype(str).str.strip()
        if not dates.empty and dates.iloc[0] != "":
            parsed_date = pd.to_datetime(dates.iloc[0], errors="coerce")
            if pd.notna(parsed_date):
                date_sort = parsed_date.normalize()
                date_value = date_sort.strftime("%Y-%m-%d")
            else:
                date_value = dates.iloc[0]

    if date_value == "Unknown":
        # Fallback: file names like 2_11_2026_player_name_...
        m = re.match(r"^(\d{1,2})_(\d{1,2})_(\d{4})_", filename)
        if m:
            month, day, year = m.groups()
            inferred = pd.to_datetime(
                f"{year}-{int(month):02d}-{int(day):02d}",
                errors="coerce",
            )
            if pd.notna(inferred):
                date_sort = inferred.normalize()
                date_value = date_sort.strftime("%Y-%m-%d")

    return str(player_name).strip() or "Unknown", str(date_value).strip() or "Unknown", date_sort


def collect_csv_metadata(folder_path: str) -> pd.DataFrame:
    records = []

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(".csv"):
            continue

        path = os.path.join(folder_path, filename)
        player_name = "Unknown"
        date_value = "Unknown"
        date_sort = pd.NaT

        try:
            sample_df = pd.read_csv(path, nrows=200)
            sample_df.columns = sample_df.columns.str.strip()
            player_name, date_value, date_sort = extract_player_and_date(sample_df, filename)

        except Exception:
            player_name = "Unreadable"
            date_value = "Unreadable"

        records.append(
            {
                "File": filename,
                "PlayerName": str(player_name).strip() or "Unknown",
                "Date": str(date_value).strip() or "Unknown",
                "DateSort": date_sort,
                "Path": path,
                "Modified": os.path.getmtime(path),
            }
        )

    return pd.DataFrame(records)


def collect_uploaded_csv_metadata(uploaded_files: list) -> pd.DataFrame:
    records = []
    for idx, uploaded_file in enumerate(uploaded_files):
        filename = uploaded_file.name
        file_bytes = uploaded_file.getvalue()
        player_name = "Unknown"
        date_value = "Unknown"
        date_sort = pd.NaT

        try:
            sample_df = pd.read_csv(io.BytesIO(file_bytes), nrows=200)
            sample_df.columns = sample_df.columns.str.strip()
            player_name, date_value, date_sort = extract_player_and_date(sample_df, filename)
        except Exception:
            player_name = "Unreadable"
            date_value = "Unreadable"

        records.append(
            {
                "File": filename,
                "PlayerName": str(player_name).strip() or "Unknown",
                "Date": str(date_value).strip() or "Unknown",
                "DateSort": date_sort,
                "FileBytes": file_bytes,
                "Modified": idx,
            }
        )

    return pd.DataFrame(records)


def sanitize_filename(value: str) -> str:
    cleaned = "".join(ch for ch in str(value) if ch.isalnum() or ch in ("-", "_", " ")).strip()
    return cleaned.replace(" ", "_") if cleaned else "Unknown"


def mode_or_unknown(series: pd.Series) -> str:
    clean_values = series.dropna().astype(str).str.strip()
    clean_values = clean_values[clean_values != ""]
    if clean_values.empty:
        return "Unknown"
    mode_values = clean_values.mode()
    if mode_values.empty:
        return "Unknown"
    return str(mode_values.iloc[0])


def normalize_tilt_clock(value: object) -> str:
    if pd.isna(value):
        return "Unknown"

    raw = str(value).strip()
    if raw == "":
        return "Unknown"

    # Handles strings already in clock style or time-like values (e.g. 13:30:00 -> 1:30).
    time_match = re.search(r"(\d{1,2}):(\d{2})", raw)
    if time_match:
        hour = int(time_match.group(1)) % 12
        minute = int(time_match.group(2))
        if hour == 0:
            hour = 12
        return f"{hour}:{minute:02d}"

    parsed_dt = pd.to_datetime(raw, errors="coerce")
    if pd.notna(parsed_dt):
        hour = parsed_dt.hour % 12
        minute = parsed_dt.minute
        if hour == 0:
            hour = 12
        return f"{hour}:{minute:02d}"

    return raw


REQUIRED_COLS = [
    "TaggedPitchType",
    "PlateLocHeight",
    "PlateLocSide",
    "RelSpeed",
    "SpinRate",
    "Tilt",
    "Extension",
    "InducedVertBreak",
    "HorzBreak",
    "VertApprAngle",
]

NUMERIC_COLS = [
    "PlateLocHeight",
    "PlateLocSide",
    "RelSpeed",
    "SpinRate",
    "Extension",
    "InducedVertBreak",
    "HorzBreak",
    "VertApprAngle",
]


def prepare_pitch_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = df.columns.str.strip()

    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s): {', '.join(missing_cols)}")

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Tilt"] = df["Tilt"].apply(normalize_tilt_clock)
    df["TaggedPitchType"] = df["TaggedPitchType"].fillna("Unknown").astype(str).str.strip()

    df["Call"] = "Ball"
    df.loc[
        (df["PlateLocHeight"] >= 1.3)
        & (df["PlateLocHeight"] <= 3.7)
        & (df["PlateLocSide"] >= -0.83)
        & (df["PlateLocSide"] <= 0.83),
        "Call",
    ] = "Strike"
    return df


def build_summary(df_source: pd.DataFrame) -> pd.DataFrame:
    summary = df_source.groupby("TaggedPitchType").agg(
        Pitches=("TaggedPitchType", "count"),
        Strikes=("Call", lambda x: (x == "Strike").sum()),
        Balls=("Call", lambda x: (x == "Ball").sum()),
        VeloAvg=("RelSpeed", "mean"),
        VeloMin=("RelSpeed", "min"),
        VeloMax=("RelSpeed", "max"),
        SpinRate=("SpinRate", "mean"),
        Tilt=("Tilt", mode_or_unknown),
        Extension=("Extension", "mean"),
        VertAvg=("InducedVertBreak", "mean"),
        HozAvg=("HorzBreak", "mean"),
        VAAAvg=("VertApprAngle", "mean"),
    ).reset_index()

    summary[
        ["VeloAvg", "VeloMin", "VeloMax", "SpinRate", "Extension", "VertAvg", "HozAvg", "VAAAvg"]
    ] = summary[
        ["VeloAvg", "VeloMin", "VeloMax", "SpinRate", "Extension", "VertAvg", "HozAvg", "VAAAvg"]
    ].round(1)
    return summary


folder_path = DEFAULT_CSV_FOLDER

if folder_path.strip() != "":
    normalized_folder = folder_path.strip().strip('"')
    source_mode = "local"
    metadata_df = pd.DataFrame()

    if normalized_folder and os.path.isdir(normalized_folder):
        metadata_df = collect_csv_metadata(normalized_folder)

    if metadata_df.empty:
        source_mode = "upload"
        st.info("Upload one or more CSV files to use the app online.")
        uploaded_files = st.file_uploader(
            "Upload Trackman CSV files",
            type=["csv"],
            accept_multiple_files=True,
        )
        if not uploaded_files:
            st.stop()
        metadata_df = collect_uploaded_csv_metadata(uploaded_files)

    if metadata_df.empty:
        st.warning("No readable CSV files available.")
        st.stop()

    player_options = sorted(metadata_df["PlayerName"].astype(str).unique().tolist(), key=str.lower)
    if not player_options:
        st.warning("No players found in CSV files.")
        st.stop()

    left_filter_col, right_filter_col = st.columns(2)
    with left_filter_col:
        selected_player = st.selectbox("PlayerName", options=player_options)

    filtered_by_player = metadata_df[metadata_df["PlayerName"] == selected_player].copy()
    filtered_by_player = filtered_by_player.sort_values(
        by=["DateSort", "Date", "Modified", "File"],
        ascending=[False, False, False, True],
        na_position="last",
    )

    date_options = filtered_by_player["Date"].drop_duplicates().tolist()
    if not date_options:
        st.warning("No dates found for selected player.")
        st.stop()

    with right_filter_col:
        selected_date = st.selectbox("Date", options=date_options)

    matched_rows = filtered_by_player[filtered_by_player["Date"] == selected_date].copy()
    if matched_rows.empty:
        st.warning("No CSV found for selected player/date.")
        st.stop()

    # If duplicates exist for same player/date, use most recently modified file.
    selected_row = matched_rows.sort_values(by=["Modified", "File"], ascending=[False, True]).iloc[0]

    player_name = selected_row["PlayerName"]
    report_date = selected_row["Date"]

    st.info(f"Selected Player: {player_name} | Date: {report_date}")

    if source_mode == "local":
        raw_df = pd.read_csv(selected_row["Path"])
    else:
        raw_df = pd.read_csv(io.BytesIO(selected_row["FileBytes"]))

    try:
        df = prepare_pitch_dataframe(raw_df)
    except ValueError as exc:
        st.error(f"{exc} in selected file.")
        st.stop()

    df["TaggedPitchType"] = df["TaggedPitchType"].fillna("Unknown").astype(str).str.strip()
    available_pitch_types = sorted(df["TaggedPitchType"].unique().tolist(), key=str.lower)
    selected_pitch_filter = st.selectbox(
        "Pitch Type",
        options=["All Pitches"] + available_pitch_types,
    )

    if selected_pitch_filter == "All Pitches":
        df_filtered = df.copy()
    else:
        df_filtered = df[df["TaggedPitchType"] == selected_pitch_filter].copy()

    if df_filtered.empty:
        st.warning("No pitches found for the selected pitch type.")
        st.stop()

    # ---------------------------
    # SUMMARY TABLE
    # ---------------------------
    summary = build_summary(df_filtered)

    # ---------------------------
    # DASHBOARD (RESPONSIVE)
    # ---------------------------
    st.subheader("Dashboard")
    compare_button_key = "compare_all_metrics_prev"
    if compare_button_key not in st.session_state:
        st.session_state[compare_button_key] = False

    if st.button("Compare with previous data", use_container_width=False):
        st.session_state[compare_button_key] = not st.session_state[compare_button_key]

    show_comparison = st.session_state[compare_button_key]
    if show_comparison:
        st.caption("Comparison mode: ON")
    else:
        st.caption("Comparison mode: OFF")

    summary_view = summary.copy()
    comparison_ready = False
    comparison_columns = [
        "VeloAvg",
        "VeloMin",
        "VeloMax",
        "SpinRate",
        "Extension",
        "VertAvg",
        "HozAvg",
        "VAAAvg",
    ]
    comparison_tolerance = {
        "VeloAvg": 0.1,
        "VeloMin": 0.1,
        "VeloMax": 0.1,
        "SpinRate": 0.1,
        "Extension": 0.1,
        "VertAvg": 0.1,
        "HozAvg": 0.1,
        "VAAAvg": 0.1,
    }

    if show_comparison:
        player_history = metadata_df[metadata_df["PlayerName"] == player_name].copy()
        if source_mode == "local":
            player_history = player_history[player_history["Path"] != selected_row["Path"]]
        else:
            player_history = player_history[player_history.index != selected_row.name]

        selected_date_sort = selected_row["DateSort"] if "DateSort" in selected_row else pd.NaT
        if pd.notna(selected_date_sort):
            previous_by_date = player_history[
                player_history["DateSort"].notna() & (player_history["DateSort"] < selected_date_sort)
            ]
            if not previous_by_date.empty:
                player_history = previous_by_date

        previous_frames = []
        for _, history_row in player_history.iterrows():
            try:
                if source_mode == "local":
                    previous_raw = pd.read_csv(history_row["Path"])
                else:
                    previous_raw = pd.read_csv(io.BytesIO(history_row["FileBytes"]))
                previous_df = prepare_pitch_dataframe(previous_raw)
                if selected_pitch_filter != "All Pitches":
                    previous_df = previous_df[previous_df["TaggedPitchType"] == selected_pitch_filter]
                if not previous_df.empty:
                    previous_frames.append(previous_df)
            except Exception:
                continue

        if previous_frames:
            previous_all = pd.concat(previous_frames, ignore_index=True)
            previous_summary = build_summary(previous_all)[["TaggedPitchType"] + comparison_columns].rename(
                columns={col: f"{col}_Prev" for col in comparison_columns}
            )
            summary_view = summary_view.merge(previous_summary, on="TaggedPitchType", how="left")
            comparison_ready = summary_view[[f"{col}_Prev" for col in comparison_columns]].notna().any().any()
        else:
            st.info("No previous data found for comparison.")

    display_columns = {
        "Pitches": "Pitches",
        "Strikes": "Strikes",
        "Balls": "Balls",
        "VeloAvg": "Velo Avg (RelSpeed)",
        "VeloMin": "Velo Min (RelSpeed)",
        "VeloMax": "Velo Max (RelSpeed)",
        "SpinRate": "SpinRate",
        "Tilt": "Tilt",
        "Extension": "Extension",
        "VertAvg": "Vert. (InducedVertBreak)",
        "HozAvg": "Hoz (HorzBreak)",
        "VAAAvg": "VAA (VertApprAngle)",
    }
    dashboard_df = summary_view[list(display_columns.keys()) + ["TaggedPitchType"]].rename(
        columns={
            "TaggedPitchType": "Pitch",
            **display_columns,
        }
    )
    dashboard_df = dashboard_df[
        [
            "Pitch",
            "Pitches",
            "Balls",
            "Strikes",
            "Velo Avg (RelSpeed)",
            "Velo Min (RelSpeed)",
            "Velo Max (RelSpeed)",
            "SpinRate",
            "Tilt",
            "Extension",
            "Vert. (InducedVertBreak)",
            "Hoz (HorzBreak)",
            "VAA (VertApprAngle)",
        ]
    ]
    dashboard_float_cols = dashboard_df.select_dtypes(include=["float32", "float64"]).columns
    if len(dashboard_float_cols) > 0:
        dashboard_df[dashboard_float_cols] = dashboard_df[dashboard_float_cols].round(1)

    if show_comparison and comparison_ready:
        style_df = pd.DataFrame("", index=dashboard_df.index, columns=dashboard_df.columns)
        for idx in summary_view.index:
            for source_col in comparison_columns:
                prev_col = f"{source_col}_Prev"
                if prev_col not in summary_view.columns:
                    continue
                current_value = summary_view.at[idx, source_col]
                previous_value = summary_view.at[idx, prev_col]
                if pd.isna(current_value) or pd.isna(previous_value):
                    continue
                tolerance = comparison_tolerance.get(source_col, 0.1)
                display_col = display_columns[source_col]
                if current_value > previous_value + tolerance:
                    style_df.at[idx, display_col] = "background-color: #f4c7c3; color: #000000; font-weight: 700;"
                elif current_value < previous_value - tolerance:
                    style_df.at[idx, display_col] = "background-color: #cfe2f3; color: #000000; font-weight: 700;"

        styled_format = {col: "{:.1f}" for col in dashboard_float_cols}
        styled_dashboard = dashboard_df.style.apply(lambda _: style_df, axis=None).format(styled_format)
        st.dataframe(styled_dashboard, use_container_width=True, hide_index=True)
        st.caption(
            "Comparison applied to pitch metrics only (not Pitches/Balls/Strikes). "
            "Higher than previous = red cell, lower than previous = blue cell, similar = normal."
        )
    else:
        st.dataframe(dashboard_df, use_container_width=True, hide_index=True)

    pitch_type_series = df["TaggedPitchType"].fillna("Unknown").astype(str).str.strip()
    pitch_types_all = sorted(pitch_type_series.unique().tolist(), key=str.lower)

    def is_fastball(pitch_name: str) -> bool:
        normalized = str(pitch_name).lower()
        fastball_tokens = ["recta", "fastball", "four-seam", "fourseam", "4-seam", "4 seam", "ff"]
        return any(token in normalized for token in fastball_tokens)

    other_pitch_colors = [
        "#2ca02c",
        "#d62728",
        "#ff7f0e",
        "#9467bd",
        "#17becf",
        "#8c564b",
        "#e377c2",
        "#bcbd22",
        "#7f7f7f",
    ]
    pitch_color_map = {}
    other_idx = 0
    for pitch_type in pitch_types_all:
        if is_fastball(pitch_type):
            pitch_color_map[pitch_type] = "#1f77b4"  # Blue for fastball/recta
        else:
            pitch_color_map[pitch_type] = other_pitch_colors[other_idx % len(other_pitch_colors)]
            other_idx += 1

    # ---------------------------
    # STRIKE ZONE PLOT
    # ---------------------------
    zone_path = os.path.join(tempfile.gettempdir(), "zone.png")
    fig_zone, ax_zone = plt.subplots(figsize=(6, 5))
    zone_df = df_filtered[["PlateLocSide", "PlateLocHeight", "TaggedPitchType"]].copy()
    zone_df = zone_df.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    zone_df["TaggedPitchType"] = zone_df["TaggedPitchType"].fillna("Unknown").astype(str).str.strip()

    zone_pitch_types = sorted(zone_df["TaggedPitchType"].unique().tolist(), key=str.lower)
    for pitch_type in zone_pitch_types:
        pitch_group = zone_df[zone_df["TaggedPitchType"] == pitch_type]
        if pitch_group.empty:
            continue
        ax_zone.scatter(
            pitch_group["PlateLocSide"],
            pitch_group["PlateLocHeight"],
            s=30,
            color=pitch_color_map[pitch_type],
            alpha=0.9,
            edgecolors="white",
            linewidths=0.3,
            label=pitch_type,
        )

    # Strike zone dimensions from code logic
    ax_zone.plot([-0.83, 0.83], [1.3, 1.3], color="black", linewidth=1.2)
    ax_zone.plot([-0.83, 0.83], [3.7, 3.7], color="black", linewidth=1.2)
    ax_zone.plot([-0.83, -0.83], [1.3, 3.7], color="black", linewidth=1.2)
    ax_zone.plot([0.83, 0.83], [1.3, 3.7], color="black", linewidth=1.2)
    ax_zone.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax_zone.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax_zone.set_xlim(-3, 3)
    ax_zone.set_ylim(0, 5)
    ax_zone.set_xticks(list(range(-3, 4, 1)))
    ax_zone.set_yticks(list(range(0, 6, 1)))
    ax_zone.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax_zone.set_xlabel("PlateLocSide")
    ax_zone.set_ylabel("PlateLocHeight")
    ax_zone.set_title("Strike Zone Plot")
    ax_zone.legend(loc="upper right", fontsize=8, frameon=True)
    fig_zone.tight_layout()
    fig_zone.savefig(zone_path, dpi=150)

    if HAS_PLOTLY:
        # Interactive strike zone plot (web)
        zone_plot_df = zone_df.copy()
        if "PitchNo" in df_filtered.columns:
            zone_plot_df = zone_plot_df.join(df_filtered[["PitchNo", "RelSpeed", "SpinRate"]])
            zone_plot_df["PitchNo"] = zone_plot_df["PitchNo"].fillna("").astype(str)
        else:
            zone_plot_df = zone_plot_df.join(df_filtered[["RelSpeed", "SpinRate"]])
            zone_plot_df["PitchNo"] = ""
        zone_plot_df["RelSpeedDisplay"] = zone_plot_df["RelSpeed"].apply(
            lambda v: f"{v:.1f}" if pd.notna(v) else "N/A"
        )
        zone_plot_df["SpinRateDisplay"] = zone_plot_df["SpinRate"].apply(
            lambda v: f"{v:.0f}" if pd.notna(v) else "N/A"
        )

        fig_zone_interactive = go.Figure()
        for pitch_type in zone_pitch_types:
            pitch_group = zone_plot_df[zone_plot_df["TaggedPitchType"] == pitch_type]
            if pitch_group.empty:
                continue
            fig_zone_interactive.add_trace(
                go.Scatter(
                    x=pitch_group["PlateLocSide"],
                    y=pitch_group["PlateLocHeight"],
                    mode="markers",
                    name=pitch_type,
                    marker=dict(size=9, color=pitch_color_map.get(pitch_type, "#7f7f7f")),
                    customdata=pitch_group[["PitchNo", "RelSpeedDisplay", "SpinRateDisplay"]],
                    hovertemplate=(
                        "Pitch Type: %{fullData.name}<br>"
                        "Pitch No: %{customdata[0]}<br>"
                        "RelSpeed: %{customdata[1]}<br>"
                        "SpinRate: %{customdata[2]}<br>"
                        "Horizontal: %{x:.2f}<br>"
                        "Vertical: %{y:.2f}<extra></extra>"
                    ),
                )
            )

        # Strike zone and axis reference lines
        zone_shapes = [
            dict(type="line", x0=-0.83, y0=1.3, x1=0.83, y1=1.3, line=dict(color="black", width=2)),
            dict(type="line", x0=-0.83, y0=3.7, x1=0.83, y1=3.7, line=dict(color="black", width=2)),
            dict(type="line", x0=-0.83, y0=1.3, x1=-0.83, y1=3.7, line=dict(color="black", width=2)),
            dict(type="line", x0=0.83, y0=1.3, x1=0.83, y1=3.7, line=dict(color="black", width=2)),
            dict(type="line", x0=-3, y0=0, x1=3, y1=0, line=dict(color="gray", width=1, dash="dash")),
            dict(type="line", x0=0, y0=0, x1=0, y1=5, line=dict(color="gray", width=1, dash="dash")),
        ]
        fig_zone_interactive.update_layout(
            title="Strike Zone Plot",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            legend_title_text="Pitch Type",
            shapes=zone_shapes,
        )
        fig_zone_interactive.update_xaxes(
            title_text="PlateLocSide",
            range=[-3, 3],
            dtick=1,
            showgrid=True,
            gridcolor="lightgray",
        )
        fig_zone_interactive.update_yaxes(
            title_text="PlateLocHeight",
            range=[0, 5],
            dtick=1,
            showgrid=True,
            gridcolor="lightgray",
        )

    # ---------------------------
    # BREAK PLOT
    # ---------------------------
    break_path = os.path.join(tempfile.gettempdir(), "break.png")
    fig_break, ax_break = plt.subplots(figsize=(5, 5))
    break_df = df_filtered[["HorzBreak", "InducedVertBreak", "TaggedPitchType"]].copy()
    break_df = break_df.dropna(subset=["HorzBreak", "InducedVertBreak"])
    break_df["TaggedPitchType"] = break_df["TaggedPitchType"].fillna("Unknown").astype(str)

    pitch_types = sorted(break_df["TaggedPitchType"].unique().tolist(), key=str.lower)
    for pitch_type in pitch_types:
        pitch_group = break_df[break_df["TaggedPitchType"] == pitch_type]
        ax_break.scatter(
            pitch_group["HorzBreak"],
            pitch_group["InducedVertBreak"],
            s=28,
            color=pitch_color_map.get(pitch_type, "#7f7f7f"),
            alpha=0.9,
            edgecolors="white",
            linewidths=0.3,
            label=pitch_type,
        )

    ax_break.set_xlim(-25, 25)
    ax_break.set_ylim(-25, 25)
    ax_break.set_xticks(list(range(-25, 26, 5)))
    ax_break.set_yticks(list(range(-25, 26, 5)))
    ax_break.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax_break.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax_break.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax_break.set_aspect("equal", adjustable="box")
    ax_break.set_xlabel("Horizontal Break")
    ax_break.set_ylabel("Vertical Break")
    ax_break.set_title("Break Plot")
    ax_break.legend(loc="upper right", fontsize=8, frameon=True)
    fig_break.tight_layout()
    fig_break.savefig(break_path, dpi=150)

    if HAS_PLOTLY:
        # Interactive break plot (web)
        break_plot_df = break_df.copy()
        if "PitchNo" in df_filtered.columns:
            break_plot_df = break_plot_df.join(df_filtered[["PitchNo", "RelSpeed", "SpinRate"]])
            break_plot_df["PitchNo"] = break_plot_df["PitchNo"].fillna("").astype(str)
        else:
            break_plot_df = break_plot_df.join(df_filtered[["RelSpeed", "SpinRate"]])
            break_plot_df["PitchNo"] = ""
        break_plot_df["RelSpeedDisplay"] = break_plot_df["RelSpeed"].apply(
            lambda v: f"{v:.1f}" if pd.notna(v) else "N/A"
        )
        break_plot_df["SpinRateDisplay"] = break_plot_df["SpinRate"].apply(
            lambda v: f"{v:.0f}" if pd.notna(v) else "N/A"
        )

        fig_break_interactive = go.Figure()
        for pitch_type in pitch_types:
            pitch_group = break_plot_df[break_plot_df["TaggedPitchType"] == pitch_type]
            if pitch_group.empty:
                continue
            fig_break_interactive.add_trace(
                go.Scatter(
                    x=pitch_group["HorzBreak"],
                    y=pitch_group["InducedVertBreak"],
                    mode="markers",
                    name=pitch_type,
                    marker=dict(size=9, color=pitch_color_map.get(pitch_type, "#7f7f7f")),
                    customdata=pitch_group[["PitchNo", "RelSpeedDisplay", "SpinRateDisplay"]],
                    hovertemplate=(
                        "Pitch Type: %{fullData.name}<br>"
                        "Pitch No: %{customdata[0]}<br>"
                        "RelSpeed: %{customdata[1]}<br>"
                        "SpinRate: %{customdata[2]}<br>"
                        "Horizontal Break: %{x:.2f}<br>"
                        "Vertical Break: %{y:.2f}<extra></extra>"
                    ),
                )
            )

        break_shapes = [
            dict(type="line", x0=-25, y0=0, x1=25, y1=0, line=dict(color="gray", width=1, dash="dash")),
            dict(type="line", x0=0, y0=-25, x1=0, y1=25, line=dict(color="gray", width=1, dash="dash")),
        ]
        fig_break_interactive.update_layout(
            title="Break Plot",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            legend_title_text="Pitch Type",
            shapes=break_shapes,
        )
        fig_break_interactive.update_xaxes(
            title_text="Horizontal Break",
            range=[-25, 25],
            dtick=5,
            showgrid=True,
            gridcolor="lightgray",
        )
        fig_break_interactive.update_yaxes(
            title_text="Vertical Break",
            range=[-25, 25],
            dtick=5,
            showgrid=True,
            gridcolor="lightgray",
            scaleanchor="x",
            scaleratio=1,
        )

    left_col, right_col = st.columns(2)
    if HAS_PLOTLY:
        with left_col:
            st.plotly_chart(fig_zone_interactive, use_container_width=True)
        with right_col:
            st.plotly_chart(fig_break_interactive, use_container_width=True)
    else:
        st.warning("Plotly is not installed. Showing static charts. Install with: pip install plotly")
        with left_col:
            st.pyplot(fig_zone, use_container_width=True)
        with right_col:
            st.pyplot(fig_break, use_container_width=True)

    plt.close(fig_zone)
    plt.close(fig_break)

    # ---------------------------
    # GENERATE PDF
    # ---------------------------
    pdf_path = os.path.join(tempfile.gettempdir(), "report.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=landscape(letter),
        leftMargin=24,
        rightMargin=24,
        topMargin=24,
        bottomMargin=24,
    )
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph(f"Pitching Report - {player_name} - {report_date}", styles["Title"]))
    elements.append(Spacer(1, 20))

    table_data = [
        [
            "Pitch",
            "Pitches",
            "Balls",
            "Strikes",
            "Velo Avg (RelSpeed)",
            "Velo Min (RelSpeed)",
            "Velo Max (RelSpeed)",
            "SpinRate",
            "Tilt",
            "Extension",
            "Vert. (InducedVertBreak)",
            "Hoz (HorzBreak)",
            "VAA (VertApprAngle)",
        ]
    ]

    for _, row in summary.iterrows():
        table_data.append(
            [
                row["TaggedPitchType"],
                int(row["Pitches"]),
                int(row["Balls"]),
                int(row["Strikes"]),
                row["VeloAvg"],
                row["VeloMin"],
                row["VeloMax"],
                row["SpinRate"],
                row["Tilt"],
                row["Extension"],
                row["VertAvg"],
                row["HozAvg"],
                row["VAAAvg"],
            ]
        )

    col_widths = [58, 38, 34, 40, 54, 54, 54, 52, 44, 58, 62, 52, 56]
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )

    elements.append(table)
    elements.append(Spacer(1, 22))
    elements.append(Image(zone_path, width=5.8 * inch, height=4.2 * inch))
    elements.append(Spacer(1, 14))
    elements.append(Image(break_path, width=5.8 * inch, height=4.2 * inch))

    doc.build(elements)

    # ---------------------------
    # DOWNLOAD BUTTON
    # ---------------------------
    safe_player = sanitize_filename(player_name)
    safe_date = sanitize_filename(report_date)

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="Download PDF Report",
            data=f,
            file_name=f"{safe_player}_{safe_date}_Pitch_Report.pdf",
            mime="application/pdf",
        )

    # ---------------------------
    # INDIVIDUAL PITCHES
    # ---------------------------
    st.subheader("Individual Pitches")
    pitch_log_df = df_filtered.reset_index(drop=True).copy()
    pitch_log_df.insert(0, "PitchSelectId", range(1, len(pitch_log_df) + 1))
    if "PitchNo" in pitch_log_df.columns:
        pitch_number_series = pitch_log_df["PitchNo"]
    else:
        pitch_number_series = pitch_log_df["PitchSelectId"]
        pitch_log_df["PitchNo"] = pitch_number_series

    st.caption(
        f"Showing {len(pitch_log_df)} pitches for filter: "
        f"{selected_pitch_filter if selected_pitch_filter else 'All Pitches'}"
    )
    pitch_table_df = pitch_log_df.drop(columns=["PitchSelectId"]).copy()
    float_cols = pitch_table_df.select_dtypes(include=["float32", "float64"]).columns
    if len(float_cols) > 0:
        pitch_table_df[float_cols] = pitch_table_df[float_cols].round(1)
    st.dataframe(
        pitch_table_df,
        use_container_width=True,
        hide_index=True,
    )
