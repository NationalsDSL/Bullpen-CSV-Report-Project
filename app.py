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
        df = pd.read_csv(selected_row["Path"])
    else:
        df = pd.read_csv(io.BytesIO(selected_row["FileBytes"]))
    df.columns = df.columns.str.strip()

    required_cols = [
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
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required column(s) in selected file: {', '.join(missing_cols)}")
        st.stop()

    for col in [
        "PlateLocHeight",
        "PlateLocSide",
        "RelSpeed",
        "SpinRate",
        "Extension",
        "InducedVertBreak",
        "HorzBreak",
        "VertApprAngle",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Tilt"] = df["Tilt"].apply(normalize_tilt_clock)

    # ---------------------------
    # STRIKE ZONE LOGIC
    # ---------------------------
    df["Call"] = "Ball"
    df.loc[
        (df["PlateLocHeight"] >= 1.3)
        & (df["PlateLocHeight"] <= 3.7)
        & (df["PlateLocSide"] >= -0.83)
        & (df["PlateLocSide"] <= 0.83),
        "Call",
    ] = "Strike"

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
    summary = df_filtered.groupby("TaggedPitchType").agg(
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

    summary[["VeloAvg", "VeloMin", "VeloMax", "SpinRate", "Extension", "VertAvg", "HozAvg", "VAAAvg"]] = summary[
        ["VeloAvg", "VeloMin", "VeloMax", "SpinRate", "Extension", "VertAvg", "HozAvg", "VAAAvg"]
    ].round(2)

    # ---------------------------
    # DASHBOARD (RESPONSIVE)
    # ---------------------------
    st.subheader("Dashboard")
    dashboard_df = summary.rename(
        columns={
            "TaggedPitchType": "Pitch",
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
    )
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

    left_col, right_col = st.columns(2)
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
    st.dataframe(
        pitch_log_df.drop(columns=["PitchSelectId"]),
        use_container_width=True,
        hide_index=True,
    )
