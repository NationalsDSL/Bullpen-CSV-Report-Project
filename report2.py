import pandas as pd
import tkinter as tk
from tkinter import filedialog
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image # pyright: ignore[reportMissingModuleSource]
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
import matplotlib.pyplot as plt
import os
import sys

# ==============================
# SELECT CSV FILE (FILE PICKER)
# ==============================

root = tk.Tk()
root.withdraw()  # Hide main window

csv_path = filedialog.askopenfilename(
    title="Select TrackMan CSV File",
    filetypes=[("CSV Files", "*.csv")]
)

if not csv_path:
    print("No file selected.")
    sys.exit(0)

print("Selected file:", csv_path)

# ==============================
# LOAD CSV
# ==============================

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# ==============================
# COLUMN NAMES (Adjust if needed)
# ==============================

velo_col = "RelSpeed"
spin_col = "SpinRate"
pitch_col = "TaggedPitchType"
result_col = "PitchCall"
plate_side_col = "PlateLocSide"
plate_height_col = "PlateLocHeight"

# ==============================
# CALCULATIONS
# ==============================

required_columns = [
    velo_col,
    spin_col,
    pitch_col,
    result_col,
    plate_side_col,
    plate_height_col,
]

missing = [col for col in required_columns if col not in df.columns]
if missing:
    print(f"Missing required column(s): {', '.join(missing)}")
    sys.exit(1)

for numeric_col in [velo_col, spin_col, plate_side_col, plate_height_col]:
    df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

df[pitch_col] = df[pitch_col].fillna("Unknown")

total_pitches = len(df)
if total_pitches == 0:
    print("The selected CSV is empty.")
    sys.exit(1)

summary = df.groupby(pitch_col).agg(
    Pitches=(pitch_col, "count"),
    AvgVelo=(velo_col, "mean"),
    MaxVelo=(velo_col, "max"),
    AvgSpin=(spin_col, "mean"),
)

summary["Usage%"] = (summary["Pitches"] / total_pitches * 100).round(1)

# Whiff %
whiffs = df[df[result_col] == "StrikeSwinging"]
whiff_rate = (
    whiffs.groupby(pitch_col).size()
    .div(df.groupby(pitch_col).size())
    .fillna(0)
)
summary["Whiff%"] = (whiff_rate * 100).round(1)

summary = summary.fillna(0).reset_index()

summary["AvgVelo"] = summary["AvgVelo"].round(1)
summary["MaxVelo"] = summary["MaxVelo"].round(1)
summary["AvgSpin"] = summary["AvgSpin"].round(0)

# ==============================
# CREATE STRIKE ZONE CHART
# ==============================

plt.figure(figsize=(4, 4))
plot_df = df[[plate_side_col, plate_height_col]].dropna()
plt.scatter(plot_df[plate_side_col], plot_df[plate_height_col])
plt.axhline(1.5)
plt.axhline(3.5)
plt.axvline(-0.83)
plt.axvline(0.83)
plt.title("Pitch Location")
plt.xlabel("Plate Side")
plt.ylabel("Plate Height")
plt.tight_layout()

# Save PDF in same folder as CSV
output_folder = os.path.dirname(csv_path)
zone_chart_path = os.path.join(output_folder, "zone_plot.png")
plt.savefig(zone_chart_path)
plt.close()

# ==============================
# CREATE PDF
# ==============================

pdf_file = os.path.join(output_folder, "Pitching_Report.pdf")

doc = SimpleDocTemplate(pdf_file, pagesize=letter)
elements = []
styles = getSampleStyleSheet()

# Title
elements.append(Paragraph("Pitching Game Report", styles["Heading1"]))
elements.append(Spacer(1, 0.3 * inch))

# Total Pitches
elements.append(Paragraph(f"Total Pitches: {total_pitches}", styles["Normal"]))
elements.append(Spacer(1, 0.3 * inch))

# Table
table_data = [["Pitch", "Pitches", "Usage%", "Avg Velo", "Max Velo", "Avg Spin", "Whiff%"]]

for _, row in summary.iterrows():
    table_data.append([
        row[pitch_col],
        row["Pitches"],
        row["Usage%"],
        row["AvgVelo"],
        row["MaxVelo"],
        row["AvgSpin"],
        row["Whiff%"]
    ])

table = Table(table_data, repeatRows=1)

table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
]))

elements.append(table)
elements.append(Spacer(1, 0.5 * inch))

# Add Chart
elements.append(Paragraph("Pitch Location Chart", styles["Heading2"]))
elements.append(Spacer(1, 0.2 * inch))
elements.append(Image(zone_chart_path, width=4*inch, height=4*inch))

# Build PDF
doc.build(elements)

if os.path.exists(zone_chart_path):
    os.remove(zone_chart_path)

print("Report Generated Successfully!")
print("Saved at:", pdf_file)
