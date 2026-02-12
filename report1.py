import pandas as pd
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import utils
from reportlab.lib.units import inch
from reportlab.platypus import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import PageBreak
import matplotlib.pyplot as plt
import os

# ==============================
# LOAD CSV
# ==============================

csv_path = "2_11_2026_abreu_juan_portable_tm_pitches.csv"
df = pd.read_csv(csv_path)

# ==============================
# BASIC CLEANING
# ==============================

df.columns = df.columns.str.strip()

# Adjust column names if needed depending on your CSV
velo_col = "RelSpeed"
spin_col = "SpinRate"
pitch_col = "TaggedPitchType"
result_col = "PitchCall"
zone_col = "PlateLocHeight"

# ==============================
# CALCULATIONS
# ==============================

total_pitches = len(df)

summary = df.groupby(pitch_col).agg(
    Pitches=(pitch_col, "count"),
    AvgVelo=(velo_col, "mean"),
    MaxVelo=(velo_col, "max"),
    AvgSpin=(spin_col, "mean"),
)

summary["Usage%"] = (summary["Pitches"] / total_pitches * 100).round(1)

# Whiff %
whiffs = df[df[result_col] == "StrikeSwinging"]
whiff_rate = whiffs.groupby(pitch_col).size() / df.groupby(pitch_col).size()
summary["Whiff%"] = (whiff_rate * 100).round(1)

summary = summary.fillna(0)
summary = summary.reset_index()

# Round values
summary["AvgVelo"] = summary["AvgVelo"].round(1)
summary["MaxVelo"] = summary["MaxVelo"].round(1)
summary["AvgSpin"] = summary["AvgSpin"].round(0)

# ==============================
# CREATE STRIKE ZONE CHART
# ==============================

plt.figure(figsize=(4,4))
plt.scatter(df["PlateLocSide"], df["PlateLocHeight"])
plt.axhline(1.5)
plt.axhline(3.5)
plt.axvline(-0.83)
plt.axvline(0.83)
plt.title("Pitch Location")
plt.xlabel("Plate Side")
plt.ylabel("Plate Height")
plt.tight_layout()

zone_chart_path = "zone_plot.png"
plt.savefig(zone_chart_path)
plt.close()

# ==============================
# CREATE PDF
# ==============================

pdf_file = "Pitching_Report.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=letter)
elements = []

styles = getSampleStyleSheet()

title_style = styles["Heading1"]
normal_style = styles["Normal"]

# Title
elements.append(Paragraph("Pitching Game Report", title_style))
elements.append(Spacer(1, 0.3 * inch))

# Total pitches
elements.append(Paragraph(f"Total Pitches: {total_pitches}", normal_style))
elements.append(Spacer(1, 0.3 * inch))

# Table Data
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
    ("BACKGROUND", (0,0), (-1,0), colors.grey),
    ("GRID", (0,0), (-1,-1), 0.5, colors.black),
    ("ALIGN", (1,1), (-1,-1), "CENTER"),
]))

elements.append(table)
elements.append(Spacer(1, 0.5 * inch))

# Add Zone Chart
elements.append(Paragraph("Pitch Location Chart", styles["Heading2"]))
elements.append(Spacer(1, 0.2 * inch))

img = Image(zone_chart_path, width=4*inch, height=4*inch)
elements.append(img)

# Build PDF
doc.build(elements)

print("Report Generated Successfully:", pdf_file)
