# Performance Polar Area

Generates interactive polar area charts from employee performance assessments. Ratings are entered through a static HTML form, exported to CSV, converted to XLSX, and rendered as a set of Plotly polar charts in a single HTML report.

## What it does

1. **`index.html`** — a standalone assessment form covering 19 competencies (Strategy and Business, Focus and Delivery, Autonomy and Change, Stakeholders and Customers). Scores are 0–5. Clicking **Save** exports the form as `user.csv`.
2. **`csv_to_xlsx_converter.py`** — converts one or more CSV files into a single XLSX workbook, one sheet per CSV. Each sheet name becomes a person's label on their chart.
3. **`PolarAreaChart.py`** — reads `user.xlsx`, builds a polar bar chart per sheet (color-coded by category group via the `properties` file), and writes them all into `Assessments.html`.

## Requirements

- Python 3.6+
- `numpy`, `plotly`, `pandas`, `openpyxl`

```bash
pip install numpy plotly pandas openpyxl
```

## Quick start

See [QUICKSTART.md](QUICKSTART.md) for the full walkthrough. Short version:

```bash
python PolarAreaChart.py
open Assessments.html
```

## Project layout

| Path | Purpose |
|---|---|
| `index.html` | Assessment form, exports ratings to CSV |
| `PolarAreaChart.py` | Builds `Assessments.html` from `user.xlsx` |
| `csv_to_xlsx_converter.py` | CSV → XLSX conversion utility (see [README-csv-to-xlsx.md](README-csv-to-xlsx.md)) |
| `properties` | Category color mapping used by the charts |
| `assessments/` | Output/working directory for generated files |

## Multiple people

Duplicate a sheet in `user.xlsx` for each person and label the tab with their name — `PolarAreaChart.py` uses the sheet name as the chart title and renders every sheet into `Assessments.html`.

## License

Proprietary. All rights reserved.
