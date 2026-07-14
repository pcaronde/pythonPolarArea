# Quickstart

Generate a polar area assessment chart end-to-end.

## 1. Install dependencies

```bash
pip install numpy plotly pandas openpyxl
```

## 2. Fill out the assessment

Open `index.html` in a browser, score each competency 0–5, and click **Save**. This downloads `user.csv` — move it into the project root.

## 3. Convert to XLSX

`PolarAreaChart.py` expects `user.xlsx`, so convert the CSV:

```bash
python csv_to_xlsx_converter.py -o user.xlsx user.csv
```

For multiple people, repeat step 2 to get one CSV per person, then convert them all into one workbook:

```bash
python csv_to_xlsx_converter.py -o user.xlsx alice.csv bob.csv
```

Rename each sheet tab to the person's name — the chart uses the sheet name as its title.

## 4. Generate the charts

```bash
python PolarAreaChart.py
```

This reads `user.xlsx` and writes `Assessments.html` (any existing report is backed up to `Assessments.html.bak`).

## 5. View the results

```bash
open Assessments.html
```

Scroll down — each sheet in `user.xlsx` produces one chart on the page.

## Running in a container instead

```bash
docker run -t -i -v ${PWD}:/tmp python bash
pip install numpy plotly pandas openpyxl
cd /tmp
python PolarAreaChart.py
```

## Customizing category colors

Edit the `properties` file — it maps rating rows to hex colors in `key=value` form, read top to bottom in the same order as the categories in `PolarAreaChart.py`.
