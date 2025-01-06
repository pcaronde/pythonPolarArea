# CSV to XLSX Converter

A Python utility for converting single or multiple CSV files into a single Excel (XLSX) workbook. Each CSV file becomes a separate worksheet in the Excel file, with the sheet name derived from the CSV filename.

## Features

- Convert single or multiple CSV files to XLSX format
- Automatic sheet naming based on CSV filenames
- Custom output filename support
- Handles Excel sheet name length limitations
- Preserves data types during conversion
- Provides progress feedback during conversion

## Requirements

- Python 3.6+
- pandas
- openpyxl

Install required packages using:
```bash
pip install pandas openpyxl
```

## Installation
The file `csv_to_xlsx_converter.py` can be used as a standalone conversion tool.
1. Clone this repository or download the script:
```bash
git clone https://github.com/pcaronde/pythonPolarArea/csv-to-xlsx-converter.git
cd csv-to-xlsx-converter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

1. Converting a single CSV file:
```bash
python csv_to_xlsx_converter.py input.csv
```
This will create `input.xlsx` in the same directory.

2. Converting multiple CSV files:
```bash
python csv_to_xlsx_converter.py file1.csv file2.csv file3.csv
```
This will create an Excel file named after the first CSV file, containing all CSV data in separate sheets.

3. Specifying a custom output filename:
```bash
python csv_to_xlsx_converter.py -o output.xlsx file1.csv file2.csv
```

### Python API

We use the converter directly in our Python code like this:

```python
from csv_to_xlsx_converter import convert_csv_to_xlsx

# Convert single file
convert_csv_to_xlsx('input.csv')

# Convert multiple files
convert_csv_to_xlsx(['file1.csv', 'file2.csv'], output_path='output.xlsx')
```

## Detailed Example

Let's walk through a complete example of using the converter with sample data.

### Sample Data

1. Create a file named `sales.csv`:
```csv
Date,Product,Revenue
2024-01-01,Widget A,1500
2024-01-02,Widget B,2000
2024-01-03,Widget A,1750
```

2. Create another file named `inventory.csv`:
```csv
Product,Quantity,Reorder_Level
Widget A,150,50
Widget B,200,75
Widget C,100,25
```

### Converting Files

1. Convert both files to a single Excel workbook:
```bash
python csv_to_xlsx_converter.py -o sales_report.xlsx sales.csv inventory.csv
```

2. The script will output:
```
Processed sales.csv -> Sheet: sales
Processed inventory.csv -> Sheet: inventory

Successfully created Excel file: sales_report.xlsx
Total sheets created: 2
```

3. The resulting `sales_report.xlsx` will contain:
   - Sheet "sales" with the sales data
   - Sheet "inventory" with the inventory data

### Using the Python API

```python
from csv_to_xlsx_converter import convert_csv_to_xlsx
import pandas as pd

# Create sample CSV files
sales_data = pd.DataFrame({
    'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    'Product': ['Widget A', 'Widget B', 'Widget A'],
    'Revenue': [1500, 2000, 1750]
})
sales_data.to_csv('sales.csv', index=False)

inventory_data = pd.DataFrame({
    'Product': ['Widget A', 'Widget B', 'Widget C'],
    'Quantity': [150, 200, 100],
    'Reorder_Level': [50, 75, 25]
})
inventory_data.to_csv('inventory.csv', index=False)

# Convert files to Excel
convert_csv_to_xlsx(
    csv_paths=['sales.csv', 'inventory.csv'],
    output_path='sales_report.xlsx'
)

# Verify the results
excel_file = pd.ExcelFile('sales_report.xlsx')
print(f"Sheets in Excel file: {excel_file.sheet_names}")

# Read and display the data
sales_sheet = pd.read_excel('sales_report.xlsx', sheet_name='sales')
inventory_sheet = pd.read_excel('sales_report.xlsx', sheet_name='inventory')

print("\nSales Sheet:")
print(sales_sheet)
print("\nInventory Sheet:")
print(inventory_sheet)
```

## Testing

Run the unit tests:
```bash
python -m unittest test_csv_to_xlsx_converter.py
```

The test suite includes:
- Single file conversion
- Multiple file conversion
- Invalid file handling
- Long filename handling

## Limitations

- Sheet names are limited to 31 characters (Excel limitation)
- All CSV files must use the same encoding
- Large CSV files may require significant memory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
