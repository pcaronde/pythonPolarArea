import pandas as pd
import sys
from pathlib import Path
from typing import List, Union, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def convert_csv_to_xlsx(
    csv_paths: Union[str, List[str]],
    output_path: Optional[str] = None
) -> None:
    """
    Efficiently convert one or multiple CSV files to XLSX format.
    """
    try:
        # Normalize input to list and convert to Path objects with validation
        csv_files = [
            Path(path) for path in ([csv_paths] if isinstance(csv_paths, str) else csv_paths)
            if Path(path).exists() and Path(path).suffix.lower() == '.csv'
        ]

        if not csv_files:
            raise ValueError("No valid CSV files found")

        # Determine output path with less overhead
        output_path = Path(output_path or csv_files[0]).with_suffix('.xlsx')

        # More efficient Excel writing
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for csv_file in csv_files:
                df = pd.read_csv(csv_file, low_memory=False)
                sheet_name = csv_file.stem[:31]  # Concise sheet name truncation
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                logging.info(f"Processed {csv_file.name} -> Sheet: {sheet_name}")

        logging.info(f"\nSuccessfully created Excel file: {output_path.name}")
        logging.info(f"Total sheets created: {len(csv_files)}")

    except Exception as e:
        logging.error(f"Error converting files: {e}")
        sys.exit(1)

def main():
    """Handle command line arguments and execute conversion."""
    if len(sys.argv) < 2:
        print("Usage: csv_to_xlsx_converter.py [-o output.xlsx] <csv_file1> [csv_file2 ...]")
        sys.exit(1)

    args = sys.argv[1:]
    output_path = None

    if "-o" in args:
        output_index = args.index("-o")
        if output_index + 1 >= len(args):
            logging.error("Error: -o flag requires an output path")
            sys.exit(1)
        output_path = args[output_index + 1]
        args = args[:output_index] + args[output_index + 2:]

    convert_csv_to_xlsx(args, output_path)

if __name__ == "__main__":
    main()