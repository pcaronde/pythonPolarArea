import pytest
import os
import sys
import pandas as pd
from pathlib import Path
from unittest.mock import patch, mock_open
import csv_to_xlsx


@pytest.fixture
def mock_csv_files(tmp_path):
    """Create mock CSV files for testing."""
    csv_data1 = "col1,col2\nvalue1,value2\nvalue3,value4"
    csv_data2 = "col3,col4\nvalue5,value6\nvalue7,value8"

    csv_file1 = tmp_path / "test_file1.csv"
    csv_file2 = tmp_path / "test_file2.csv"

    csv_file1.write_text(csv_data1)
    csv_file2.write_text(csv_data2)

    return [str(csv_file1), str(csv_file2)]


@pytest.mark.parametrize("input_files, output_path, expected_behavior, test_id", [
    # Happy path tests
    (["test.csv"], None, "single_file_default_output", "single_file_default"),
    (["test1.csv", "test2.csv"], "output.xlsx", "multiple_files_custom_output", "multiple_files_custom"),

    # Edge cases
    (
    ["test_with_very_long_filename_that_exceeds_31_characters.csv"], None, "long_filename_truncation", "long_filename"),
    (["test.csv"], "output_with_no_extension", "auto_add_xlsx_extension", "no_extension_output"),

    # Error cases
    (["nonexistent.csv"], None, "file_not_found", "nonexistent_file"),
    (["invalid_file.txt"], None, "invalid_file_type", "invalid_file_type"),
], ids=lambda *args: args[-1])
def test_convert_csv_to_xlsx(mock_csv_files, input_files, output_path, expected_behavior, test_id, tmp_path):
    # Arrange
    if test_id in ["nonexistent_file", "invalid_file_type"]:
        if test_id == "nonexistent_file":
            input_files = ["/path/to/nonexistent.csv"]
        else:
            input_files = ["/path/to/invalid_file.txt"]
    elif test_id in ["single_file_default", "multiple_files_custom", "long_filename", "no_extension_output"]:
        input_files = [mock_csv_files[0]] if len(input_files) == 1 else mock_csv_files

    if output_path:
        output_path = str(tmp_path / output_path)

    # Act & Assert
    if test_id in ["nonexistent_file", "invalid_file_type"]:
        with pytest.raises((FileNotFoundError, ValueError)):
            csv_to_xlsx.convert_csv_to_xlsx(input_files, output_path)
    else:
        with patch('builtins.print') as mock_print:
            csv_to_xlsx.convert_csv_to_xlsx(input_files, output_path)

        # Verify output file
        if output_path is None:
            output_path = Path(input_files[0]).with_suffix('.xlsx')
        elif not output_path.lower().endswith('.xlsx'):
            output_path = Path(output_path).with_suffix('.xlsx')

        assert os.path.exists(output_path)

        # Check Excel file
        with pd.ExcelFile(output_path) as xls:
            if test_id == "long_filename":
                assert len(xls.sheet_names[0]) <= 31
            else:
                assert len(xls.sheet_names) == len(input_files)


def test_convert_csv_to_xlsx_single_string_input(mock_csv_files, tmp_path):
    # Arrange
    input_file = mock_csv_files[0]
    output_path = str(tmp_path / "output.xlsx")

    # Act
    with patch('builtins.print'):
        csv_to_xlsx.convert_csv_to_xlsx(input_file, output_path)

    # Assert
    assert os.path.exists(output_path)
    with pd.ExcelFile(output_path) as xls:
        assert len(xls.sheet_names) == 1


@pytest.mark.parametrize("cli_args, expected_behavior", [
    # Happy path tests
    (["test.csv"], "single_file_conversion"),
    (["-o", "output.xlsx", "test1.csv", "test2.csv"], "multiple_files_custom_output"),

    # Error cases
    ([], "insufficient_arguments"),
    (["-o"], "missing_output_path"),
], ids=lambda *args: args[1])
def test_main_cli(mock_csv_files, cli_args, expected_behavior, monkeypatch, tmp_path):
    # Arrange
    if expected_behavior == "single_file_conversion":
        cli_args = [mock_csv_files[0]]
    elif expected_behavior == "multiple_files_custom_output":
        cli_args = ["-o", str(tmp_path / "output.xlsx")] + mock_csv_files

    # Modify sys.argv for testing
    monkeypatch.setattr(sys, 'argv', ['csv_to_xlsx.py'] + cli_args)

    # Act & Assert
    if expected_behavior in ["insufficient_arguments", "missing_output_path"]:
        with pytest.raises(SystemExit):
            with patch('builtins.print'):
                csv_to_xlsx.main()
    else:
        with patch('builtins.print'), patch('csv_to_xlsx.convert_csv_to_xlsx'):
            csv_to_xlsx.main()
