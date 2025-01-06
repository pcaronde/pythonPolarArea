import unittest
import pandas as pd
import os
from pathlib import Path
import logging
from csv_to_xlsx_converter import convert_csv_to_xlsx

class TestCSVToXLSXConverter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Suppress logging during tests"""
        logging.disable(logging.CRITICAL)

    def setUp(self):
        """Set up test environment before each test"""
        self.test_dir = Path('test_files')
        self.test_dir.mkdir(exist_ok=True)
        
        # Create sample test data
        self.sample_data1 = pd.DataFrame({
            'name': ['John', 'Jane'],
            'age': [30, 25],
            'city': ['New York', 'London']
        })
        
        self.sample_data2 = pd.DataFrame({
            'product': ['Apple', 'Orange', 'Banana'],
            'price': [1.0, 0.5, 0.75],
            'stock': [100, 150, 200]
        })
        
        # Create test CSV files
        self.csv_path1 = self.test_dir / 'test1.csv'
        self.csv_path2 = self.test_dir / 'test2.csv'
        
        self.sample_data1.to_csv(self.csv_path1, index=False)
        self.sample_data2.to_csv(self.csv_path2, index=False)

    def tearDown(self):
        """Clean up test environment after each test"""
        for file in self.test_dir.glob('*'):
            file.unlink()
        self.test_dir.rmdir()

    def test_single_file_conversion(self):
        """Test converting a single CSV file"""
        output_path = self.test_dir / 'output_single.xlsx'
        convert_csv_to_xlsx(str(self.csv_path1), str(output_path))
        
        self.assertTrue(output_path.exists())
        df = pd.read_excel(output_path)
        pd.testing.assert_frame_equal(df, self.sample_data1)

    def test_multiple_file_conversion(self):
        """Test converting multiple CSV files"""
        output_path = self.test_dir / 'output_multiple.xlsx'
        convert_csv_to_xlsx([str(self.csv_path1), str(self.csv_path2)], str(output_path))
        
        self.assertTrue(output_path.exists())
        
        # Verify both sheets exist and contain correct data
        excel_file = pd.ExcelFile(output_path)
        self.assertEqual(set(excel_file.sheet_names), {'test1', 'test2'})
        
        df1 = pd.read_excel(output_path, sheet_name='test1')
        df2 = pd.read_excel(output_path, sheet_name='test2')
        
        pd.testing.assert_frame_equal(df1, self.sample_data1)
        pd.testing.assert_frame_equal(df2, self.sample_data2)

    def test_default_output_name(self):
        """Test default output filename generation"""
        convert_csv_to_xlsx(str(self.csv_path1))
        expected_output = self.csv_path1.with_suffix('.xlsx')
        
        self.assertTrue(expected_output.exists())
        df = pd.read_excel(expected_output)
        pd.testing.assert_frame_equal(df, self.sample_data1)

    def test_long_sheet_name_truncation(self):
        """Test handling of long sheet names (>31 characters)"""
        long_name = 'very_long_file_name_that_exceeds_excel_limit.csv'
        long_path = self.test_dir / long_name
        self.sample_data1.to_csv(long_path, index=False)
        
        output_path = self.test_dir / 'output_long.xlsx'
        convert_csv_to_xlsx(str(long_path), str(output_path))
        
        excel_file = pd.ExcelFile(output_path)
        self.assertTrue(all(len(name) <= 31 for name in excel_file.sheet_names))

    def test_invalid_csv_path(self):
        """Test handling of non-existent CSV file"""
        with self.assertRaises(SystemExit):
            convert_csv_to_xlsx('nonexistent.csv')

    def test_empty_input_list(self):
        """Test handling of empty input list"""
        with self.assertRaises(SystemExit):
            convert_csv_to_xlsx([])

if __name__ == '__main__':
    unittest.main(verbosity=2)
