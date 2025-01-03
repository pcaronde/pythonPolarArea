import plotly.graph_objects as go
import unittest

from PolarAreaChart import get_fig_data


class TestGetFigData(unittest.TestCase):
    def test_get_fig_data(self):
        r_values = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        user_name = "Test Person"
        fig = get_fig_data(r_values, user_name)

        # Assert that the returned figure is of type go.Figure
        self.assertIsInstance(fig, go.Figure)

        # Add more assertions as needed to validate the expected behavior of get_fig_data()


if __name__ == '__main__':
    unittest.main()