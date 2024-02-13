import unittest
import matplotlib.pyplot as plt
import numpy as np
import os

class TestPolarAreaChart(unittest.TestCase):

    def test_polar_area_chart(self):
        # Data for the chart
        categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
        values = [30, 20, 15, 10, 25]

        # Create a figure and a set of subplots
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        # Calculate the angles for each category
        angles = [i / len(categories) * 2 * 3.14159 for i in range(len(categories))]
        angles += angles[:1]  # Close the circle

        # Plot the data as a polar area chart
        ax.fill(angles, values, alpha=0.5)

        # Set the labels for each category
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        # Set the chart title
        ax.set_title('Polar Area Chart')

        # Save the figure
        plt.savefig('test_polar_area_chart.png')

        # Close the figure
        plt.close()

        # Check if the generated image matches the baseline image
        baseline_path = 'baseline_polar_area_chart.png'
        generated_path = 'test_polar_area_chart.png'

        self.assertTrue(self.compare_images(baseline_path, generated_path))

    def compare_images(self, path1, path2):
        # Load images
        image1 = plt.imread(path1)
        image2 = plt.imread(path2)

        # Ensure the images have the same shape
        self.assertEqual(image1.shape, image2.shape)

        # Compute the absolute difference
        diff = np.sum(np.abs(image1 - image2))

        # Set a threshold for the test
        threshold = 1e-10

        # Check if the images are similar
        return diff < threshold


if __name__ == '__main__':
    unittest.main()