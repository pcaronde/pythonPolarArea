import React, { useEffect, useState } from 'react';
import { Chart as ChartJS, RadialLinearScale, ArcElement, Tooltip, Legend } from 'chart.js';
import { PolarArea } from 'react-chartjs-2';

ChartJS.register(RadialLinearScale, ArcElement, Tooltip, Legend);

const PerformanceChart = () => {
  const [chartData, setChartData] = useState({
    labels: [],
    datasets: []
  });

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await window.fs.readFile('user-v2.csv', { encoding: 'utf8' });
        const rows = response.split('\n').slice(1); // Skip header

        // Define the themes and their corresponding metrics
        const themes = {
          'Strategic Vision': ['sharedVision', 'strategy', 'businessAlignment', 'customerFocus'],
          'Focus and Engagement': ['crossFunctionalTeams', 'clarityInPriorities', 'acceptanceCriteria', 'enablingFocus', 'engagement'],
          'Autonomy and Change': ['feedback', 'enablingAutonomy', 'changeAndAmbiguity', 'desiredCulture', 'workAutonomously'],
          'Stakeholders and Team': ['stakeholders', 'teamAttrition', 'teams', 'developingPeople', 'subordinatesForSuccess']
        };

        // Process the data
        const dataMap = new Map();
        rows.forEach(row => {
          if (row) {
            const [key, value] = row.split(',');
            dataMap.set(key, parseFloat(value));
          }
        });

        // Prepare data for the chart
        const labels = [];
        const values = [];
        const backgroundColors = [];
        const borderColors = [];

        // Color schemes for each theme
        const colorSchemes = {
          'Strategic Vision': ['rgba(255, 99, 132, 0.5)', 'rgba(255, 99, 132, 1)'],
          'Focus and Engagement': ['rgba(54, 162, 235, 0.5)', 'rgba(54, 162, 235, 1)'],
          'Autonomy and Change': ['rgba(255, 206, 86, 0.5)', 'rgba(255, 206, 86, 1)'],
          'Stakeholders and Team': ['rgba(75, 192, 192, 0.5)', 'rgba(75, 192, 192, 1)']
        };

        // Calculate average for each theme
        Object.entries(themes).forEach(([theme, metrics]) => {
          const validValues = metrics
            .map(metric => dataMap.get(metric))
            .filter(value => !isNaN(value));

          if (validValues.length > 0) {
            const average = validValues.reduce((a, b) => a + b, 0) / validValues.length;
            labels.push(theme);
            values.push(average.toFixed(2));
            backgroundColors.push(colorSchemes[theme][0]);
            borderColors.push(colorSchemes[theme][1]);
          }
        });

        setChartData({
          labels,
          datasets: [
            {
              data: values,
              backgroundColor: backgroundColors,
              borderColor: borderColors,
              borderWidth: 1,
            },
          ],
        });
      } catch (error) {
        console.error('Error loading data:', error);
      }
    };

    loadData();
  }, []);

  const options = {
    scales: {
      r: {
        min: 0,
        max: 5,
        ticks: {
          stepSize: 1
        }
      }
    },
    plugins: {
      legend: {
        position: 'top',
      },
      tooltip: {
        callbacks: {
          label: (context) => `Score: ${context.formattedValue}`
        }
      }
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto p-4">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold mb-4 text-center">Performance Assessment Results</h2>
        <div className="h-96">
          <PolarArea data={chartData} options={options} />
        </div>
      </div>
    </div>
  );
};

export default PerformanceChart;