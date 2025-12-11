import { ASSESSMENT_CONFIG } from './config.js';

/**
 * Performance Assessment Chart Application
 * 
 * Creates interactive polar area charts for employee performance evaluation
 * across 4 themes and 19 criteria.
 */
class PerformanceAssessment {
    constructor() {
        this.chart = null;
        this.employeeName = '';
        this.themes = ASSESSMENT_CONFIG.themes;
    }

    /**
     * Initialize the assessment application
     */
    init() {
        this.setCurrentDate();
        this.initializeChart();
        this.attachEventListeners();
        this.loadFromLocalStorage();
    }

    /**
     * Set current date in the header
     */
    setCurrentDate() {
        const dateElement = document.getElementById('currentDate');
        if (dateElement) {
            dateElement.textContent = new Date().toLocaleDateString();
        }
    }

    /**
     * Validate and clamp input value between min and max
     * @param {string|number} value - Input value to validate
     * @returns {number} - Validated value clamped between 0-5
     */
    validateInput(value) {
        const num = parseFloat(value);
        if (isNaN(num)) return 0;
        return Math.max(
            ASSESSMENT_CONFIG.ratingScale.min,
            Math.min(ASSESSMENT_CONFIG.ratingScale.max, num)
        );
    }

    /**
     * Sanitize employee name to prevent XSS
     * @param {string} name - Raw employee name
     * @returns {string} - Sanitized name
     */
    sanitizeEmployeeName(name) {
        const div = document.createElement('div');
        div.textContent = name;
        return div.innerHTML.trim();
    }

    /**
     * Sanitize filename for downloads
     * @param {string} name - Raw filename
     * @returns {string} - Sanitized filename
     */
    sanitizeFilename(name) {
        return name
            .trim()
            .replace(/[^a-z0-9_-]/gi, '_')
            .replace(/_{2,}/g, '_')
            .substring(0, 50);
    }

    /**
     * Initialize Chart.js polar area chart
     */
    initializeChart() {
        try {
            const canvas = document.getElementById('performanceChart');
            if (!canvas) {
                throw new Error('Canvas element not found');
            }

            const ctx = canvas.getContext('2d');
            const labels = [];
            const backgroundColor = [];
            const borderColor = [];

            // Build labels and colors from theme configuration
            Object.entries(this.themes).forEach(([theme, themeData]) => {
                themeData.metrics.forEach(metric => {
                    labels.push(`${theme}: ${metric.label}`);
                    backgroundColor.push(themeData.color.replace('%a', '0.5'));
                    borderColor.push(themeData.color.replace('%a', '1'));
                });
            });

            this.chart = new Chart(ctx, {
                type: 'polarArea',
                data: {
                    labels: labels,
                    datasets: [{
                        data: new Array(labels.length).fill(0),
                        backgroundColor: backgroundColor,
                        borderColor: borderColor,
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        r: {
                            min: ASSESSMENT_CONFIG.ratingScale.min,
                            max: ASSESSMENT_CONFIG.ratingScale.max,
                            ticks: {
                                stepSize: 1,
                                display: true,
                                backdropColor: 'rgba(255, 255, 255, 0.8)',
                            },
                            grid: {
                                circular: true,
                                color: 'rgba(0, 0, 0, 0.1)'
                            },
                            angleLines: {
                                display: true,
                                color: 'rgba(0, 0, 0, 0.1)'
                            },
                            pointLabels: {
                                display: true,
                                centerPointLabels: true,
                                font: {
                                    size: 10
                                },
                                callback: function (label) {
                                    // Split the label at the colon and return only the metric name
                                    return label.split(': ')[1];
                                }
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                generateLabels: () => {
                                    // Return only the four main themes for the legend
                                    return Object.entries(this.themes).map(([theme, themeData]) => ({
                                        text: theme,
                                        fillStyle: themeData.color.replace('%a', '0.5'),
                                        strokeStyle: themeData.color.replace('%a', '1'),
                                        lineWidth: 1,
                                        hidden: false,
                                    }));
                                }
                            },
                        },
                        title: {
                            display: true,
                            text: 'Results'
                        },
                        tooltip: {
                            callbacks: {
                                label: (context) => `Score: ${context.formattedValue}`
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Failed to initialize chart:', error);
            this.showErrorMessage('Failed to load chart. Please refresh the page.');
        }
    }

    /**
     * Update chart with current form values
     */
    updateChart() {
        if (!this.chart) return;

        const values = [];

        Object.values(this.themes).forEach(themeData => {
            themeData.metrics.forEach(metric => {
                const input = document.getElementById(metric.id);
                if (!input) {
                    console.warn(`Missing input field: ${metric.id}`);
                    values.push(0);
                    return;
                }
                const value = this.validateInput(input.value);
                values.push(value);
            });
        });

        this.chart.data.datasets[0].data = values;
        this.chart.update();
    }

    /**
     * Debounce function to limit execution frequency
     * @param {Function} func - Function to debounce
     * @param {number} wait - Wait time in milliseconds
     * @returns {Function} - Debounced function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func.apply(this, args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Save assessment data to localStorage
     */
    saveToLocalStorage() {
        try {
            const formData = {};
            const inputs = document.querySelectorAll('#inputForm input[type="number"]');

            inputs.forEach(input => {
                formData[input.id] = input.value;
            });

            const data = {
                employeeName: this.employeeName,
                date: new Date().toISOString(),
                formData
            };

            localStorage.setItem('performanceAssessment', JSON.stringify(data));
        } catch (error) {
            console.error('Failed to save to localStorage:', error);
        }
    }

    /**
     * Load assessment data from localStorage
     */
    loadFromLocalStorage() {
        try {
            const saved = localStorage.getItem('performanceAssessment');
            if (!saved) return;

            const data = JSON.parse(saved);

            // Check if data is from today
            const savedDate = new Date(data.date);
            const today = new Date();
            if (savedDate.toDateString() !== today.toDateString()) {
                return; // Don't load old data
            }

            // Restore employee name
            const nameInput = document.getElementById('employeeName');
            if (nameInput && data.employeeName) {
                nameInput.value = data.employeeName;
                this.employeeName = data.employeeName;
            }

            // Restore form values
            Object.entries(data.formData).forEach(([id, value]) => {
                const input = document.getElementById(id);
                if (input) input.value = value;
            });

            this.updateChart();
        } catch (error) {
            console.error('Failed to load saved data:', error);
        }
    }

    /**
     * Export assessment data to CSV
     */
    saveToCSV() {
        try {
            const formData = new FormData(document.getElementById('inputForm'));
            let csvContent = "Categories,Ratings\n";

            for (let [key, value] of formData.entries()) {
                csvContent += `${key},${value}\n`;
            }

            // Create and trigger download
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);

            // Use the employee name for the filename if available
            const sanitizedName = this.employeeName
                ? this.sanitizeFilename(this.employeeName)
                : 'user';
            const dateStr = new Date().toISOString().split('T')[0];
            const filename = `${sanitizedName}_assessment_${dateStr}.csv`;

            link.setAttribute("download", filename);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (error) {
            console.error('Failed to save CSV:', error);
            this.showErrorMessage('Failed to save CSV file. Please try again.');
        }
    }

    /**
     * Show error message to user
     * @param {string} message - Error message to display
     */
    showErrorMessage(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        errorDiv.setAttribute('role', 'alert');
        errorDiv.style.cssText = `
            padding: 10px;
            margin: 10px 0;
            background-color: #f44336;
            color: white;
            border-radius: 4px;
        `;

        const container = document.querySelector('.chart-container');
        if (container) {
            container.prepend(errorDiv);
            setTimeout(() => errorDiv.remove(), 5000);
        }
    }

    /**
     * Attach event listeners to form elements
     */
    attachEventListeners() {
        // Employee name input
        const nameInput = document.getElementById('employeeName');
        if (nameInput) {
            nameInput.addEventListener('input', (e) => {
                this.employeeName = this.sanitizeEmployeeName(e.target.value);

                // Update chart title
                if (this.chart) {
                    this.chart.options.plugins.title.text = this.employeeName
                        ? `${this.employeeName} - Results`
                        : 'Results';
                    this.chart.update();
                }
            });
        }

        // Number inputs - update chart on change
        const debouncedUpdate = this.debounce(() => {
            this.updateChart();
            this.saveToLocalStorage();
        }, 300);

        document.querySelectorAll('input[type="number"]').forEach(input => {
            input.addEventListener('input', debouncedUpdate.bind(this));
        });

        // Auto-save on employee name change
        if (nameInput) {
            const debouncedSave = this.debounce(() => {
                this.saveToLocalStorage();
            }, 1000);
            nameInput.addEventListener('input', debouncedSave.bind(this));
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + S to save
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                this.saveToCSV();
            }
        });
    }
}

// Initialize application when DOM is loaded
let assessmentApp;

document.addEventListener('DOMContentLoaded', () => {
    assessmentApp = new PerformanceAssessment();
    assessmentApp.init();
});

// Export for use in HTML
window.saveOnly = function () {
    if (assessmentApp) {
        assessmentApp.saveToCSV();
    }
};
