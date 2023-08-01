import numpy
import plotly.graph_objects as go
import plotly.io as pio
import sys
import ast
import openpyxl
import pandas as pd

# Add user data
categories = ['Shared Vision', 'Strategy', 'Business Alignment', 'Subordinates for Success', 'Cross-functional teams',
              'Clarity on priorities', 'Acceptance Criteria', 'Enable Focus', 'Engagement', 'Feedback',
              'Enable Autonomy', 'Change and ambiguity', 'Desired Culture', 'Work autonomously', 'Stakeholders',
              'Customer Focus', 'Attrition', 'Teams', 'Develop People']

def get_fig_data(r_values, user_name):
    # Create a polar area chart using four coolors, static categories, and variable values
    fig = go.Figure(go.Barpolar(
        r=r_values,
        theta=categories,
        marker_color=["#E4FF87", "#E4FF87", "#E4FF87", "#E4FF87", '#709BFF', '#709BFF', '#709BFF', '#709BFF', '#709BFF',
                      '#6E1786', '#6E1786', '#6E1786', '#6E1786', '#6E1786', '#FFAA70', '#FFAA70', '#FFAA70', '#FFAA70',
                      '#FFAA70', ],
        marker_line_color="white",
        marker_line_width=1,
        opacity=0.8
    ))

    # Add labels and values to the chart
    for category, value in zip(categories, r_values):
        fig.add_annotation(
            x=category,
            y=value,
            text=str(value),
            showarrow=False,
            font=dict(size=14, )
        )

    # Set chart look and feel
    fig.update_layout(
        template=None,
        title=f'{user_name} Assessments',
        polar=dict(
            radialaxis=dict(range=[0, 5], showticklabels=True, ticks=''),
            angularaxis=dict(showticklabels=True, ticks='')
        )
    )

    return fig

# Define input excel. Change this is using a different named file
xls = pd.ExcelFile("Users.xlsx")
html_content = ''


# Read data from excel sheet
for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    r_values = df['Ratings'].tolist()
    user_name = sheet_name
    fig = get_fig_data(r_values, user_name)
    chart_html = pio.to_html(fig, full_html=False)
    html_content += chart_html


with open("Assessments.html", "w") as f:
    f.write(html_content)

