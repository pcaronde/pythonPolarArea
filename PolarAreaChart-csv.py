import numpy
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from jinja2 import Template
import os
import csv
import ast
import openpyxl
import pandas as pd

# Define the category names
categories = [
    'Shared Vision', 'Strategy', 'Business Alignment', 'Subordinates for Success',
    'Cross-functional teams', 'Clarity on priorities', 'Acceptance Criteria',
    'Enable Focus', 'Engagement', 'Feedback', 'Enable Autonomy', 'Change and ambiguity',
    'Desired Culture', 'Work autonomously', 'Stakeholders', 'Customer Focus',
    'Attrition', 'Teams', 'Develop People'
]



# if os.path.isfile(input_file):
#     print(f'The input file {input_file} exists')
# else:
#     print(f'The input file {input_file} does not exist. Exiting.')
#     exit()
#
# if os.path.isfile("Assessments-csv.html"):
#     print(f'The output file {output_file} already exists. Overwriting.')
#     print(f'A backup copy will be saved to .bak')
#     os.rename(output_file, output_file + '.bak')
# else:
#     print(f'The output file {output_file} does not exist. Executing ...')


# TODO Rename this here and in `read_csv_file`
#def read_csv_file(input_file, csvdata, html_content):
def read_csv_file(input_file):
    try:
        with open(input_file, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            #chart_html = pio.to_html(fig, full_html=False)
            #html_content += chart_html
            print("\nFirst 19 rows of data:")
            for row in csv_reader:
                print(row)
    except Exception:
        print("failed to open the file")


# Read in colour array from properties file
def read_file(file_path: object) -> object:
    data = []
    with open(file_path, 'r') as file:
        print(r, filepath)
        for line in file:
            key, value = line.strip().split('=')
            data.append((value))
    return data


def get_fig_data(r_values, user_name):
    # Create a polar area chart using four coolors, static categories, and variable values
    fig = go.Figure(go.Barpolar(
        r=r_values,
        theta=categories,
        marker_color=result,
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
        title=f'{user_name} Assessment',  # Change this to $username later
        polar=dict(
            radialaxis=dict(range=[0, 5], showticklabels=True, ticks=''),
            angularaxis=dict(showticklabels=True, ticks='')
        )
    )
    return fig


#with open("Assessments-csv.html", "w") as f:
#    f.write(html_content)


if __name__ == "__main__":
  html_content = ''
  user_name = "Peter"
  file_path = "properties"
  input_file = './form_data.csv'  # filename = 'Users.csv'
  filename = './form_data.csv'
  # input_file = './Users.xlsx'
  output_file = "Assessments-csv.html"
  read_csv_file(input_file)

