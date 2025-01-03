import numpy
import plotly.graph_objects as go
import plotly.io as pio
import os
import csv
import ast
import openpyxl
import pandas as pd

# Add user data
categories = ['Shared Vision', 'Strategy', 'Business Alignment', 'Subordinates for Success', 'Cross-functional teams',
              'Clarity on priorities', 'Acceptance Criteria', 'Enable Focus', 'Engagement', 'Feedback',
              'Enable Autonomy', 'Change and ambiguity', 'Desired Culture', 'Work autonomously', 'Stakeholders',
              'Customer Focus', 'Attrition', 'Teams', 'Develop People']


# Read in colour array from properties file
def read_file(file_path: object) -> object:
     data = []
     with open(file_path, 'r') as file:
         for line in file:
             key, value = line.strip().split('=')
             data.append((value))
     return data

# Read the CSV file
# def read_csv_file(file_path: object) -> object:
#     data = []
#     try:
#         with open(file_path, 'r') as csvfile:
#             reader = csv.DictReader(csvfile)
#             for line in csvfile:
#                 data.append(value)
#         print(f"Successfully read {len(data)} rows from {file_path}")
#         return data
#     except FileNotFoundError:
#         print(f"Error: File '{file_path}' not found.")
#         return None
#     except csv.Error as e:
#         print(f"Error reading CSV file: {e}")
#         return None
      
# def read_file(file_path: object) -> object:
#     data = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             key, value = line.strip().split('=')
#             data.append((value))
#     return data



# Usage
file_path = 'properties'
result = read_file(file_path)
#result = read_csv_file(file_path)

# print(result)


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
        title=f'{user_name} Assessment',
        polar=dict(
            radialaxis=dict(range=[0, 5], showticklabels=True, ticks=''),
            angularaxis=dict(showticklabels=True, ticks='')
        )
    )

    return fig


#input_file = './Users.csv' #     filename = 'Users.csv'
input_file = './Users.xlsx'
check_file = os.path.isfile(input_file)

if os.path.isfile(input_file):
    print(f'The input file {input_file} exists')
else:
    print(f'The input file {input_file} does not exist. Exiting.')
    exit()

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


output_file = './Assessments.html'
#check_file = os.path.isfile(input_file)

#if os.path.isfile("Assessments.html"):

#output_file = './Assessments.html'
#check_file = os.path.isfile(input_file)

if os.path.isfile("Assessments.html"):
    print(f'The output file {output_file} already exists. Overwriting.')
    print(f'A backup copy will be saved to .bak')
    os.rename(output_file, output_file+'.bak')
else:
    print(f'The output file {output_file} does not exist. Executing ...')


#with open("Assessments-csv.html", "w") as f:

with open("Assessments.html", "w") as f:

    f.write(html_content)
