# polar-area-chart-v2.py
import numpy
import plotly.graph_objects as go
import plotly.io as pio
import os
import pandas as pd
from csv_to_xlsx_converter import convert_csv_to_xlsx

# Add user data categories
categories = ['Shared Vision', 'Strategy', 'Business Alignment', 'Customer Focus', 'Cross-functional Cooperation',
              'Prioritisation', 'Acceptance Criteria', 'Enable Focus', 'Engagement', 'Feedback',
              'Enable Autonomy', 'Change and Ambiguity', 'Desired Culture', 'Works Autonomously', 'Stakeholder Mgmt',
               'Team Attrition', 'Team Structure', 'Develops People', 'Subordinates for Success']

#[TODO] Rearrange categories to match
# Strategy
# Focus
# Autonomy
# Stakeholder

# Convert the CSV file
convert_csv_to_xlsx('user-v2.csv')

# Read in colour array from properties file
def read_file(file_path: object) -> object:
     data = []
     with open(file_path, 'r') as file:
         for line in file:
             key, value = line.strip().split('=')
             data.append(value)
     return data

# Usage
file_path = 'properties'
result = read_file(file_path)

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
        # [TODO] Add if statement if user=blank
        title=f'{user_name} Assessment',
        polar=dict(
            radialaxis=dict(range=[0, 5], showticklabels=True, ticks=''),
            angularaxis=dict(showticklabels=True, ticks='')
        )
    )

    return fig

# Define the explanatory text to be appended
explanatory_html = """
<div style="font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px;">
    <h1>Performance Assessment</h1>
    <p>The main goal of a performance assessment is to evaluate an employee's job performance in accordance with predefined standards and criteria.
    Organisations may pinpoint their strengths and weaknesses, celebrate accomplishments and address areas for development which may be enhanced through performance improvement planning.</p>
    
    <p>This assessment is not meant to be a comprehensive evaluation of an employee's performance but a tool to help you identify areas for improvement.</p>
    
    <p>The assessment is divided into 4 themes and 12 categories and each category has 5 ratings. The ratings are as follows:</p>
    <ul>
        <li>0 - Not Applicable</li>
        <li>1 - Very Poor</li>
        <li>2 - Poor</li>
        <li>3 - Fair</li>
        <li>4 - Good</li>
        <li>5 - Excellent</li>
    </ul>

    <h2>Assessment Categories</h2>
    
    <h3>Strategic Vision and Business Alignment</h3>
    <div><strong>Shared Vision:</strong> Consistently demonstrates alignment between personal objectives and company vision through actions and decisions.</div>
    <div><strong>Strategy:</strong> Effectively develops strategic plans and translates them into actionable tactical steps.</div>
    <div><strong>Business Alignment:</strong> Demonstrates clear connection between daily activities and broader company objectives.</div>
    <div><strong>Customer Focus:</strong> Consistently prioritises customer needs and satisfaction in decision-making and actions.</div>

    <h3>Focus and Engagement</h3>
    <div><strong>Enables Cross-functional Work:</strong> Successfully builds relationships and collaborates across teams and departments to achieve shared objectives.</div>
    <div><strong>Clarity of Priorities:</strong> Systematically evaluates and ranks tasks based on importance, ensuring efficient execution.</div>
    <div><strong>Acceptance Criteria:</strong> Clearly defines and communicates quality standards for deliverables.</div>
    <div><strong>Focus:</strong> Creates clear roadmaps with specific milestones and maintains team focus on key objectives.</div>
    <div><strong>Engagement:</strong> Shows consistent enthusiasm and dedication in approaching work responsibilities.</div>

    <h3>Autonomy and Change</h3>
    <div><strong>Feedback:</strong> Delivers constructive feedback effectively while remaining open and responsive to receiving input from others.</div>
    <div><strong>Enables Autonomy:</strong> Empowers team members to make decisions and innovate within appropriate boundaries.</div>
    <div><strong>Accepts Change and Ambiguity:</strong> Maintains effectiveness and adapts quickly when facing uncertain or changing circumstances.</div>
    <div><strong>Promotes Desired Culture:</strong> Demonstrates and promotes behaviours that align with and strengthen company values.</div>
    <div><strong>Works Autonomously:</strong> Effectively self-manages time and priorities to deliver optimal results independently.</div>

    <h3>Stakeholders and Team</h3>
    <div><strong>Stakeholder Management:</strong> Identifies key stakeholders and demonstrates a deep understanding of their explicit and implicit needs.</div>
    <div><strong>Team Attrition:</strong> Maintains strong working relationships that inspire loyalty and respect from team members.</div>
    <div><strong>Team Focus:</strong> Consistently prioritises team success over individual recognition or achievement.</div>
    <div><strong>Develops People:</strong> Actively identifies and creates opportunities for team members' professional growth and skill development.</div>
    <div><strong>Prepare subordinates for success:</strong> Actively mentors and provides resources to direct reports to help them achieve their goals.</div>

</div>
"""

# [TODO] use input_file from user name
input_file = './user-v2.xlsx'
check_file = os.path.isfile(input_file)

if os.path.isfile(input_file):
    print(f'The input file {input_file} exists')
else:
    print(f'The input file {input_file} does not exist. Exiting.')
    exit()

# Define input excel. Change this is using a different named file
xls = pd.ExcelFile(input_file)
html_content = ''

# Read data from excel sheet
for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    r_values = df['Ratings'].tolist()
    user_name = sheet_name
    fig = get_fig_data(r_values, user_name)
    chart_html = pio.to_html(fig, full_html=False)
    html_content += chart_html

# Add the explanatory text to the bottom of the content
html_content += explanatory_html

# Here we output our results to html
output_file = './assessment-v2.html'

# Make sure we back up any existing output file
if os.path.isfile("assessment-v2.html"):
     print(f'The output file {output_file} already exists. Overwriting.')
     print('A backup copy will be saved to .bak')
     os.rename(output_file, f'{output_file}.bak')
else:
     print(f'The output file {output_file} does not exist. Executing ...')

with open("assessment-v2.html", "w") as f:
    f.write(html_content)