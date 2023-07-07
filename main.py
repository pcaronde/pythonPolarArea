import numpy
import plotly.graph_objects as go

# Add user data
categories = ['Shared Vision', 'Strategy', 'Business Alignment', 'Subordinates for Success', 'Cross-functional teams','Clarity on priorities', 'Acceptance Criteria', 'Enable Focus', 'Engagement', 'Feedback', 'Enable Autonomy', 'Change and ambiguity', 'Desired Culture', 'Work autonomously', 'Stakeholders',  'Customer Focus', 'Attrition', 'Teams', 'Develop People']
values = [3, 2, 2, 3, 2.5, 2, 2.5, 4, 5, 2, 4, 1, 1, 2, 1, 4, 3, 4, 4.5]

# Create a polar area chart using four coolors, static categories, and variable values
fig = go.Figure(go.Barpolar(
    r=values,
    theta=categories,
    marker_color=["#E4FF87", "#E4FF87", "#E4FF87", "#E4FF87", '#709BFF', '#709BFF', '#709BFF', '#709BFF', '#709BFF', '#6E1786', '#6E1786',  '#6E1786', '#6E1786', '#6E1786',  '#FFAA70',  '#FFAA70',  '#FFAA70',  '#FFAA70', '#FFAA70',],
    marker_line_color="white",
    marker_line_width=1,
    opacity=0.8
))

# Add labels and values to the chart
for category, value in zip(categories, values):
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
    title='Assessments',
    polar=dict(
        radialaxis=dict(range=[0, 5], showticklabels=True, ticks=''),
        angularaxis=dict(showticklabels=True, ticks='')
    )
)

fig.show()
