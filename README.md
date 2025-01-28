# Create polar area charts 
A webpage to collect with some values and and output a polar chart using chart.js

## Requirements
index.html
styles.css

## How to run
set index.html in the / directory of your web server

# DEPRECATED
## How to run
1. start server.py 
```python3 server.py```
2. This starts a server on port 5000
3. Navigate a browser to http://localhost:5000/
4. Fill in form and choose generate visualisation button
# DEPRECATED

### Script in a Container
1. Create a folder and place the attached script & Users.xlsx file in the folder (download this Google Sheet here and rename Users.xlsx) - Or change line 48 if you don't want to rename.
2. Open a terminal and cd to that folder
3. And then run the below command
```docker run -t -i -v ${PWD}:/tmp python bash```
4. Then install the below packages
```pip install numpy plotly pandas openpyxl```
5. From inside the container, cd to /tmp location and run the command
```python main.py```
6. This will generate an assessment-v2.html file inside the same directory, all the diagrams will be in a single html page labelled, scroll down.
You only need to duplicate the tabs and keep the format for each of the people you want to perform an assessment of. Label each tab with the person's name as the script uses this as a label for the diagrams.
7. To edit, change column B in Users.xlsx. Use a new tab for each person.

#### Script locally
1. Create a folder and place the attached script & Users.xlsx file in the folder (download this Google Sheet here and rename Users.xlsx) - Or change line 48 if you don't want to rename.
2. Open a terminal and cd to that folder
3. And then run the below command
```python polar-area-chart-v2.py```
4. This will generate an Assessments.html file inside the same directory, all the diagrams will be in a single html page labelled, scroll down.
You only need to duplicate the tabs and keep the format for each of the people you want to perform an assessment of. Label each tab with the person's name as the script uses this as a label for the diagrams.
5. To edit, change column B in Users.xlsx. Use a new tab for each person.
