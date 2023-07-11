A simple python script to create a polar area chart and fill it on with some values and labels.

How to run
Executive Assessment Script
1. Create a folder and place the attached script & Users.xlsx file in the folder (download this googlesheet here and rename users.xlsx) - Or change line 48 if you don't want to rename.
2. Open a terminal and cd to that folder
3. And then run the below command
docker run -t -i -v ${PWD}:/tmp python bash
4. Then install the below packages
pip install numpy
pip install plotly
pip install pandas
pip install openpyxl
5. And then inside the container cd to /tmp location and run the below command
python plt.py
6. Above command will generate an Assesment.html file inside the same directory, all of the diagrams will be in a single html page labelled, scroll down.
You just need to duplicate the tabs and keep the format for each of people you want to perform an assessment of, label each tab with the persons name as the script uses this as a label for the diagrams.