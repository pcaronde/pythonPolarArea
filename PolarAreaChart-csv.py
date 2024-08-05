import csv

# Define the category names
categories = [
    'Shared Vision', 'Strategy', 'Business Alignment', 'Subordinates for Success',
    'Cross-functional teams', 'Clarity on priorities', 'Acceptance Criteria',
    'Enable Focus', 'Engagement', 'Feedback', 'Enable Autonomy', 'Change and ambiguity',
    'Desired Culture', 'Work autonomously', 'Stakeholders', 'Customer Focus',
    'Attrition', 'Teams', 'Develop People'
]


# Read the CSV file
def read_csv_file(filename):
    data = []
    try:
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        print(f"Successfully read {len(data)} rows from {filename}")
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except csv.Error as e:
        print(f"Error reading CSV file: {e}")
        return None


# Main function to demonstrate usage
def main():
    filename = 'Users.csv'
    user_data = read_csv_file(filename)

    if user_data:
        # Print the first few rows as an example
        print("\nFirst 3 rows of data:")
        for row in user_data[:3]:
            print(row)


if __name__ == "__main__":
    main()