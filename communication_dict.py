import csv

def load_communication_dict(csv_file):
    communication_dict = {}
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            gesture_id = row['gesture_id']
            description = row['response']
            communication_dict[gesture_id] = description
    return communication_dict

communication_dict = load_communication_dict('communication_dict.csv')
