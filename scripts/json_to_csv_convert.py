import sys
import json
import csv

def json_to_csv(json_file, csv_file):
    with open(json_file, 'r') as json_file:
        data = json.load(json_file)

    # Assuming the JSON data is a list of dictionaries
    if isinstance(data, list):
        header = data[0].keys() if data else []
        with open(csv_file, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=header)
            writer.writeheader()
            writer.writerows(data)
        print(f'Conversion successful: {json_file} -> {csv_file}')
    else:
        print('Invalid JSON format. Expecting a list of dictionaries.')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python script.py input.json output.csv')
    else:
        input_json = sys.argv[1]
        output_csv = sys.argv[2]
        json_to_csv(input_json, output_csv)
