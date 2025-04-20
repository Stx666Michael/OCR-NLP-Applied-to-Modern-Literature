import os
import csv


def process_csv_files(folder_path, output_path, sort=True):
    output_data = []
    filenames = os.listdir(folder_path)

    if sort:
        # Sort files by name
        filenames.sort()

    # Loop through all files in the folder
    for filename in filenames:
        if filename.endswith(".csv"):
            # Split the filename to get the prefix for the output
            file_prefix = filename.split('.')[0].replace('_', '')
            
            # Read the CSV file
            with open(os.path.join(folder_path, filename), encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    # Extract the required columns
                    type = row['欄目']
                    title = row['題目']
                    page_value = row['起始頁']
                    
                    # Format the 起始頁 value
                    formatted_page_value = page_value.zfill(4)
                    
                    # Create the formatted output line
                    output_line = f"{file_prefix}_{formatted_page_value},{type},{title}"
                    output_data.append(output_line)

                print(f"Processed file: {filename}")
    
    # Write the output data to csv
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for line in output_data:
            outfile.write(line + '\n')


def process_text_files(folder_path, input_data, sort=True):
    output_data = []
    filenames = os.listdir(folder_path)

    if sort:
        # Sort files by name
        filenames.sort()
    
    # List all .txt files and extract filenames without extension
    for filename in filenames:
        if filename.endswith(".txt"):
            file_key = filename[:-4]  # Remove .txt extension
            
            # Initialize values for the data row
            data_row = [file_key]
            found = False
            
            # Try to find exact or closest match in input_data
            search_key = file_key
            while not found and int(search_key.split('_')[1]) >= 0:
                if search_key in input_data:
                    data_row.extend(input_data[search_key])
                    found = True
                else:
                    # Decrement the number part by 1
                    num_part = int(search_key.split('_')[1])
                    num_part -= 1
                    search_key = f"{search_key.split('_')[0]}_{str(num_part).zfill(4)}"
            
            # If no match found, append empty values for other columns
            if not found:
                data_row.extend([""] * len(input_data[list(input_data.keys())[0]]))
            
            output_data.append(data_row)
            print(f"Processed file: {filename}")
    
    return output_data


def read_csv(file_path):
    input_data = {}
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            key = row[0]
            input_data[key] = row[1:]
    return input_data


def write_csv(output_path, output_data):
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for row in output_data:
            writer.writerow(row)


if __name__ == '__main__':
    # Combine all CSV files in the folder to a single CSV file
    folder_path = 'target/tables/'
    output_path = 'target/data_raw.csv'
    process_csv_files(folder_path, output_path)

    # Paths
    input_csv_path = 'target/data_raw.csv'
    text_folder_path = 'output/text'
    output_csv_path = 'target/data.csv'

    # Read input data
    input_data = read_csv(input_csv_path)

    # Process text files and generate output data
    output_data = process_text_files(text_folder_path, input_data)

    # Write the output data to data.csv
    write_csv(output_csv_path, output_data)