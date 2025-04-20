import os
import csv
import shutil
import pandas as pd


"""
Move files from one directory to its subdirectories based on its category, which is obtained from the second column of a csv file, with the first column being the file name without the extension.
"""
def sort_type(csv_file, source_dir, dest_dir):
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)

        for row in reader:
            file_name, category = row[0]+'.txt', row[1]
            if category == '':
                category = 'Others'
            source_file = os.path.join(source_dir, file_name)
            dest_dir_category = os.path.join(dest_dir, category)
            dest_file = os.path.join(dest_dir_category, file_name)

            if os.path.exists(source_file):
                if not os.path.exists(dest_dir_category):
                    os.makedirs(dest_dir_category)
                shutil.copy(source_file, dest_file)
                print(f'Moved {file_name} to {dest_dir_category}')


"""
Move files from one directory to its subdirectories based on its year, which is obtained from the first part of the file name before the first underscore.
"""
def sort_year(source_dir, dest_dir):
    for file_name in os.listdir(source_dir):
        year = "19" + file_name.split('_')[0][:2]
        source_file = os.path.join(source_dir, file_name)
        dest_dir_year = os.path.join(dest_dir, year)
        dest_file = os.path.join(dest_dir_year, file_name)

        if os.path.exists(source_file):
            if not os.path.exists(dest_dir_year):
                os.makedirs(dest_dir_year)
            shutil.copy(source_file, dest_file)
            print(f'Moved {file_name} to {dest_dir_year}')


"""
Move files from one directory to its subdirectories based on its confidence, which is obtained from the second column of a csv file, with the first column being the file name without the extension.
"""
def sort_confidence(source_dir, dest_dir, filenames, confidences, threshold_list):
    for filename, confidence in zip(filenames, confidences):
        for threshold in threshold_list:
            if confidence > threshold:
                output_dir = os.path.join(dest_dir, str(threshold))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                source_file = os.path.join(source_dir, filename+'.txt')
                target_file = os.path.join(output_dir, filename+'.txt')
                shutil.copy(source_file, target_file)
                print(f'Moved {filename} to {output_dir} with confidence {confidence}')
                break


if __name__ == '__main__':
    csv_file = 'target/data.csv'
    source_dir = 'output/text_simplified/'
    dest_dir_type = 'output/text_sorted_type/'
    dest_dir_year = 'output/text_sorted_year/'
    dest_dir_confidence = 'output/text_sorted_confidence/'

    df = pd.read_csv('output/confidence.csv', header=None)
    filenames, confidences = df[0], df[1]
    threshold_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    sort_type(csv_file, source_dir, dest_dir_type)
    sort_year(source_dir, dest_dir_year)
    sort_confidence(source_dir, dest_dir_confidence, filenames, confidences, threshold_list)