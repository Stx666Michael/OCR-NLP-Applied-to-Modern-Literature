import os
import time
import requests
import argparse
import pandas as pd
from PIL import Image
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.common.by import By


sample_urls = [
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=1501&p=41",
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=2101&p=113",
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=1805&p=49",
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=2509&p=35",
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=2812&p=66",
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=1603&p=84",
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=1708&p=80",
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=1911&p=108",
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=2002&p=21",
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=2203&p=135",
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=2306&p=142",
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=2407&p=121",
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=2602&p=7",
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=2711&p=139",
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=2910&p=144",
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=3004&p=52",
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=3112&p=65",
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=1605&p=123",
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=1501&p=57",
    "https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b=2001&p=62"
]
table_path = 'target/tables/'
image_path = 'target/images/'
sample_image_path = 'target/samples/images/'


def download_tables(output_dir: str, year: int, month: int):
    filename = output_dir+f'{year}_{month:02d}.csv'

    if os.path.exists(filename):
        print(f'Table data {filename} already exists.')
        return

    # Open the webpage
    driver.get(f'https://mhdb.mh.sinica.edu.tw/fnzz/view.php?year={year}&month={month:02d}')

    # Wait for the JavaScript to execute and render the table
    time.sleep(1)  # Adjust the sleep time if necessary

    # Locate the table
    table = driver.find_element(By.XPATH, '//table')

    # Extract table headers
    headers = [header.text for header in table.find_elements(By.XPATH, './/th')]

    # Extract table rows
    rows = []
    for row in table.find_elements(By.XPATH, './/tr')[1:]:
        cells = row.find_elements(By.XPATH, './/td')
        rows.append([cell.text for cell in cells])

    # Create a DataFrame
    df = pd.DataFrame(rows, columns=headers)

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)

    print('Table data saved to ' + filename)


def download_images(output_dir: str, year: int, month: int):
    b = f"{year:02d}{month:02d}"
    for p in range(1, 1000):
        filename = output_dir + f"{b}_{p:04d}.jpg"
        if os.path.exists(filename):
            print(f"Image {filename} already exists.")
            continue
        url = f"https://mhdb.mh.sinica.edu.tw/fnzz/giveimage.php?b={b}&p={p:04d}"
        response = requests.get(url, stream=True, verify=False)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            if img.size == (500, 500):
                print(f"Image {filename} is empty in the database.")
                continue
            else:
                with open(filename, 'wb') as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                print(f"Image {filename} downloaded.")
        else:
            print(f"Unable to download image. HTTP response code: {response.status_code}")
            break


def download_sample_images(output_dir: str, urls: list):
    for i in range(len(urls)):
        filename = output_dir + f"{i+1}.jpg"
        if os.path.exists(filename):
            print(f"Image {filename} already exists.")
            continue
        response = requests.get(urls[i], stream=True, verify=False)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            if img.size == (500, 500):
                print(f"Image {filename} is empty in the database.")
                continue
            else:
                with open(filename, 'wb') as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                print(f"Image {filename} downloaded.")
        else:
            print(f"Unable to download image. HTTP response code: {response.status_code}")
            break


"""
Example usage:
    python download.py -t -i -s
This command will download tables, images, and sample images.
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tables', action='store_true', help='Download tables.')
    parser.add_argument('-i', '--images', action='store_true', help='Download images.')
    parser.add_argument('-s', '--samples', action='store_true', help='Download sample images.')
    args = parser.parse_args()

    if args.tables:
        if not os.path.exists(table_path):
            os.makedirs(table_path)
        driver = webdriver.Chrome()
        for year in range(15, 32):
            for month in range(1, 13):
                download_tables(table_path, year, month)
        driver.quit()
    if args.images:
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        for year in range(15, 32):
            for month in range(1, 13):
                download_images(image_path, year, month)
    if args.samples:
        if not os.path.exists(sample_image_path):
            os.makedirs(sample_image_path)
        download_sample_images(sample_image_path, sample_urls)