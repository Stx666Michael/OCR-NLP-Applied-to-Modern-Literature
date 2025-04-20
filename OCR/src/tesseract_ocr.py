import os
import time
import pytesseract
from PIL import Image, ImageDraw


input_path = 'target/samples/images/'
output_text_path = 'output/samples/text_tesseract/'
output_position_path = 'output/samples/positions_tesseract/'
output_image_path = 'output/samples/images_tesseract/'
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


if __name__ == '__main__':
    if not os.path.exists(output_text_path):
        os.makedirs(output_text_path)
    if not os.path.exists(output_position_path):
        os.makedirs(output_position_path)
    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)

    start_time = time.time()

    for img_name in os.listdir(input_path):
        img_path = os.path.join(input_path, img_name)
        image = Image.open(img_path)
        result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang='chi_tra')
        draw = ImageDraw.Draw(image)

        text = ''
        positions = []
        scores = []

        for i in range(len(result['text'])):
            if result['text'][i] != '':
                text += result['text'][i] + '\n'
                x, y, w, h = result['left'][i], result['top'][i], result['width'][i], result['height'][i]
                positions.append([x, y, x + w, y + h])
                scores.append(result['conf'][i])
                draw.rectangle([x, y, x + w, y + h], outline='red')

        text_path = output_text_path + img_name.split('.')[0] + '.txt'
        position_path = output_position_path + img_name.split('.')[0] + '.txt'
        image_path = output_image_path + img_name

        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
            print("Save text to " + text_path)

        with open(position_path, 'w') as f:
            for position, score in zip(positions, scores):
                for p in position:
                    f.write(str(p) + ',')
                f.write('1,' + str(score) + '\n')
            print("Save position to " + position_path)

        image.save(image_path)
        print("Save image to " + image_path)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f}s, Average time per file: {(end_time - start_time) / len(os.listdir(input_path)):.2f}s")