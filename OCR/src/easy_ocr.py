import os
import cv2
import time
import easyocr
import numpy as np


input_path = 'target/samples/images/'
output_text_path = 'output/samples/text_easyocr/'
output_position_path = 'output/samples/positions_easyocr/'
output_image_path = 'output/samples/images_easyocr/'
reader = easyocr.Reader(['ch_tra'])


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
        image = cv2.imread(img_path)
        result = reader.readtext(img_path)

        text = ''
        positions = []
        scores = []

        for line in result:
            text_line = line[1]
            text += text_line + '\n'
            box = line[0]
            score = line[2]
            scores.append(score)
            # Convert box to [xmin, ymin, xmax, ymax]
            x_list, y_list = [int(p[0]) for p in box], [int(p[1]) for p in box]
            positions.append([min(x_list), min(y_list), max(x_list), max(y_list)])
            cv2.polylines(image, [np.array(box).astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=1)

        text_path = output_text_path + img_name.split('.')[0] + '.txt'
        position_path = output_position_path + img_name.split('.')[0] + '.txt'
        image_path = output_image_path + img_name

        with open(text_path, 'w') as f:
            f.write(text)
            print("Save text to " + text_path)

        with open(position_path, 'w') as f:
            for position, score in zip(positions, scores):
                for p in position:
                    f.write(str(p) + ',')
                f.write('1,' + str(score) + '\n')
            print("Save position to " + position_path)

        cv2.imwrite(image_path, image)
        print("Save image to " + image_path)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f}s, Average time per file: {(end_time - start_time) / len(os.listdir(input_path)):.2f}s")