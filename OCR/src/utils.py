import os
import cv2
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Tuple
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def filter_raw_output(json_path, output_positions_path, output_text_path, line_threshold, layout_lambda):
    with open(json_path, "r") as file:
        data = json.load(file)

    filename = data["filename"][:-4]
    output_position_file_path = os.path.join(output_positions_path, filename + ".txt")
    output_text_file_path = os.path.join(output_text_path, filename + ".txt")
    text_lines = []
    positions = []
    labels = []

    with open(output_position_file_path, "w") as file:
        for line in data["lines"]:
            position = line["position"]
            label = line["label"]
            score = line["score"]
            if score < line_threshold:
                continue
            text_lines.append(line["text"])
            positions.append(position)
            labels.append(label)
            file.write(f"{position[0]},{position[1]},{position[2]},{position[3]},{label},{score}\n")

    with open(output_text_file_path, "w") as file:
        ordered_text, _ = order_text(text_lines, positions, labels, line_break=False, layout_lambda=layout_lambda)
        file.write(ordered_text)


def calculate_iou(box1, box2):
    # Unpack the bounding box coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate the coordinates of the intersection rectangle
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Compute the area of intersection
    inter_width = max(inter_x_max - inter_x_min, 0)
    inter_height = max(inter_y_max - inter_y_min, 0)
    intersection_area = inter_width * inter_height

    # Compute the area of both the prediction and ground-truth rectangles
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Compute the area of union
    union_area = box1_area + box2_area - intersection_area

    # Compute the IoU
    if union_area == 0:
        return 0
    else:
        return intersection_area / union_area


def compute_precision_recall(predictions, ground_truths, iou_threshold=0.5):
    # Sort predictions by confidence score in descending order
    predictions = sorted(predictions, key=lambda x: x[5], reverse=True)
    tp = []
    fp = []
    
    # Track ground truths that have been matched
    matched_gt = set()
    
    for pred in predictions:
        pred_box = pred[:4]
        best_iou = 0
        best_gt_idx = -1
        
        # Find the best IoU with ground truths
        for gt_idx, gt_box in enumerate(ground_truths):
            iou = calculate_iou(pred_box, gt_box[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp.append(1)
            fp.append(0)
            matched_gt.add(best_gt_idx)
        else:
            tp.append(0)
            fp.append(1)
    
    # Convert tp and fp lists to cumulative counts
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    
    # Compute recall and precision
    recall = tp / (len(ground_truths))
    precision = tp / (tp + fp)

    return precision, recall


def calculate_ap(precision, recall):
    # Sort recall and precision values by recall
    sorted_indices = np.argsort(recall)
    recall = np.array(recall)[sorted_indices]
    precision = np.array(precision)[sorted_indices]

    # Append endpoints for recall
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))

    # Compute the precision-recall area
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    
    return ap


def calculate_map(ap_by_class):
    # Mean of average precisions
    mAP = sum(ap_by_class.values()) / len(ap_by_class) if ap_by_class else 0
    return mAP


def read_annotations(file_path, is_gt=True):
    # Read the annotations from a file
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    annotations = []
    for line in lines:
        if is_gt:
            x_min, y_min, x_max, y_max, class_label = map(int, line.strip().split(","))
            annotations.append([x_min, y_min, x_max, y_max, class_label])
        else:
            x_min, y_min, x_max, y_max, class_label, confidence = map(float, line.strip().split(","))
            annotations.append([x_min, y_min, x_max, y_max, class_label, confidence])

    return annotations


def edit_distance(s1: str, s2: str) -> int:
        """ 
        Calculate the Levenshtein distance between two strings 
        """
        if len(s1) < len(s2):
            return edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


def calculate_metrics(ground_truth: str, prediction: str) -> Tuple[float, float]:
    # Character Error Rate (CER)
    cer_distance = edit_distance(ground_truth, prediction)
    CER = cer_distance / max(len(ground_truth), len(prediction))

    # BLEU-4 Score
    smoothing_function = SmoothingFunction().method1
    BLEU_4 = sentence_bleu([list(ground_truth)], list(prediction), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)

    return CER, BLEU_4


def draw_boxes(input_image_path, pred_pos_label):
    input_image = cv2.imread(input_image_path)

    for i in range(len(pred_pos_label)):
        x_min, y_min, x_max, y_max = pred_pos_label[i][1]
        label = pred_pos_label[i][2]
        box_color = (0, 0, 255) if label == 1 else (0, 255, 0)
        cv2.rectangle(input_image, (x_min, y_min), (x_max, y_max), box_color, 1)
        cv2.putText(input_image, str(i+1), (x_max, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    return input_image


def order_text(text_lines, positions, labels, line_break, layout_lambda):
    # Get the vertical center of the image
    y_min = min([pos[1] for pos in positions])
    y_max = max([pos[3] for pos in positions])
    y_center = (y_min + y_max) / 2

    # Zip the predictions and positions
    pred_pos_label = list(zip(text_lines, positions, labels))
    vertical_upper_count = len([pp for pp in pred_pos_label if pp[2] == 1 and (pp[1][1]+pp[1][3])/2 < y_center])
    vertical_lower_count = len([pp for pp in pred_pos_label if pp[2] == 1 and (pp[1][1]+pp[1][3])/2 >= y_center])
    layout_threshold = layout_lambda * len(pred_pos_label)

    # Sort the detected lines according to text layout
    if vertical_upper_count > layout_threshold and vertical_lower_count > layout_threshold:
        # Sort the detected lines from right to left
        pred_pos_label = sorted(pred_pos_label, key=lambda x: (x[1][0]+x[1][2]), reverse=True)
        # Split the detected lines into two rows
        upper = [pp for pp in pred_pos_label if (pp[1][1]+pp[1][3])/2 < y_center]
        lower = [pp for pp in pred_pos_label if (pp[1][1]+pp[1][3])/2 >= y_center]
        pred_pos_label = upper + lower
    else:
        # Split the detected lines into vertical and horizontal lines
        vertical_lines = [pp for pp in pred_pos_label if pp[2] == 1]
        horizontal_lines = [pp for pp in pred_pos_label if pp[2] == 2]
        # Sort the vertical lines from right to left
        vertical_lines = sorted(vertical_lines, key=lambda x: (x[1][0]+x[1][2]), reverse=True)
        # Sort the horizontal lines from top to bottom
        horizontal_lines = sorted(horizontal_lines, key=lambda x: (x[1][1]+x[1][3]))
        pred_pos_label = vertical_lines + horizontal_lines

    # Concatenate the text in the detected lines
    ordered_text = ''
    for text, _, _ in pred_pos_label:
        ordered_text += text + "\n" if line_break else text

    return ordered_text, pred_pos_label


def predict(filename, input_path, output_path, model_line, model_char, device, character_list, num_classes_char, line_size, output_raw=False, output_text=False, output_positions=False, output_image=False, output_confidence=False, return_image_and_confidence=False, line_break=False, overwrite=True, layout_lambda=0.25):
    input_image_path = input_path + filename
    output_image_path = output_path + "images/" + filename
    output_text_path = output_path + "text/" + filename[:-4] + ".txt"
    output_positions_path = output_path + "positions/" + filename[:-4] + ".txt"
    output_confidence_path = output_path + "confidence.csv"
    output_raw_path = output_path + "raw/" + filename[:-4] + ".json"

    if not overwrite and os.path.exists(output_raw_path):
        print("Skip:", filename)
        return
    if not os.path.exists(output_path + "raw/") and output_raw:
        os.makedirs(output_path + "raw/")
    if not os.path.exists(output_path + "text/") and output_text:
        os.makedirs(output_path + "text/")
    if not os.path.exists(output_path + "positions/") and output_positions:
        os.makedirs(output_path + "positions/")
    if not os.path.exists(output_path + "images/") and output_image:
        os.makedirs(output_path + "images/")

    input_image = np.array(Image.open(input_image_path).convert('L'))
    input_image = torch.from_numpy(input_image).unsqueeze(0).unsqueeze(0).float().to(device)

    prediction_lines = model_line(input_image)[0]
    patch_line_images = []
    patch_line_positions = []
    patch_line_labels = []
    patch_line_scores = []

    # If no lines are detected, save an empty text file and the original image
    if len(prediction_lines['boxes']) == 0:
        if output_raw:
            with open(output_raw_path, "w", encoding='UTF-8') as file:
                file.write("")
            print("Save raw output:", output_raw_path)

        if output_text:
            with open(output_text_path, "w", encoding='UTF-8') as file:
                file.write("")
            print("Save text:", output_text_path)

        if output_positions:
            with open(output_positions_path, "w", encoding='UTF-8') as file:
                file.write("")
            print("Save positions:", output_positions_path)

        if output_image:
            input_image = cv2.imread(input_image_path)
            cv2.imwrite(output_image_path, input_image)
            print("Save image:", output_image_path)
        return

    # Extract patches from the detected lines
    for i in range(len(prediction_lines['boxes'])):
        x_min, y_min, x_max, y_max = prediction_lines['boxes'][i].int().tolist()
        label = prediction_lines['labels'][i].item()
        # Extract patches from the image
        patch_line_image = input_image[:, :, y_min:y_max, x_min:x_max]

        # Resize the patches to the required size
        if label == 1: # Vertical line
            patch_line_image = nn.functional.interpolate(patch_line_image, size=line_size, mode='bilinear')
            patch_line_image = torch.rot90(patch_line_image, dims=[2, 3])
        else: # Horizontal line
            patch_line_image = nn.functional.interpolate(patch_line_image, size=line_size[::-1], mode='bilinear')

        patch_line_images.append(patch_line_image)
        patch_line_positions.append((x_min, y_min, x_max, y_max))
        patch_line_labels.append(label)
        patch_line_scores.append(prediction_lines['scores'][i].item())

    # Detect characters in the lines in a batch
    patch_line_images = torch.cat(patch_line_images).to(device)
    patch_line_predictions = model_char(patch_line_images)

    # Decode the character predictions
    predictions_text = []
    predictions_prob = []
    for i in range(len(patch_line_positions)):
        prediction_index = torch.argmax(patch_line_predictions[:, i, :], dim=1)
        prediction_prob = patch_line_predictions[:, i, :].softmax(dim=1).max(dim=1).values
        prediction_text = ""
        for index, prob in zip(prediction_index, prediction_prob):
            if index.item() != num_classes_char:
                prediction_text += character_list[index.item()]
                predictions_prob.append(prob.item())
        predictions_text.append(prediction_text)

    # Save the confidence of the predictions
    if output_confidence and len(predictions_prob) > 0:
        with open(output_confidence_path, "a", encoding='UTF-8') as file:
            file.write(filename[:-4] + "," + str(np.mean(patch_line_scores)) + "," + str(np.mean(predictions_prob)) + "\n")

    # Get ordered text according to the positions of the detected lines
    ordered_text, pred_pos_label = order_text(predictions_text, patch_line_positions, patch_line_labels, line_break, layout_lambda)

    # Save the raw output of the system in json format
    if output_raw:
        lines = []
        for i in range(len(patch_line_positions)):    
            line = {
                "position": patch_line_positions[i],
                "label": patch_line_labels[i],
                "score": patch_line_scores[i],
                "text": predictions_text[i]
            }
            lines.append(line)
        raw_output = {
            "filename": filename,
            "confidence": np.mean(predictions_prob) if len(predictions_prob) > 0 else 0,
            "lines": lines
        }
        with open(output_raw_path, "w", encoding='UTF-8') as file:
            json.dump(raw_output, file)
            print("Save raw output:", output_raw_path)

    # Save the predicted text to a text file
    if output_text:
        with open(output_text_path, "w", encoding='UTF-8') as file:
            file.write(ordered_text)
            print("Save text:", output_text_path)

    # Save the positions of the detected lines to a text file
    if output_positions:
        with open(output_positions_path, "w", encoding='UTF-8') as file:
            for pos, label, score in zip(patch_line_positions, patch_line_labels, patch_line_scores):
                for p in pos:
                    file.write(str(p) + ",")
                file.write(str(label) + "," + str(score) + "\n")
            print("Save positions:", output_positions_path)

    # Save the image with bounding boxes and text order
    if output_image:
        image_output = draw_boxes(input_image_path, pred_pos_label)
        cv2.imwrite(output_image_path, image_output)
        print("Save image:", output_image_path)

    if return_image_and_confidence:
        image_output = draw_boxes(input_image_path, pred_pos_label)
        confidence = np.mean(predictions_prob) if len(predictions_prob) > 0 else 0
        return ordered_text, image_output, confidence
    else:
        return ordered_text


def pred_eval(input_path, output_path, gt_path, model_line, model_char, device, character_list, num_classes_char, line_size):
    CERs = []
    BLEU_4s = []
    file_list = [f for f in os.listdir(input_path) if f.endswith('.jpg')]

    for filename in file_list:
        prediction = predict(filename, input_path, output_path, model_line, model_char, device, character_list, num_classes_char, line_size)
        
        with open(gt_path + filename[:-4] + '.txt', 'r', encoding='UTF-8') as f:
            ground_truth = f.read().strip()

        CER, BLEU_4 = calculate_metrics(ground_truth, prediction)
        CERs.append(CER)
        BLEU_4s.append(BLEU_4)

        print(f'File: {filename}, CER: {CER}, BLEU-4: {BLEU_4}')

        CER_avg = sum(CERs) / len(CERs)
        BLEU_4_avg = sum(BLEU_4s) / len(BLEU_4s)

    print(f'Average Character Error Rate (CER): {CER_avg}')
    print(f'Average BLEU-4 Score: {BLEU_4_avg}')