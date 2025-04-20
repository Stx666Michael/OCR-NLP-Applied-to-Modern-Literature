import os
import numpy as np
from collections import defaultdict
from utils import read_annotations, compute_precision_recall, calculate_ap, calculate_metrics


def get_map_results_for_files(position_gt_path, position_pred_path):
    map_results = []
    map_results_for_file = defaultdict(list)

    for iou_threshold in np.arange(0.5, 1.0, 0.05):
        ground_truths_all = []
        predictions_all = []

        for file in sorted(os.listdir(position_gt_path)):
            if file.endswith('.txt'):
                gt_file_path = os.path.join(position_gt_path, file)
                pred_file_path = os.path.join(position_pred_path, file)
                ground_truths = read_annotations(gt_file_path, is_gt=True)
                predictions = read_annotations(pred_file_path, is_gt=False)
                ground_truths_all.append(ground_truths)
                predictions_all.append(predictions)

                precision, recall = compute_precision_recall(predictions, ground_truths, iou_threshold)
                ap = calculate_ap(precision, recall)
                map_results_for_file[file].append(ap)

        ground_truths_all = [item for sublist in ground_truths_all for item in sublist]
        predictions_all = [item for sublist in predictions_all for item in sublist]     

        precision, recall = compute_precision_recall(predictions_all, ground_truths_all, iou_threshold)
        ap = calculate_ap(precision, recall)
        map_results.append(ap)

        print(f'IoU threshold: {iou_threshold:.2f}, mAP: {map_results[-1]:.4f}')

    print(f'mAP@[0.5:0.95]: {np.mean(map_results):.4f}')

    return map_results, map_results_for_file


def get_cer_bleu_for_files(text_gt_path, text_pred_path):
    CERs = []
    BLEU_4s = []
    filenames = sorted(os.listdir(text_gt_path))

    for filename in filenames:

        with open(text_gt_path + filename, 'r') as f:
            ground_truth = f.read().strip()

        with open(text_pred_path + filename, 'r') as f:
            prediction = f.read().strip()

        CER, BLEU_4 = calculate_metrics(ground_truth, prediction)
        CERs.append(CER)
        BLEU_4s.append(BLEU_4)

        print(f'File: {filename}, CER: {CER}, BLEU-4: {BLEU_4}')

    CER_avg = sum(CERs) / len(CERs)
    BLEU_4_avg = sum(BLEU_4s) / len(BLEU_4s)

    print(f'Average Character Error Rate (CER): {CER_avg}')
    print(f'Average BLEU-4 Score: {BLEU_4_avg}')

    return CERs, BLEU_4s