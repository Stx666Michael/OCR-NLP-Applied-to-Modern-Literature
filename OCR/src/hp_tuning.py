import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties
from utils import filter_raw_output, read_annotations, compute_precision_recall, calculate_ap, calculate_metrics


output_plot_path = "plots/"
input_path = "output/samples/raw/"
output_positions_path = "output/samples/positions/"
output_text_path = "output/samples/text/"
position_gt_path = "target/samples/positions/"
text_gt_path = "target/samples/text/"

line_threshold_default = 0
layout_lambda_default = 0.25
line_threshold_optimal = 0.6
layout_lambda_optimal = 0.2

font_path = '/Users/tianxiangsong/Library/Fonts/cambria.ttf'
font_l = FontProperties(fname=font_path, size=36)
font_s = FontProperties(fname=font_path, size=24)
font_xs = FontProperties(fname=font_path, size=18)


def detection_tuning():
    line_threshold_map = defaultdict(list)

    for line_threshold in np.arange(0, 1.0, 0.1):
        # Filter raw output
        for file in os.listdir(input_path):
            if file.endswith(".json"):
                json_path = os.path.join(input_path, file)
                filter_raw_output(json_path, output_positions_path, output_text_path, line_threshold, layout_lambda_default)

        # Evaluate detection results
        for iou_threshold in np.arange(0.5, 1.0, 0.05):
            ground_truths_all = []
            predictions_all = []
            for file in os.listdir(position_gt_path):
                if file.endswith(".txt"):
                    gt_file_path = os.path.join(position_gt_path, file)
                    pred_file_path = os.path.join(output_positions_path, file)
                    ground_truths = read_annotations(gt_file_path, is_gt=True)
                    predictions = read_annotations(pred_file_path, is_gt=False)
                    ground_truths_all.append(ground_truths)
                    predictions_all.append(predictions)

            # Flatten the lists to compute mAP across all files
            ground_truths_all = [item for sublist in ground_truths_all for item in sublist]
            predictions_all = [item for sublist in predictions_all for item in sublist]

            # Compute precision and recall
            precision, recall = compute_precision_recall(predictions_all, ground_truths_all, iou_threshold=iou_threshold)

            # Compute average precision
            ap = calculate_ap(precision, recall)
            line_threshold_map[line_threshold].append(ap)
        
        print(f"Text Detection Threshold = {line_threshold:.1f}, mAP@[0.5:0.95] = {np.mean(line_threshold_map[line_threshold]):.3f}")

    # Plot the mAP versus each IoU threshold, for each line threshold
    plt.figure(figsize=(30, 10))
    for line_threshold, ap_list in line_threshold_map.items():
        plt.plot(np.arange(0.5, 1.0, 0.05), ap_list, marker='o', label=f'DT = {line_threshold:.1f}, mAP@[0.5:0.95] = {np.mean(ap_list):.3f}')

    plt.xlabel('IoU Threshold', fontproperties=font_l)
    plt.ylabel('mAP', fontproperties=font_l)
    plt.title('Mean average precision versus IoU threshold for different detection thresholds', fontproperties=font_l)
    plt.xticks(fontproperties=font_s)
    plt.yticks(fontproperties=font_s)
    plt.legend(prop=font_s)
    plt.savefig(os.path.join(output_plot_path, 'mAP_vs_IoU_DT.png'), bbox_inches='tight')


def recognition_tuning():
    threshold_lambda_cer = defaultdict(dict)
    threshold_lambda_bleu = defaultdict(dict)
    threshold_range = np.arange(0, 1.0, 0.1)
    lambda_range = np.arange(0.05, 0.5, 0.05)
    filenames = os.listdir(text_gt_path)

    for line_threshold in threshold_range:
        for layout_lambda in lambda_range:
            # Filter raw output
            for file in os.listdir(input_path):
                if file.endswith(".json"):
                    json_path = os.path.join(input_path, file)
                    filter_raw_output(json_path, output_positions_path, output_text_path, line_threshold, layout_lambda)
                    
            # Evaluate recognition results
            CERs = []
            BLEU_4s = []
            for filename in filenames:
                with open(text_gt_path + filename, 'r', encoding='UTF-8') as f:
                    ground_truth = f.read().strip()

                with open(output_text_path + filename, 'r', encoding='UTF-8') as f:
                    prediction = f.read().strip()

                CER, BLEU_4 = calculate_metrics(ground_truth, prediction)
                CERs.append(CER)
                BLEU_4s.append(BLEU_4)
            
            threshold_lambda_cer[line_threshold][layout_lambda] = np.mean(CERs)
            threshold_lambda_bleu[line_threshold][layout_lambda] = np.mean(BLEU_4s)

            print(f'Text Detection Threshold: {line_threshold:.1f}, Layout Lambda: {layout_lambda:.2f}, Average CER: {np.mean(CERs):.6f}, Average BLEU-4: {np.mean(BLEU_4s):.6f}, BLEU-4 - CER: {np.mean(BLEU_4s) - np.mean(CERs):.6f}')

    # Plot the CER and BLEU-4 scores versus layout lambda, for each text detection threshold
    plt.figure(figsize=(30, 10))
    CER_min = {"value": 1, "line_threshold": 0, "layout_lambda": 0}
    BLEU_4_max = {"value": 0, "line_threshold": 0, "layout_lambda": 0}
    for line_threshold in threshold_range:
        CER_avg_list = [threshold_lambda_cer[line_threshold][layout_lambda] for layout_lambda in lambda_range]
        BLEU_4_avg_list = [threshold_lambda_bleu[line_threshold][layout_lambda] for layout_lambda in lambda_range]
        BLEU_4_minus_CER_list = np.array(BLEU_4_avg_list) - np.array(CER_avg_list)
        # Update the minimum CER and maximum BLEU-4 values with corresponding λ and DT values
        if np.min(CER_avg_list) < CER_min["value"]:
            CER_min["value"] = np.min(CER_avg_list)
            CER_min["line_threshold"] = line_threshold
            CER_min["layout_lambda"] = lambda_range[np.argmin(CER_avg_list)]

        if np.max(BLEU_4_avg_list) > BLEU_4_max["value"]:
            BLEU_4_max["value"] = np.max(BLEU_4_avg_list)
            BLEU_4_max["line_threshold"] = line_threshold
            BLEU_4_max["layout_lambda"] = lambda_range[np.argmax(BLEU_4_avg_list)]

        plt.plot(lambda_range, CER_avg_list, marker='o', color='red', alpha=0.1, linestyle='--')
        plt.plot(lambda_range, BLEU_4_avg_list, marker='o', color='blue', alpha=0.1, linestyle='--')
        plt.plot(lambda_range, BLEU_4_minus_CER_list, marker='o', label=f'BLEU-4 - CER (max={np.max(BLEU_4_minus_CER_list):.3f} with λ={lambda_range[np.argmax(BLEU_4_minus_CER_list)]:.2f}), DT = {line_threshold:.1f}')

    plt.xlabel('Layout Parameter λ', fontproperties=font_l)
    plt.ylabel('Scores', fontproperties=font_l)
    plt.title('Character error rate (CER) and BLEU-4 score versus λ for different detection thresholds', fontproperties=font_l)
    plt.xticks(fontproperties=font_s)
    plt.yticks(fontproperties=font_s)

    red_lines = mlines.Line2D([], [], color='red', label=f'CER (min={CER_min["value"]:.3f} with λ={CER_min["layout_lambda"]:.2f} and DT={CER_min["line_threshold"]:.1f})', alpha=0.1, linestyle='--', marker='o')
    blue_lines = mlines.Line2D([], [], color='blue', label=f'BLEU-4 (max={BLEU_4_max["value"]:.3f} with λ={BLEU_4_max["layout_lambda"]:.2f} and DT={BLEU_4_max["line_threshold"]:.1f})', alpha=0.1, linestyle='--', marker='o')
    handles, _ = plt.gca().get_legend_handles_labels()
    handles.extend([red_lines, blue_lines])

    plt.legend(handles=handles, prop=font_xs, ncol=2)
    plt.savefig(os.path.join(output_plot_path, 'CER_BLEU.png'), bbox_inches='tight')


def reset_output_files():
    for file in os.listdir(input_path):
        if file.endswith(".json"):
            json_path = os.path.join(input_path, file)
            filter_raw_output(json_path, output_positions_path, output_text_path, line_threshold_optimal, layout_lambda_optimal)


if __name__ == "__main__":
    detection_tuning()
    recognition_tuning()
    reset_output_files()