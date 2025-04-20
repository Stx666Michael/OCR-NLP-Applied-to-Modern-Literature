import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from utils import read_annotations, compute_precision_recall, calculate_ap, calculate_map


output_plot_path = "plots/"
gt_path = "target/samples/positions/"
pred_path = "output/samples/positions/"

font_path = '/Users/tianxiangsong/Library/Fonts/cambria.ttf'
font_l = FontProperties(fname=font_path, size=36)
font_s = FontProperties(fname=font_path, size=24)
font_xs = FontProperties(fname=font_path, size=18)


if __name__ == "__main__":
    precision_dict = {}
    recall_dict = {}
    ap_dict_iou = {}
    ap_dict_file = defaultdict(list)

    # iou thresholds from 0.5 to 0.95 with a step of 0.05
    for iou_threshold in np.arange(0.5, 1.0, 0.05):
        print(f"IoU Threshold: {iou_threshold:.2f}")
        ground_truths_all = []
        predictions_all = []
        for file in os.listdir(gt_path):
            if file.endswith(".txt"):
                gt_file_path = os.path.join(gt_path, file)
                pred_file_path = os.path.join(pred_path, file)
                ground_truths = read_annotations(gt_file_path, is_gt=True)
                predictions = read_annotations(pred_file_path, is_gt=False)
                ground_truths_all.append(ground_truths)
                predictions_all.append(predictions)

                # Compute precision and recall
                precision, recall = compute_precision_recall(predictions, ground_truths, iou_threshold=iou_threshold)

                # Compute average precision
                ap = calculate_ap(precision, recall)
                ap_dict_file[file].append(ap)
                print(file, "Average Precision:", ap)

        # Flatten the lists to compute mAP across all files
        ground_truths_all = [item for sublist in ground_truths_all for item in sublist]
        predictions_all = [item for sublist in predictions_all for item in sublist]

        # Compute precision and recall
        precision, recall = compute_precision_recall(predictions_all, ground_truths_all, iou_threshold=iou_threshold)
        precision_dict[iou_threshold] = precision
        recall_dict[iou_threshold] = recall

        # Compute average precision
        ap = calculate_ap(precision, recall)
        ap_dict_iou[iou_threshold] = ap
        print(f"mAP@{iou_threshold:.2f}: {ap}")
        print()

    # Compute mAP[0.5:0.95]
    mAP_05_095 = calculate_map(ap_dict_iou)
    print(f"mAP@[0.5:0.95]: {mAP_05_095}")
    
    # Plot the precision-recall curves for each IoU threshold
    plt.figure(figsize=(30, 10))
    for iou_threshold in list(ap_dict_iou.keys()):
        plt.plot(recall_dict[iou_threshold], precision_dict[iou_threshold], label=f"IoUT = {iou_threshold:.2f}, AP = {ap_dict_iou[iou_threshold]:.3f}")

    plt.xlabel("Recall", fontproperties=font_l)
    plt.ylabel("Precision", fontproperties=font_l)
    plt.title("Precision-recall curves for different IoU threshold", fontproperties=font_l)
    plt.xticks(fontproperties=font_s)
    plt.yticks(fontproperties=font_s)
    plt.legend(prop=font_s)
    plt.savefig(os.path.join(output_plot_path, "PR_curves.png"), bbox_inches='tight')

    # Plot mAP[0.5:0.95] for each file
    plt.figure(figsize=(30, 10))
    # Sort the files by mAP[0.5:0.95] in descending order
    sorted_files = sorted(ap_dict_file.items(), key=lambda x: np.mean(x[1]), reverse=True)
    for file, ap_values in sorted_files[:10]:
        plt.plot(np.arange(0.5, 1.0, 0.05), ap_values, marker='o', label=f"Filename = {file[:-4]}, mAP@[0.5:0.95] = {np.mean(ap_values):.3f}")
    for file, ap_values in sorted_files[10:]:
        plt.plot(np.arange(0.5, 1.0, 0.05), ap_values, marker='o', linestyle='--', label=f"Filename = {file[:-4]}, mAP@[0.5:0.95] = {np.mean(ap_values):.3f}")

    plt.xlabel("IoU Threshold", fontproperties=font_l)
    plt.ylabel("mAP", fontproperties=font_l)
    plt.title("Mean average precision versus IoU threshold for each test image", fontproperties=font_l)
    plt.xticks(fontproperties=font_s)
    plt.yticks(fontproperties=font_s)
    plt.legend(prop=font_xs, ncol=2)
    plt.savefig(os.path.join(output_plot_path, "mAP_vs_IoU_files.png"), bbox_inches='tight')
