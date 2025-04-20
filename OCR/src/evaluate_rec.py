import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from utils import calculate_metrics, read_annotations, compute_precision_recall, calculate_ap


output_plot_path = "plots/"
gt_path_text = 'target/samples/text/'
pred_path_text = 'output/samples/text/'
gt_path_positions = "target/samples/positions/"
pred_path_positions = "output/samples/positions/"
confidence_path = 'output/samples/confidence.csv'

font_path = '/Users/tianxiangsong/Library/Fonts/cambria.ttf'
font_l = FontProperties(fname=font_path, size=36)
font_s = FontProperties(fname=font_path, size=24)


if __name__ == "__main__":
    CERs = []
    BLEU_4s = []
    mAPs = []
    filenames = os.listdir(gt_path_text)

    for filename in filenames:
        with open(gt_path_text + filename, 'r', encoding='UTF-8') as f:
            ground_truth = f.read().strip()

        with open(pred_path_text + filename, 'r', encoding='UTF-8') as f:
            prediction = f.read().strip()

        CER, BLEU_4 = calculate_metrics(ground_truth, prediction)
        CERs.append(CER)
        BLEU_4s.append(BLEU_4)

        ap_list = []
        ground_truth = read_annotations(os.path.join(gt_path_positions, filename), is_gt=True)
        prediction = read_annotations(os.path.join(pred_path_positions, filename), is_gt=False)

        for iou_threshold in np.arange(0.5, 1.0, 0.05):
            precision, recall = compute_precision_recall(prediction, ground_truth, iou_threshold=iou_threshold)
            ap = calculate_ap(precision, recall)
            ap_list.append(ap)

        mAPs.append(np.mean(ap_list))

        print(f'File: {filename}, CER: {CER}, BLEU-4: {BLEU_4}, mAP@[0.5:0.95]: {np.mean(ap_list)}')

    CER_avg = sum(CERs) / len(CERs)
    BLEU_4_avg = sum(BLEU_4s) / len(BLEU_4s)

    # Sort metrics based on BLEU-4 - CER
    metrics = list(zip(filenames, CERs, BLEU_4s, mAPs))
    metrics.sort(key=lambda x: x[2]-x[1], reverse=True)
    filenames, CERs, BLEU_4s, mAPs = zip(*metrics)

    print(f'Average Character Error Rate (CER): {CER_avg}')
    print(f'Average BLEU-4 Score: {BLEU_4_avg}')
    print(f'Average mAP@[0.5:0.95]: {np.mean(mAPs)}')

    x = np.arange(len(metrics))  # the label locations

    # Plot CER, BLEU-4 and mAP for each file
    plt.figure(figsize=(30, 10))
    plt.plot(x, CERs, color='red', label='CER', marker='o', linestyle='--')
    plt.plot(x, BLEU_4s, color='blue', label='BLEU-4', marker='o', linestyle='--')
    plt.plot(x, np.array(BLEU_4s) - np.array(CERs), color='orange', label='BLEU-4 - CER', marker='o')
    plt.plot(x, mAPs, color='green', label='mAP@[0.5:0.95]', marker='o')

    plt.xlabel('Filename', fontproperties=font_l)
    plt.ylabel('Value', fontproperties=font_l)
    plt.title('Character error rate (CER), BLEU-4 and mAP for each file, ordered by (BLEU-4 - CER)', fontproperties=font_l)
    plt.xticks(x, [filename[:-4] for filename in filenames], fontproperties=font_s)
    plt.yticks(fontproperties=font_s)
    plt.legend(prop=font_s)
    plt.savefig(os.path.join(output_plot_path, "CER_BLEU_mAP.png"), bbox_inches='tight')

    # Plot CER, BLEU-4 and confidences for each file
    confidences = {}
    with open(confidence_path, 'r', encoding='UTF-8') as f:
        confidences_list = f.readlines()
    for line in confidences_list:
        line = line.strip().split(',')
        confidences[line[0]] = (line[1], line[2])

    # Sort confidences based on filenames
    confidences_det = [float(confidences[filename[:-4]][0]) for filename in filenames]
    confidences_rec = [float(confidences[filename[:-4]][1]) for filename in filenames]

    width = 0.4  # the width of the bars
    plt.figure(figsize=(30, 10))
    plt.bar(x - width/2, CERs, width, color='red', label='CER')
    plt.bar(x + width/2, BLEU_4s, width, color='blue', label='BLEU-4')
    plt.plot(x, confidences_det, color='green', label='Det. Confidence', marker='o')
    plt.plot(x, confidences_rec, color='orange', label='Rec. Confidence', marker='o')

    plt.xlabel('Filename', fontproperties=font_l)
    plt.ylabel('Value', fontproperties=font_l)
    plt.title('Character error rate (CER), BLEU-4 and confidences for each file, ordered by (BLEU-4 - CER)', fontproperties=font_l)
    plt.xticks(x, [filename[:-4] for filename in filenames], fontproperties=font_s)
    plt.yticks(fontproperties=font_s)
    plt.legend(prop=font_s, loc='upper right')
    plt.savefig(os.path.join(output_plot_path, "CER_BLEU_Confidence.png"), bbox_inches='tight')