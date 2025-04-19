import os
import json
import argparse
import pandas as pd
from typing import List, Tuple
from IPython.display import display

# Calculate Intersection over Union (IoU) between two 1D segments
def calculate_iou(segment_a: List[float], segment_b: List[float]) -> float:
    start_a, end_a = segment_a
    start_b, end_b = segment_b
    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)
    inter_length = max(0, inter_end - inter_start)
    union_length = max(end_a, end_b) - min(start_a, start_b)
    return inter_length / union_length if union_length > 0 else 0.0

# Compute event-level metrics from labeled and predicted segments
def evaluate_metrics(args) -> None:
    json_path = args.json_path
    with open(json_path, 'r') as f:
        data = json.load(f)

    count_iou_50 = count_iou_30 = count_iou_70 = 0
    tp = fp = fn = tn = 0
    sum_iou = 0

    for item in data:
        label = item.get("label", "")
        predict = item.get("predict", "")

        if '{' in label and '}' in label:
            label = '[' + label.split('{')[1].split('}')[0] + ']'
        elif 'yes' in label.lower():
            label = 'yes'
        else:
            label = 'no'

        if label != "no":
            if predict != "no":
                tp += 1  # GT = yes; model = yes
                try:
                    gt_coords = json.loads(label) if label != "yes" else []
                    pred_coords = json.loads(predict)
                    iou = calculate_iou(gt_coords, pred_coords) if gt_coords else 0.0

                    if iou > 0.3:
                        count_iou_30 += 1
                    if iou > 0.5:
                        count_iou_50 += 1
                    if iou > 0.7:
                        count_iou_70 += 1

                    sum_iou += iou
                except json.JSONDecodeError:
                    print(f"Error decoding JSON for item with image: {item.get('image')}")
                    continue
            else:
                fn += 1  # GT = yes; model = no
        else:
            if predict != "no":
                fp += 1  # GT = no; model = yes
            else:
                tn += 1  # GT = no; model = no

    # Compute classification metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    precision_P = tp / (tp + fp + 1e-10)
    recall_P = tp / (tp + fn + 1e-10)
    f1_P = 2 * (precision_P * recall_P) / (precision_P + recall_P + 1e-10)

    precision_N = tn / (tn + fn + 1e-10)
    recall_N = tn / (tn + fp + 1e-10)
    f1_N = 2 * (precision_N * recall_N) / (precision_N + recall_N + 1e-10)

    ap30 = count_iou_30 / tp if tp else 0
    ap50 = count_iou_50 / tp if tp else 0
    ap70 = count_iou_70 / tp if tp else 0
    miou = sum_iou / tp if tp else 0

    metrics_list = [
        accuracy, precision_P, recall_P, f1_P,
        precision_N, recall_N, f1_N,
        ap30, ap50, ap70, miou
    ]
    metrics_names = [
        "Accuracy", "Precision_Accident", "Recall_Accident", "F1_Score_Accident",
        "Precision_Normal", "Recall_Normal", "F1_Score_Normal",
        "AP30", "AP50", "AP70", "mIoU"
    ]

    # Format to 4 decimal places
    formatted_metrics = list(map(lambda x: f"{x:.4f}", metrics_list))

    # Create and display metrics DataFrame
    metrics_df = pd.DataFrame([formatted_metrics], columns=metrics_names)
    pd.set_option('display.float_format', '{:.4f}'.format)
    display(metrics_df)

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate event-level detection performance.')
    parser.add_argument('--json_path', type=str, required=True, help='Path to JSON file containing predictions and ground truths')
    return parser.parse_args()

# Main entry point
if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    evaluate_metrics(args)
