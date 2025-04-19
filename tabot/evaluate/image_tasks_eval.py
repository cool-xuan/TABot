import os
import json
import argparse
import pandas as pd
from typing import List, Tuple
from IPython.display import display

# Function to calculate Intersection over Union (IoU)
def calculate_iou(box_a: List[float], box_b: List[float]) -> float:
    x_a, y_a, width_a, height_a = box_a
    x_b, y_b, width_b, height_b = box_b

    # Compute intersection
    x_inter1 = max(x_a, x_b)
    y_inter1 = max(y_a, y_b)
    x_inter2 = min(x_a + width_a, x_b + width_b)
    y_inter2 = min(y_a + height_a, y_b + height_b)

    width_inter = max(0, x_inter2 - x_inter1)
    height_inter = max(0, y_inter2 - y_inter1)
    area_inter = width_inter * height_inter

    # Compute union
    area_a = width_a * height_a
    area_b = width_b * height_b
    area_union = area_a + area_b - area_inter

    iou = area_inter / area_union
    return iou


# Function to calculate event-level evaluation metrics
def calculate_metrics(args) -> Tuple[float, float, float, float]:
    json_path = args.json_path
    with open(json_path, 'r') as f:
        data = json.load(f)

    count_iou_50 = count_iou_30 = count_iou_70 = 0
    tp = fp = fn = tn = 0
    sum_iou = 0

    for item in data:
        label = item.get("label", "")
        predict = item.get("predict", "")

        # Standardize label format
        if '[' in label and ']' in label:
            label = '[' + label.split('[')[1].split(']')[0] + ']'
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


# Argument parser
def get_args():
    parser = argparse.ArgumentParser(description='Evaluate detection and classification metrics.')
    parser.add_argument('--json_path', type=str, default='./', help='Path to JSON file')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)
    calculate_metrics(args)
