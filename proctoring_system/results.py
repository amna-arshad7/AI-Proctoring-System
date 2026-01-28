import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def calculate_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-8)
    return iou

def evaluate_detection(y_true, y_pred, iou_threshold=0.5, show_matrix=True):

    ious = []
    true_labels = []
    pred_labels = []

    min_len = min(len(y_true), len(y_pred))
    for i in range(min_len):
        true_box, true_label = y_true[i]
        pred_box, pred_label = y_pred[i]
        iou = calculate_iou(true_box, pred_box)
        ious.append(iou)
        true_labels.append(true_label)
        pred_labels.append(pred_label)

    # IoU metrics
    avg_iou = np.mean(ious) if ious else 0
    above_thresh = np.sum(np.array(ious) >= iou_threshold) if ious else 0

    print(f"Average IoU: {avg_iou:.3f}")
    print(f"Detections above {iou_threshold} IoU: {above_thresh}/{len(ious)}")

    # Confusion matrix & accuracy
    if show_matrix:
        cm = confusion_matrix(true_labels, pred_labels, labels=list(set(true_labels + pred_labels)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(set(true_labels + pred_labels)))
        disp.plot(cmap="Blues")
        plt.title("Detection Confusion Matrix")
        plt.show()

    accuracy = np.mean([tl == pl for tl, pl in zip(true_labels, pred_labels)]) if true_labels else 0
    print(f"Detection Accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    
    y_true = [
        ([100, 50, 200, 180], "headset"),
        ([300, 100, 400, 200], "cell phone")
    ]
    y_pred = [
        ([98, 45, 202, 182], "headset"),
        ([305, 97, 399, 203], "cell phone")
    ]
    evaluate_detection(y_true, y_pred)
