# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report


from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from src.misc import dist


__all__ = ['CocoEvaluator',]

from src.nn.criterion.utils import AverageMeter

""" Modified COCO evaluator for image classification evaluation.
"""
class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types, top_k=300, conf_threshold=0.92):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.top_k = top_k
        self.conf_threshold = conf_threshold

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self.all_preds = [] # Store overall predicted category for each image
        self.all_targets = [] # Store ground truth category for each image

        # Initialize AverageMeter for Top-1
        self.top1_meter = AverageMeter()

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        results, accuracy_results = self.prepare(predictions)

        # Update overall predicted categories and ground truth categories
        for acc_result in accuracy_results:
            img_id = acc_result['image_id']
            top1_label = acc_result['top1_label']
            gt_ann = self.coco_gt.loadAnns(self.coco_gt.getAnnIds(imgIds=[img_id], iscrowd=None))

            if not gt_ann:
                print(f"No ground truth found for image_id {img_id}, skipping this prediction.")
                continue

            gt_cat_id = gt_ann[0]['category_id']

            # Store overall predicted category and ground truth category for the image
            self.all_preds.append(top1_label)
            self.all_targets.append(gt_cat_id)

            # Update Top-1 accuracy
            top1_correct = 1 if top1_label == gt_cat_id else 0
            self.top1_meter.update(top1_correct)



    def synchronize_between_processes(self):
        # for iou_type in self.iou_types:
        #     self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
        #     create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])
        pass

    def accumulate(self):
        # for coco_eval in self.coco_eval.values():
        #     coco_eval.accumulate()
        pass #

    def summarize(self):
        labels = sorted(set(self.all_targets))
        predicted_labels = sorted(set(self.all_preds))
        cm_labels = sorted(set(self.all_targets) | set(self.all_preds))  # All labels that appear

        if not predicted_labels:
            print("No predictions were made.")
            return

        print("\n" + "=" * 50)
        print(f"Top-1 Accuracy: {self.top1_meter.avg:.4f}")
        print("=" * 50)

        accuracy, precision, recall = self.compute_classification_metrics()

        print("\n" + "=" * 50)
        print("============= Classification Report =============")
        print("=" * 50)
        print(f"Overall Accuracy:   {accuracy:.4f}")
        print(f"Precision (Macro):  {precision:.4f}")
        print(f"Recall (Macro):     {recall:.4f}")
        print("=" * 50)


        print("\n============ Confusion Matrix ============")
        cm = confusion_matrix(self.all_targets, self.all_preds, labels=cm_labels)
        print(f"Labels: {cm_labels}")
        print(cm)
        print("=" * 50)


        print("\n========== Classification Report ==========")
        report = classification_report(self.all_targets, self.all_preds, labels=labels, zero_division=0)
        print(report)
        print("=" * 50)


        return {
            "top1_accuracy": self.top1_meter.avg,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": cm,
            "classification_report": report
        }

    def compute_classification_metrics(self):
        accuracy = accuracy_score(self.all_targets, self.all_preds)
        precision = precision_score(self.all_targets, self.all_preds, average='macro', zero_division=0, labels=sorted(set(self.all_targets)))
        recall = recall_score(self.all_targets, self.all_preds, average='macro', zero_division=0, labels=sorted(set(self.all_targets)))
        return accuracy, precision, recall

    def prepare(self, predictions):
        results = []
        accuracy_results = []

        for img_id, prediction in predictions.items():
            labels = prediction['labels'].tolist()
            boxes = prediction['boxes'].tolist()
            scores = prediction['scores'].tolist()

            # Filter boxes using confidence threshold
            filtered_preds = []
            for label, box, score in zip(labels, boxes, scores):
                if score >= self.conf_threshold:
                    filtered_preds.append({'label': label, 'box': box, 'score': score})

            # Skip image if no detection boxes pass the threshold
            if not filtered_preds:
                continue

            # Sort by confidence and take top k
            sorted_preds = sorted(filtered_preds, key=lambda x: x['score'], reverse=True)[:self.top_k]

            # Count occurrences of each category to determine overall image prediction
            labels_np = np.array([pred['label'] for pred in sorted_preds])
            top300 = labels_np[:300] if labels_np.size >= 300 else labels_np

            # Count occurrences of each category
            if top300.size > 0:
                unique, counts = np.unique(top300, return_counts=True)
                category_counts = dict(zip(unique, counts))

                # Top-1 = category with the highest occurrence count
                top1_label = max(category_counts, key=category_counts.get)
            else:
                # Skip image if no detection boxes
                continue

            # Save detection results (for detection task)
            for pred in sorted_preds:
                result = {
                    'image_id': img_id,
                    'category_id': pred['label'],
                    'score': pred['score'],
                    'bbox': pred['box']
                }
                results.append(result)

            # Save accuracy results (for classification task)
            accuracy_results.append({
                'image_id': img_id,
                'top1_label': top1_label
            })

        return results, accuracy_results




