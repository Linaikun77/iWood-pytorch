"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

"""
import json

import math
import os
import sys
from typing import Iterable

import numpy as np
import torch
import torch.amp
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt, patches
from PIL import Image
from src.data import iWoodEvaluator, mscoco_category2name
from src.misc import (MetricLogger, SmoothedValue, reduce_dict, logger)
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)

    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # class_error = loss_dict_reduced.get('class_error', torch.tensor(0.0)).item()

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        # metric_logger.update(class_error=class_error)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device,
             output_dir, generate_bboxes=False, generate_confusion_matrix=False):
    # print("====generate_bboxes:", generate_bboxes)
    # print("====generate_confusion_matrix:", generate_confusion_matrix)
    # print("====output_result:", output_result)
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    if hasattr(postprocessors, 'iou_types'):
        iou_types = postprocessors.iou_types
    else:
        iou_types = ['bbox']

    iwood_evaluator = iWoodEvaluator(base_ds, iou_types)

    all_res = {}
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if iwood_evaluator is not None:
            iwood_evaluator.update(res)

        all_res.update(res)

    # if output_result:
    #     print("========outputting top 1 result for each image========")
    #
    #
    #     for img_id, output in all_res.items():
    #         labels = output['labels'].cpu().numpy()
    #         scores = output['scores'].cpu().numpy()
    #
    #         score_threshold = 0.9
    #         valid_indices = scores >= score_threshold
    #         labels = labels[valid_indices]
    #         scores = scores[valid_indices]
    #
    #         if len(labels) == 0:
    #             print(f"No valid detections for image {img_id} with score threshold {score_threshold}")
    #             continue
    #
    #         top_1_idx = np.argmax(scores)
    #         top_1_label = labels[top_1_idx]
    #         top_1_score = scores[top_1_idx]
    #
    #         top_1_label_name = mscoco_category2name.get(top_1_label, "Unknown")
    #
    #         print(f"Top 1 predicted label for image {img_id}: Class {top_1_label_name} ({top_1_score:.2f})")

    # if generate_confusion_matrix:
    #     print("========generating confusion matrix========")
    #     labels_unique = sorted(set(iwood_evaluator.all_preds))
    #     print("===sorted(set(iwood_evaluator.all_targets)",labels_unique)
    #     cm = confusion_matrix(iwood_evaluator.all_targets, iwood_evaluator.all_preds, labels=labels_unique)
    #
    #     label_names = [mscoco_category2name[label] for label in labels_unique]
    #
    #
    #     plt.figure(figsize=(14, 10))
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    #     plt.xticks(rotation=45, ha="right", fontsize=10)
    #     plt.yticks(fontsize=10)
    #
    #     plt.xlabel('Predicted Label', fontsize=12)
    #     plt.ylabel('True Label', fontsize=12)
    #     plt.title('Confusion Matrix', fontsize=15)
    #     cm_save_path = os.path.join(output_dir, 'confusion_matrix.png')
    #     plt.savefig(cm_save_path, bbox_inches='tight')
    #     plt.close()
    #
    #     print(f"Saved confusion matrix to: {cm_save_path}")
    #
    #
    # if generate_bboxes:
    #     print("========generating bboxes========")
    #     base_path = data_loader.dataset.img_folder
    #     print("===base_path:",base_path)
    #
    #     for img_id, output in all_res.items():
    #         labels = output['labels'].cpu().numpy()
    #         scores = output['scores'].cpu().numpy()
    #
    #
    #         img_path = base_ds.loadImgs(img_id)[0]['file_name']
    #
    #         full_img_path = os.path.join(base_path, img_path)
    #         print("Full image path:", full_img_path)
    #
    #
    #         img = Image.open(full_img_path).convert("RGB")
    #         plt.figure(figsize=(12, 12))
    #         plt.imshow(img)
    #         ax = plt.gca()
    #
    #
    #         ax.text(0.5, 1.05, f"Image ID: {img_id}", transform=ax.transAxes, fontsize=16, color='black',
    #                 weight='bold', ha='center', va='top', backgroundcolor='white')
    #
    #
    #         boxes = output['boxes'].cpu().numpy()
    #
    #         score_threshold = 0.8
    #         valid_indices = scores >= score_threshold
    #         boxes = boxes[valid_indices]
    #         labels = labels[valid_indices]
    #         scores = scores[valid_indices]
    #
    #
    #         if len(boxes) == 0:
    #             print(f"No valid detections for image {img_id} with score threshold {score_threshold}")
    #             continue
    #
    #         top_k = 5
    #         top_k_indices = np.argsort(scores)[-top_k:][::-1]
    #         colors = list(mcolors.TABLEAU_COLORS.values())
    #
    #         for idx, color in zip(top_k_indices, colors):
    #             box = boxes[idx]
    #             label = labels[idx]
    #             score = scores[idx]
    #             if score >= score_threshold:
    #                 rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
    #                                          linewidth=5, edgecolor=color, facecolor='none')
    #                 ax.add_patch(rect)
    #                 ax.text(box[0], box[1] - 10, f"Class: {label} ({score:.2f})", color=color, fontsize=12,
    #                         backgroundcolor='white')
    #
    #         save_dir = os.path.join(output_dir, 'detected_images5')
    #         os.makedirs(save_dir, exist_ok=True)
    #         save_path = os.path.join(save_dir, f"detected_{img_id}.png")
    #         plt.savefig(save_path)
    #         plt.close()
    #
    #         print(f"Saved detected image with bounding boxes to: {save_path}")

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if iwood_evaluator is not None:
        iwood_evaluator.synchronize_between_processes()

    if iwood_evaluator is not None:
        iwood_evaluator.accumulate()
        classification_stats = iwood_evaluator.summarize()


    stats = {}
    if classification_stats is not None:
        stats['classification'] = classification_stats


    return stats, iwood_evaluator


@torch.no_grad()
def evaluate_single_pic(model: torch.nn.Module, postprocessors, data_loader, base_ds, device,
                        output_dir, generate_bboxes=False, print_results=False):
    # print("===generate_bboxes:",generate_bboxes)
    # print("===print_results:",print_results)
    model.eval()
    eval_results = []

    for samples, targets in data_loader:
        samples = samples.to(device)

        outputs = model(samples)

        if isinstance(targets[0]["orig_size"], torch.Tensor):
            orig_target_size = targets[0]["orig_size"].to(device)
        else:
            orig_target_size = torch.tensor(targets[0]["orig_size"], dtype=torch.float32).to(device)

        results = postprocessors(outputs, orig_target_size)
        # print("===results:",results)

        for i, output in enumerate(results):
            labels_np = output['labels'].cpu().numpy()
            scores_np = output['scores'].cpu().numpy()

            img_id = targets[i]['image_id'].item()
            info = base_ds.loadImgs(img_id)[0]
            fname = info['file_name']

            res = {
                'file_name': fname,
                'output': output,
                'image_label': None
            }

            # 计算 top300 most common label

            top300 = labels_np[:300] if labels_np.size >= 300 else labels_np
            if top300.size > 0:
                res['image_label'] = Counter(top300).most_common(1)[0][0]

            # print(f"[{fname}]  image_label (top300 most common): {res['image_label']}")

            eval_results.append(res)

            if generate_bboxes:
                fullimg = os.path.join(data_loader.dataset.img_folder, fname)
                img = Image.open(fullimg).convert("RGB")
                plt.figure(figsize=(12, 12))
                plt.imshow(img);
                ax = plt.gca()

                boxes = output['boxes'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                scores = output['scores'].cpu().numpy()

                topk = min(3, scores.shape[0])
                colors = list(mcolors.TABLEAU_COLORS.values())
                for idx, c in zip(range(topk), colors):
                    x0, y0, x1, y1 = boxes[idx]
                    lbl = labels[idx];
                    scr = scores[idx]
                    cls_name = mscoco_category2name.get(lbl, "Unknown")
                    rect = patches.Rectangle(
                        (x0, y0), x1 - x0, y1 - y0,
                        linewidth=2, edgecolor=c, facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(x0, y0 - 10,
                            f"{cls_name} ({scr:.2f})",
                            color=c, fontsize=12,
                            backgroundcolor='white')

                save_dir = os.path.join(output_dir, 'detected_images')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(
                    save_dir, f"detected_{img_id}_{i}.png"
                )
                plt.savefig(save_path)
                plt.close()

    return eval_results