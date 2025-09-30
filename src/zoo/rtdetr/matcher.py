"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.
"""

import torch
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from src.core import register

@register
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha_dict=None, gamma_dict=None, num_classes=13):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss

        if isinstance(alpha_dict, (float, int)):
            self.alpha_dict = {i: alpha_dict for i in range(num_classes)}
        elif isinstance(alpha_dict, dict):
            self.alpha_dict = alpha_dict
        else:
            self.alpha_dict = {}

        # Handle gamma_dict
        if isinstance(gamma_dict, (float, int)):
            self.gamma_dict = {i: gamma_dict for i in range(num_classes)}
        elif isinstance(gamma_dict, dict):
            self.gamma_dict = gamma_dict
        else:
            self.gamma_dict = {}

        # print("matcher initialized")
        # print("alpha_dict", self.alpha_dict)
        # print("gamma_dict", self.gamma_dict)

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

    def update_alpha_gamma_based_on_metrics(self, precision, recall):
        """Based on precision and recall, dynamically update alpha and gamma for each class."""

        # Adjust class labels by subtracting 1 to align with alpha_dict and gamma_dict
        adjusted_precision = {
            int(cls): pr for cls, pr in precision.items()
            if cls.isdigit() and int(cls) >= 0  # Keep only classes >= 0
        }
        adjusted_recall = {
            int(cls): r for cls, r in recall.items()
            if cls.isdigit() and int(cls) >= 0
        }

        # Set thresholds for high precision and recall
        all_precisions_high = all(pr >= 0.8 for pr in adjusted_precision.values())
        all_recalls_high = all(r >= 0.75 for r in adjusted_recall.values())

        if all_precisions_high and all_recalls_high:
            # print("All classes have high precision and recall. Keeping alpha and gamma unchanged.")
            # print("alpha_dict:", self.alpha_dict)
            # print("gamma_dict:", self.gamma_dict)
            return

        for cls in adjusted_precision:
            pr = adjusted_precision.get(cls, 0.0)
            recall_value = adjusted_recall.get(cls, 0.0)

            # Get current alpha and gamma values for this class, ensure float type
            alpha_value = str(self.alpha_dict.get(cls, 0.5)).replace(',', '')
            gamma_value = str(self.gamma_dict.get(cls, 4.0)).replace(',', '')

            alpha_value = float(alpha_value)
            gamma_value = float(gamma_value)

            # Adjust alpha based on recall thresholds
            if recall_value < 0.7:
                self.alpha_dict[cls] = min(1.0, alpha_value + 0.02)
            elif recall_value > 0.98:
                self.alpha_dict[cls] = max(0.5, alpha_value - 0.01)

            # Adjust gamma based on precision thresholds
            if pr < 0.8:
                self.gamma_dict[cls] = min(10.0, gamma_value + 0.02)
            elif pr > 0.98:
                self.gamma_dict[cls] = max(4.0, gamma_value - 0.01)

        # print("Updated alpha_dict:", self.alpha_dict)
        # print("Updated gamma_dict:", self.gamma_dict)

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: A dict containing at least:
                 "pred_logits": Tensor of shape [batch_size, num_queries, num_classes] with classification logits
                 "pred_boxes": Tensor of shape [batch_size, num_queries, 4] with predicted box coordinates

            targets: A list of targets (len(targets) = batch_size), each target is a dict containing:
                 "labels": Tensor of shape [num_target_boxes] containing class labels
                 "boxes": Tensor of shape [num_target_boxes, 4] containing target box coordinates

        Returns:
            A list of size batch_size containing tuples of (index_i, index_j) where:
                - index_i: indices of selected predictions
                - index_j: indices of corresponding selected targets
            For each batch element:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten for batch cost computation
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Concatenate target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if self.use_focal_loss:
            alpha_list = []
            gamma_list = []

            for cls in tgt_ids:
                cls_value = cls.item()

                alpha_value = self.alpha_dict.get(cls_value, 0.5)
                if isinstance(alpha_value, str):
                    alpha_value = float(alpha_value.replace(',', '').strip())
                alpha_list.append(alpha_value)

                gamma_value = self.gamma_dict.get(cls_value, 4.0)
                if isinstance(gamma_value, str):
                    gamma_value = float(gamma_value.replace(',', '').strip())
                gamma_list.append(gamma_value)

            alpha_tensor = torch.tensor(alpha_list, device=tgt_ids.device)
            gamma_tensor = torch.tensor(gamma_list, device=tgt_ids.device)

            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - alpha_tensor) * (out_prob ** gamma_tensor) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha_tensor * ((1 - out_prob) ** gamma_tensor) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute L1 cost between predicted and target boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute generalized IoU cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final combined cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        # Compute assignment using Hungarian algorithm
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
