
import re
import time 
import json
import datetime
from typing import Optional

import numpy as np
import torch

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate, evaluate_single_pic
import torchvision.transforms as T
from PIL import Image

class DetSolver(BaseSolver):

    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()
            
            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            classification_report = test_stats.get('classification', {}).get('classification_report', '')

            pattern = r'^\s*(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)'
            matches = re.findall(pattern, classification_report, re.MULTILINE)

            precision = {}
            recall = {}

            for match in matches:
                label = match[0]

                if re.match(r'^\d+$', label) and not any(
                        label.lower().startswith(prefix) for prefix in ['accuracy', 'avg']):
                    precision[label] = float(match[1])
                    recall[label] = float(match[2])

            precision = {k: v for k, v in precision.items() if k not in ['-1']}
            recall = {k: v for k, v in recall.items() if k not in ['-1']}
            # print("===precision", precision)
            # print("===recall", recall)
            self.criterion.matcher.update_alpha_gamma_based_on_metrics(precision, recall)

            # TODO
            for k, v in test_stats.items():
                if k == 'classification':
                    overall_accuracy = v.get('accuracy')  # 获取 Overall Accuracy
                    if 'classification' not in best_stat or overall_accuracy > best_stat['classification'].get('accuracy', 0):
                        best_stat['classification'] = v
                        best_stat['epoch'] = epoch
                else:
                    if k in best_stat:
                        if v > best_stat[k]:
                            best_stat[k] = v
                            best_stat['epoch'] = epoch
                    else:
                        best_stat[k] = v
                        best_stat['epoch'] = epoch
            if 'classification' in best_stat:
                accuracy = best_stat['classification'].get('accuracy', 0)
                print(f"best_stat: epoch: {best_stat['epoch']},accuracy: {accuracy:.4f}")
            else:
                print('No classification accuracy found.')

            def convert_ndarray_to_list(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_ndarray_to_list(v) for v in obj]
                return obj

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            log_stats = convert_ndarray_to_list(log_stats)

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                    if coco_evaluator is not None:
                        (self.output_dir / 'eval').mkdir(exist_ok=True)
                        if "classification" in test_stats:
                            filenames = ['latest_classification.pth']
                            if epoch % 50 == 0:
                                filenames.append(f'{epoch:03}_classification.pth')
                            for name in filenames:
                                torch.save(test_stats['classification'], self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def val(self, ):
        self.eval()
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        module = self.ema.module if self.ema else self.model

        print("====evaluate====")
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                                              self.val_dataloader, base_ds, self.device, self.output_dir,
                                              generate_bboxes=False, generate_confusion_matrix=True)

        # if self.output_dir:
        #     dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        #
        if self.output_dir and dist.is_main_process():
            if "classification" in test_stats:
                dist.save_on_master(test_stats["classification"],
                                    self.output_dir / "eval.pth")
        return


    def val_single_pic(self,
                       print_results: bool = False,
                       plot: bool = False,
                       output_dir: Optional[str] = None):
        self.eval()
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = self.cfg.yaml_cfg.get('output_dir', 'exps/test')


        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        module = self.ema.module if self.ema else self.model

        results = evaluate_single_pic(
            module, self.postprocessor, self.val_dataloader,
            base_ds, self.device, self.output_dir, generate_bboxes=plot, print_results=print_results)

        return results
