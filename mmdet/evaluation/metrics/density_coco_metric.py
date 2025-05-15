from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from collections import defaultdict
import io
import sys
import contextlib


@METRICS.register_module()
class DensityCocoMetric(CocoMetric):
    def __init__(self,
                 ann_file=None,
                 metric='bbox',
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 silent_density_eval=True,  # 新增参数
                 **kwargs):
        super().__init__(
            ann_file=ann_file,
            metric=metric,
            classwise=classwise,
            proposal_nums=proposal_nums,
            iou_thrs=iou_thrs,
            **kwargs)

        # 密度阈值设置
        self.density_thresholds = {
            'sparse': 3,
            'normal': 15,
            'dense': float('inf')
        }

        # 分级结果存储
        self.density_results = defaultdict(list)

        # 是否静默输出密度级别评估结果
        self.silent_density_eval = silent_density_eval

    @contextlib.contextmanager
    def _redirect_stdout(self):
        """临时重定向标准输出的上下文管理器"""
        if self.silent_density_eval:
            # 创建临时输出缓冲区
            temp_stdout = io.StringIO()
            # 保存原始stdout
            old_stdout = sys.stdout
            # 重定向到临时缓冲区
            sys.stdout = temp_stdout
            try:
                yield temp_stdout
            finally:
                # 恢复原始stdout
                sys.stdout = old_stdout
        else:
            # 不重定向，直接执行
            yield sys.stdout

    def process(self, data_batch: dict, data_samples: list) -> None:
        """处理一批数据样本"""
        # 调用父类处理方法
        super().process(data_batch, data_samples)

        # 按密度分级处理
        for data_sample in data_samples:
            gt_instances = data_sample.get('gt_instances', {})
            num_instances = len(gt_instances.get('bboxes', []))

            # 确定密度级别
            if num_instances <= self.density_thresholds['sparse']:
                density_level = 'sparse'
            elif num_instances <= self.density_thresholds['normal']:
                density_level = 'normal'
            else:
                density_level = 'dense'

            # 存储分级结果
            self.density_results[density_level].append(data_sample)

    def compute_metrics(self, results: list) -> dict:
        """计算评估指标"""
        # 获取基础评估指标
        base_metrics = super().compute_metrics(results)

        # 检查COCO API是否可用
        if not hasattr(self, '_coco_api') or self._coco_api is None:
            return base_metrics

        # 按密度计算分级指标
        density_metrics = {}
        density_map_values = {}  # 存储密度级别的 mAP 值

        for density_level, samples in self.density_results.items():
            if not samples:
                continue

            # 筛选图像ID和准备评估数据
            img_ids = list({s['img_id'] for s in samples if 'img_id' in s})
            if not img_ids:
                continue

            # 准备该密度级别的预测结果
            predictions = []
            for sample in samples:
                if 'pred_instances' not in sample:
                    continue

                pred_instances = sample['pred_instances']
                if not all(key in pred_instances for key in ['bboxes', 'scores', 'labels']):
                    continue

                # 转换预测结果格式
                for bbox, score, label in zip(
                        pred_instances['bboxes'],
                        pred_instances['scores'],
                        pred_instances['labels']
                ):
                    predictions.append({
                        'image_id': sample['img_id'],
                        'bbox': self.xyxy2xywh(bbox),
                        'score': float(score),
                        'category_id': self.cat_ids[label]
                    })

            if not predictions:
                continue

            try:
                # 准备评估器
                coco_dt = self._coco_api.loadRes(predictions)
                coco_eval = COCOeval(self._coco_api, coco_dt, 'bbox')

                # 设置评估参数（与父类保持一致）
                coco_eval.params.imgIds = img_ids
                if hasattr(self, 'proposal_nums'):
                    coco_eval.params.maxDets = list(self.proposal_nums)
                if hasattr(self, 'iou_thrs') and self.iou_thrs is not None:
                    coco_eval.params.iouThrs = self.iou_thrs

                # 使用重定向捕获输出
                with self._redirect_stdout():
                    # 执行评估
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()

                # 记录该密度级别的指标
                density_metrics.update({
                    f'bbox_mAP_{density_level}': float(coco_eval.stats[0]),
                    f'bbox_mAP_50_{density_level}': float(coco_eval.stats[1]),
                    f'bbox_mAP_75_{density_level}': float(coco_eval.stats[2])
                })

                # 存储密度级别的 mAP 值（用于扩展 bbox_mAP_copypaste）
                density_map_values[density_level] = float(coco_eval.stats[0])

            except Exception as e:
                print(f"[DensityEval] Error evaluating {density_level}: {str(e)}")

        # 修改 bbox_mAP_copypaste
        if 'bbox_mAP_copypaste' in base_metrics and density_map_values:
            original_copypaste = base_metrics['bbox_mAP_copypaste']

            # 扩展 copypaste 字符串，添加每个密度级别的 mAP
            density_values = []
            for level in ['sparse', 'normal', 'dense']:
                if level in density_map_values:
                    density_values.append(f"{density_map_values[level]:.3f}")
                else:
                    density_values.append("N/A")

            # # 将密度 mAP 值添加到原始 copypaste 字符串中
            # extended_copypaste = f"{original_copypaste} {' '.join(density_values)}"

            # 更新 metrics 中的 copypaste 字符串
            # 将密度 mAP 值添加到原始 copypaste 字符串中
            extended_copypaste = f"{original_copypaste} s:{density_values[0]} n:{density_values[1]} d:{density_values[2]}"
            # 更新 metrics 中的 copypaste 字符串
            base_metrics['bbox_mAP_copypaste'] = extended_copypaste

        if density_map_values:
            density_values = []
            for level in ['sparse', 'normal', 'dense']:
                if level in density_map_values:
                    density_values.append(f"{density_map_values[level]:.3f}")
                else:
                    density_values.append("N/A")

            # 创建一个新的包含密度级别的copypaste字符串
            base_metrics['bbox_mAP_density_copypaste'] = f"\n{base_metrics.get('bbox_mAP', 'N/A'):.3f} s:{density_values[0]} n:{density_values[1]} d:{density_values[2]}"

        # 合并指标
        all_metrics = {**base_metrics, **density_metrics}

        # 清理临时数据
        self.density_results.clear()

        return all_metrics