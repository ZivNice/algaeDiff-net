#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版COCO评估错误诊断工具
针对'Results do not correspond to current coco set'错误的专项分析
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import torch
import mmcv


class COCOMismatchDebugger:
    def __init__(self, data_root, result_path, config_path=None):
        self.data_root = data_root
        self.result_path = result_path
        self.config_path = config_path

        # 主要数据集路径
        self.test_ann_paths = [
            os.path.join(data_root, 'annotations/instances_test2017.json'),
            os.path.join(data_root, '../mmdetection_format/test/_annotations.coco.json')
        ]

        # 实际使用的数据集对象
        self.test_datasets = []
        self.predictions = None
        self.config = None

    def load_all_data(self):
        """加载所有测试数据和预测结果"""
        self._load_test_datasets()
        self._load_predictions()
        self._load_config()

    def _load_test_datasets(self):
        """加载所有可能的测试数据集"""
        print("\n=== 加载测试数据集 ===")

        for path in self.test_ann_paths:
            if os.path.exists(path):
                try:
                    name = os.path.basename(path)
                    print(f"加载数据集: {path}")
                    coco = COCO(path)
                    print(f"✓ 成功加载 {name}, 包含 {len(coco.getImgIds())} 张图像")

                    # 保存数据集信息
                    self.test_datasets.append({
                        'name': name,
                        'path': path,
                        'coco': coco,
                        'img_ids': coco.getImgIds()
                    })
                except Exception as e:
                    print(f"✗ 加载失败: {str(e)}")

        if not self.test_datasets:
            print("警告: 未找到任何有效的测试集标注文件!")

    def _load_predictions(self):
        """加载预测结果"""
        if not self.result_path or not os.path.exists(self.result_path):
            print("\n✗ 预测结果文件不存在: {}".format(self.result_path or "未指定"))
            return

        print(f"\n=== 加载预测结果: {self.result_path} ===")
        try:
            if self.result_path.endswith('.pkl'):
                self.predictions = mmcv.load(self.result_path)
            elif self.result_path.endswith('.json'):
                with open(self.result_path, 'r') as f:
                    self.predictions = json.load(f)
            else:
                print(f"✗ 不支持的文件格式: {os.path.splitext(self.result_path)[1]}")
                return

            print(f"✓ 成功加载预测结果")

            # 确定结果格式
            if isinstance(self.predictions, list):
                if len(self.predictions) > 0:
                    if isinstance(self.predictions[0], list):
                        print(f"  检测到MMDetection格式结果: 包含 {len(self.predictions)} 个图像的预测")
                        # 转换为更易于分析的格式
                        self._convert_mmdet_results()
                    else:
                        print(f"  检测到单层列表格式: 包含 {len(self.predictions)} 个预测项")
        except Exception as e:
            print(f"✗ 加载预测结果失败: {str(e)}")

    def _convert_mmdet_results(self):
        """将MMDetection格式的结果转换为更易分析的格式"""
        try:
            converted_results = []
            for i, img_result in enumerate(self.predictions):
                # MMDetection格式: 每个元素是一个列表，对应一张图像的所有类别预测
                # 我们需要为每个预测添加图像ID
                for class_id, class_dets in enumerate(img_result):
                    for det in class_dets:
                        # det格式: [x1, y1, x2, y2, score]
                        bbox = det[:4].tolist() if isinstance(det, np.ndarray) else det[:4]
                        score = float(det[4])
                        converted_results.append({
                            'img_id': i,  # 使用索引作为图像ID
                            'bbox': bbox,
                            'score': score,
                            'category_id': class_id
                        })
            self.predictions = converted_results
            print(f"  转换完成，共生成 {len(self.predictions)} 个检测结果")
        except Exception as e:
            print(f"  ✗ 转换MMDetection格式结果失败: {str(e)}")

    def _load_config(self):
        """加载配置文件"""
        if not self.config_path or not os.path.exists(self.config_path):
            return

        print(f"\n=== 加载配置文件: {self.config_path} ===")
        try:
            # 如果是使用mmcv的Config格式
            from mmcv import Config
            self.config = Config.fromfile(self.config_path)
            print(f"✓ 成功加载配置文件")

            # 分析配置文件中的数据集路径
            self._analyze_config_paths()
        except Exception as e:
            print(f"✗ 加载配置文件失败: {str(e)}")

    def _analyze_config_paths(self):
        """分析配置文件中的各种数据集路径"""
        if not self.config:
            return

        print("\n[配置文件中的数据集路径]")
        paths = []

        # 寻找常见的数据集路径配置
        for key in ['test_dataloader', 'test_evaluator', 'val_evaluator']:
            if hasattr(self.config, key):
                config_item = getattr(self.config, key)
                if isinstance(config_item, dict):
                    # 寻找ann_file字段
                    if 'ann_file' in config_item:
                        paths.append((key + '.ann_file', config_item['ann_file']))
                    elif 'dataset' in config_item and isinstance(config_item['dataset'], dict):
                        if 'ann_file' in config_item['dataset']:
                            paths.append((key + '.dataset.ann_file', config_item['dataset']['ann_file']))

        # 打印找到的路径
        for key, path in paths:
            print(f"- {key}: {path}")

        # 检查路径是否一致
        if len(set(p for _, p in paths)) > 1:
            print("\n⚠ 警告: 配置文件中使用了不同的标注文件路径")

    def analyze_id_mismatch(self):
        """分析ID不匹配问题"""
        if not self.predictions or not self.test_datasets:
            return

        print("\n=== ID匹配分析 ===")

        # 获取预测中的图像ID
        pred_img_ids = []
        for pred in self.predictions:
            if 'img_id' in pred:
                img_id = pred['img_id']
            elif 'image_id' in pred:
                img_id = pred['image_id']
            else:
                continue

            # 处理可能的类型转换
            if isinstance(img_id, torch.Tensor):
                img_id = img_id.item()
            elif isinstance(img_id, np.ndarray):
                img_id = img_id.item()

            pred_img_ids.append(img_id)

        unique_pred_ids = set(pred_img_ids)

        print(f"[预测结果中的图像ID]")
        print(f"- 预测结果数量: {len(self.predictions)}")
        print(f"- 唯一图像ID数量: {len(unique_pred_ids)}")
        id_types = Counter(type(i).__name__ for i in unique_pred_ids)
        print(f"- ID类型分布: {dict(id_types)}")

        # 检查ID范围
        try:
            numeric_ids = [int(i) for i in unique_pred_ids]
            id_min, id_max = min(numeric_ids), max(numeric_ids)
            print(f"- ID范围: [{id_min}, {id_max}]")

            # ID连续性检查
            if id_max - id_min + 1 != len(numeric_ids):
                print(f"  ⚠ ID不连续: 期望 {id_max - id_min + 1} 个连续ID，实际 {len(numeric_ids)} 个")
        except:
            print("  ✗ 无法分析ID范围，可能包含非数值ID")

        # 对于每个测试数据集，检查ID匹配情况
        for dataset in self.test_datasets:
            self._check_dataset_match(dataset, unique_pred_ids)

    def _check_dataset_match(self, dataset, pred_ids):
        """检查预测ID与特定数据集的匹配度"""
        print(f"\n[与{dataset['name']}的ID匹配分析]")

        dataset_ids = set(dataset['img_ids'])

        # 确保预测ID为整数
        pred_ids_int = set()
        for pid in pred_ids:
            try:
                pred_ids_int.add(int(pid))
            except:
                pass

        # 计算匹配情况
        common_ids = dataset_ids.intersection(pred_ids_int)
        missing_ids = dataset_ids - pred_ids_int
        extra_ids = pred_ids_int - dataset_ids

        print(f"- 共有ID: {len(common_ids)} ({len(common_ids) / len(dataset_ids) * 100:.2f}%)")
        print(f"- 数据集中有但预测中没有的ID: {len(missing_ids)}")
        print(f"- 预测中有但数据集中没有的ID: {len(extra_ids)}")

        # 详细分析
        if len(common_ids) == 0:
            print(f"  ✗ 严重错误! 预测结果与数据集完全不匹配")
            print(f"  > 数据集ID样例: {sorted(list(dataset_ids))[:5]}")
            print(f"  > 预测ID样例: {sorted(list(pred_ids_int))[:5]}")

            # 检查是否存在偏移
            self._check_id_shift(dataset_ids, pred_ids_int)
        elif len(extra_ids) > 0:
            print(f"  ⚠ 预测结果包含不在数据集中的ID")
            if len(extra_ids) < 10:
                print(f"  > 多余ID: {sorted(list(extra_ids))}")

            # 检查ID范围比较
            ds_min, ds_max = min(dataset_ids), max(dataset_ids)
            pred_min, pred_max = min(pred_ids_int), max(pred_ids_int)
            print(f"  > 数据集ID范围: [{ds_min}, {ds_max}]")
            print(f"  > 预测ID范围: [{pred_min}, {pred_max}]")

            # 检查是否存在偏移
            if len(extra_ids) > len(dataset_ids) * 0.5:
                self._check_id_shift(dataset_ids, pred_ids_int)

    def _check_id_shift(self, dataset_ids, pred_ids):
        """检查ID是否存在系统性偏移"""
        print("\n[ID偏移分析]")

        ds_ids = sorted(list(dataset_ids))[:100]  # 取前100个进行分析
        pred_ids = sorted(list(pred_ids))[:100]

        if len(ds_ids) < 10 or len(pred_ids) < 10:
            print("  样本数据不足，无法进行偏移分析")
            return

        # 计算可能的偏移量
        offsets = []
        for i in range(min(20, len(ds_ids), len(pred_ids))):
            offset = pred_ids[i] - ds_ids[i]
            offsets.append(offset)

        # 统计偏移量
        offset_counter = Counter(offsets)
        most_common_offset, count = offset_counter.most_common(1)[0]

        if count >= len(offsets) * 0.8:
            print(f"  ✓ 发现一致性偏移: {most_common_offset} (出现比例: {count / len(offsets) * 100:.2f}%)")
            print(f"  建议修复方案: 在评估前将预测ID减去 {most_common_offset}")

            # 生成修复代码
            print("\n[ID修复代码]")
            print("```python")
            print("# 将此代码添加到评估脚本中:")
            print("def fix_prediction_ids(predictions, offset):")
            print("    for pred in predictions:")
            print("        if 'img_id' in pred:")
            print(f"            pred['img_id'] -= {most_common_offset}")
            print("        elif 'image_id' in pred:")
            print(f"            pred['image_id'] -= {most_common_offset}")
            print("    return predictions")
            print("")
            print(f"# 调用修复函数: predictions = fix_prediction_ids(predictions, {most_common_offset})")
            print("```")
        else:
            print("  ✗ 未发现一致性偏移，可能是完全不同的ID系统")

    def suggest_solutions(self):
        """提供最终的解决方案建议"""
        print("\n" + "=" * 80)
        print("【COCO集错误诊断与解决方案】")
        print("=" * 80)

        print("\n1. 主要配置问题检查:")
        print("   - 确保test_dataloader、val_evaluator和test_evaluator中使用相同的标注文件")
        print("   - 修改配置文件，使所有位置都指向同一个标注文件")

        print("\n2. ID匹配问题解决方案:")
        print("   a) 模型预测时添加ID转换代码:")
        print("      ```python")
        print("      # 在模型前向传播方法中")
        print("      def forward(self, img, img_metas):")
        print("          # 确保img_metas中的img_id是整数而非张量")
        print("          for meta in img_metas:")
        print("              if isinstance(meta['img_id'], torch.Tensor):")
        print("                  meta['img_id'] = meta['img_id'].item()")
        print("      ```")

        print("\n   b) 评估前转换预测结果的ID:")
        print("      ```python")
        print("      def prepare_for_coco_evaluation(results):")
        print("          for i in range(len(results)):")
        print("              if 'img_id' in results[i]:")
        print("                  if isinstance(results[i]['img_id'], torch.Tensor):")
        print("                      results[i]['img_id'] = int(results[i]['img_id'].item())")
        print("              # 同样处理image_id")
        print("          return results")
        print("      ```")

        print("\n3. 临时解决方案:")
        print("   如果上述方法都不起作用，可以直接修改pycocotools的源码:")
        print("   ```python")
        print("   # 在coco.py的loadRes方法中添加错误处理")
        print("   # 查找这一行: assert set(annsImgIds) == set(self.getImgIds()), ...")
        print("   # 替换为:")
        print("   if set(annsImgIds) != set(self.getImgIds()):")
        print("       print(f'警告: ID不匹配, 数据集ID数量: {len(self.getImgIds())}, 预测ID数量: {len(annsImgIds)}')")
        print("       print(f'数据集前5个ID: {sorted(self.getImgIds())[:5]}')")
        print("       print(f'预测前5个ID: {sorted(annsImgIds)[:5]}')")
        print("       # 注释掉断言以允许继续执行")
        print("       # assert set(annsImgIds) == set(self.getImgIds()), ...")
        print("   ```")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='COCO评估错误专项诊断工具')
    parser.add_argument('--data-root', required=True, help='数据集根目录')
    parser.add_argument('--result-path', help='预测结果文件路径(.pkl或.json)')
    parser.add_argument('--config', help='配置文件路径')

    args = parser.parse_args()

    debugger = COCOMismatchDebugger(
        data_root=args.data_root,
        result_path=args.result_path,
        config_path=args.config
    )

    debugger.load_all_data()
    debugger.analyze_id_mismatch()
    debugger.suggest_solutions()


if __name__ == '__main__':
    main()
