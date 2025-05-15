import random
import cv2
import numpy as np
import torch
from mmcv import imdenormalize
from mmdet.datasets.transforms import RandomFlip, Resize
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class IAOAugmentation(object):
    def __init__(self,
                 strategies: list,
                 policy_weights: list,
                 prob: float = 0.5):
        self.strategies = strategies
        self.policy_weights = policy_weights
        self.prob = prob
        self._init_augmenters()

    def _init_augmenters(self):
        """初始化各增强策略的实现"""
        self.aug_map = {
            'mixup': self._mixup,
            'gridmask': self._gridmask,
            'mosaic': self._mosaic,
            'color_jitter': self._color_jitter,
            'random_rotate': self._random_rotate,
            'cutout': self._cutout,
            'flip': RandomFlip(prob=1.0),
            'box_jitter': self._box_jitter
        }

    def _mixup(self, results: dict) -> dict:
        """MixUp增强实现"""
        if random.random() > self.prob:
            return results

        # 随机选择混合样本
        idx = random.randint(0, len(results['img']) - 1)
        img1 = results['img']
        img2 = results['img'][idx]

        # 混合系数
        alpha = 0.8
        mixed_img = (alpha * img1 + (1 - alpha) * img2).astype(np.float32)

        # 混合标注
        mixed_bboxes = np.concatenate(
            [results['gt_bboxes'], results['gt_bboxes'][idx]], axis=0)
        mixed_labels = np.concatenate(
            [results['gt_labels'], results['gt_labels'][idx]], axis=0)

        results.update(img=mixed_img,
                       gt_bboxes=mixed_bboxes,
                       gt_labels=mixed_labels)
        return results

    def _gridmask(self, results: dict) -> dict:
        """网格遮挡增强"""
        img = results['img']
        h, w = img.shape[:2]

        # 生成网格参数
        d = np.random.randint(96, 128)
        r = np.random.randint(d // 2, d)
        l = np.random.randint(0, d // 2)

        # 创建网格遮罩
        mask = np.ones((h, w), np.float32)
        for i in range(h // d + 1):
            for j in range(w // d + 1):
                if (i + j) % 2 == 0:
                    x1 = j * d + l
                    y1 = i * d + l
                    x2 = min(x1 + r, w)
                    y2 = min(y1 + r, h)
                    mask[y1:y2, x1:x2] = 0.0

        # 应用遮罩
        results['img'] = (img * mask[..., np.newaxis]).astype(np.float32)
        return results

    def _mosaic(self, results: dict) -> dict:
        """马赛克增强实现

        将四张图像拼接成一张，实现类似YOLO系列的马赛克数据增强

        Args:
            results (dict): 包含图像和标注信息的字典

        Returns:
            dict: 增强后的结果字典
        """
        if not isinstance(results['img'], list) and len(results) < 4:
            # 如果没有足够的图像进行马赛克增强，则返回原始结果
            return results

        # 获取当前图像和标注
        img = results['img']
        h, w = img.shape[:2]

        # 随机选择三张额外图像
        indices = list(range(len(results['img'])))
        indices.remove(0)  # 移除当前图像索引
        if len(indices) < 3:
            # 如果没有足够的图像，则复制现有图像
            indices = indices * (3 // len(indices) + 1)

        # 随机选择三张不同的图像
        mosaic_indices = random.sample(indices, 3)
        mosaic_imgs = [results['img'][i] for i in mosaic_indices]
        mosaic_imgs.insert(0, img)  # 添加当前图像

        # 计算马赛克图像尺寸
        mosaic_h = h * 2
        mosaic_w = w * 2

        # 创建马赛克画布
        mosaic_img = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.float32)

        # 定义四个区域的位置
        # 左上、右上、左下、右下
        regions = [(0, 0), (w, 0), (0, h), (w, h)]

        # 合并标注框和标签
        merged_bboxes = []
        merged_labels = []

        # 填充四个区域
        for i, (region_x, region_y) in enumerate(regions):
            # 获取当前区域的图像
            curr_img = mosaic_imgs[i]
            curr_h, curr_w = curr_img.shape[:2]

            # 计算缩放比例
            scale_ratio = min(h / curr_h, w / curr_w)
            resized_h = int(curr_h * scale_ratio)
            resized_w = int(curr_w * scale_ratio)

            # 缩放图像
            resized_img = cv2.resize(curr_img, (resized_w, resized_h))

            # 计算填充量
            pad_h = h - resized_h
            pad_w = w - resized_w
            top_pad = pad_h // 2
            left_pad = pad_w // 2

            # 将图像放入马赛克画布
            mosaic_img[region_y:region_y + resized_h + top_pad,
            region_x:region_x + resized_w + left_pad] = \
                resized_img[0:resized_h, 0:resized_w]

            # 处理标注框
            if i == 0:
                # 当前图像的标注框
                curr_bboxes = results['gt_bboxes'].copy()
                curr_labels = results['gt_labels'].copy()
            else:
                # 其他图像的标注框
                curr_bboxes = results['gt_bboxes'][mosaic_indices[i - 1]].copy()
                curr_labels = results['gt_labels'][mosaic_indices[i - 1]].copy()

            # 如果没有标注框，则跳过
            if len(curr_bboxes) == 0:
                continue

            # 缩放标注框
            curr_bboxes[:, [0, 2]] = curr_bboxes[:, [0, 2]] * scale_ratio
            curr_bboxes[:, [1, 3]] = curr_bboxes[:, [1, 3]] * scale_ratio

            # 调整标注框位置
            curr_bboxes[:, [0, 2]] = curr_bboxes[:, [0, 2]] + region_x + left_pad
            curr_bboxes[:, [1, 3]] = curr_bboxes[:, [1, 3]] + region_y + top_pad

            # 裁剪超出边界的标注框
            curr_bboxes[:, 0] = np.maximum(0, curr_bboxes[:, 0])
            curr_bboxes[:, 1] = np.maximum(0, curr_bboxes[:, 1])
            curr_bboxes[:, 2] = np.minimum(mosaic_w, curr_bboxes[:, 2])
            curr_bboxes[:, 3] = np.minimum(mosaic_h, curr_bboxes[:, 3])

            # 过滤掉无效的标注框
            valid_inds = (curr_bboxes[:, 2] > curr_bboxes[:, 0]) & \
                         (curr_bboxes[:, 3] > curr_bboxes[:, 1])
            curr_bboxes = curr_bboxes[valid_inds]
            curr_labels = curr_labels[valid_inds]

            # 合并标注
            merged_bboxes.append(curr_bboxes)
            merged_labels.append(curr_labels)

        # 合并所有标注
        if len(merged_bboxes) > 0:
            merged_bboxes = np.concatenate(merged_bboxes, axis=0)
            merged_labels = np.concatenate(merged_labels, axis=0)
        else:
            merged_bboxes = np.zeros((0, 4), dtype=np.float32)
            merged_labels = np.zeros((0,), dtype=np.int64)

        # 更新结果
        results['img'] = mosaic_img
        results['gt_bboxes'] = merged_bboxes
        results['gt_labels'] = merged_labels

        # 更新图像尺寸信息
        results['img_shape'] = (mosaic_h, mosaic_w, 3)

        return results

    def __call__(self, results: dict) -> dict:
        """执行增强策略选择"""
        if np.random.rand() > self.prob:
            return results

        strategy = random.choices(
            self.strategies,
            weights=self.policy_weights,
            k=1)[0]

        return self.aug_map[strategy](results)
