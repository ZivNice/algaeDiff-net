import os
import numpy as np
import cv2
from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class LoadDensityAnnotation(BaseTransform):
    """从边界框生成密度图的转换类

    Args:
        sigma (float): 高斯核的标准差，默认4.0
        method (str): 密度图生成方法，目前支持'gaussian'
        min_overlap (float): 最小重叠阈值，默认0.7
        cache_dir (str, optional): 密度图缓存目录，默认None
    """

    def __init__(self,
                 sigma=4.0,
                 method='gaussian',
                 min_overlap=0.7,
                 cache_dir=None):
        super().__init__()
        self.sigma = sigma
        self.method = method
        self.min_overlap = min_overlap
        self.cache_dir = cache_dir

    def _generate_gaussian_map(self, height, width, center_x, center_y, bbox_size=None):
        """生成单个目标的高斯密度图

        Args:
            height (int): 图像高度
            width (int): 图像宽度
            center_x (float): 目标中心点x坐标
            center_y (float): 目标中心点y坐标
            bbox_size (tuple, optional): 边界框大小(w, h)，用于自适应sigma

        Returns:
            np.ndarray: 生成的高斯密度图
        """
        # 如果提供了边界框大小，根据大小调整sigma
        if bbox_size is not None:
            w, h = bbox_size
            sigma = min(w, h) * 0.1  # 根据边界框大小自适应调整sigma
        else:
            sigma = self.sigma

        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]

        # 生成高斯分布
        gaussian = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))

        return gaussian

    def transform(self, results):
        """执行转换

        Args:
            results (dict): 包含图像和标注信息的结果字典

        Returns:
            dict: 更新后的结果字典，包含密度图
        """
        # 1. 尝试从缓存加载
        if self.cache_dir is not None:
            # 优先使用img_id，如果没有则使用文件名作为标识
            img_id = results.get('img_id', os.path.splitext(os.path.basename(results['img_path']))[0])
            cache_file = os.path.join(self.cache_dir, f"{img_id}_density.npy")

            if os.path.exists(cache_file):
                results['gt_density'] = np.load(cache_file)
                return results

        # 2. 获取图像尺寸
        img_shape = results['img_shape']  # (H, W, C)
        height, width = img_shape[:2]

        # 3. 初始化密度图
        density_map = np.zeros((height, width), dtype=np.float32)

        # 4. 获取边界框并生成密度图
        if 'gt_bboxes' in results and len(results['gt_bboxes']) > 0:
            bboxes = results['gt_bboxes']

            # 确保边界框是绝对坐标
            if bboxes.dtype == np.float32 and np.all(bboxes <= 1.0):
                bboxes = bboxes.copy()
                bboxes[:, [0, 2]] *= width
                bboxes[:, [1, 3]] *= height

            # 对每个边界框生成密度图
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox[:4]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                bbox_w, bbox_h = x2 - x1, y2 - y1

                # 生成该目标的高斯密度图
                gaussian = self._generate_gaussian_map(
                    height, width, center_x, center_y,
                    bbox_size=(bbox_w, bbox_h)
                )

                # 累加到总密度图
                density_map += gaussian

        # 5. 归一化密度图
        if density_map.max() > 0:
            density_map = density_map / density_map.max()

        # 6. 保存到缓存
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(cache_file, density_map)

        # 7. 添加到结果字典
        results['gt_density'] = density_map

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(sigma={self.sigma}, '
        repr_str += f'method={self.method}, '
        repr_str


import numpy as np
from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import PackDetInputs


@TRANSFORMS.register_module()
class PackDetInputsWithDensity(PackDetInputs):
    """打包检测输入并包含密度图。"""

    def transform(self, results):
        """将结果字典转换为DetDataSample。

        Args:
            results (dict): 包含图像和标注信息的结果字典。

        Returns:
            dict: 包含打包后数据的字典。
        """
        # 调用父类方法进行基本打包
        packed_results = super().transform(results)
        data_sample = packed_results['data_samples']

        # 添加密度图到data_sample
        if 'gt_density' in results:
            data_sample.gt_density = results['gt_density']

        return packed_results
