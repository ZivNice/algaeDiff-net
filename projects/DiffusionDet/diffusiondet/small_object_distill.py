# projects/DiffusionDet/diffusiondet/small_object_distill.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()
class SmallObjectAwareDistillation(nn.Module):
    """小目标感知的知识蒸馏模块"""

    def __init__(self,
                 base_temp=2.0,
                 small_temp=1.0,
                 large_temp=3.0,
                 small_weight=0.7,
                 large_weight=0.3,
                 small_area_thresh=0.02,
                 use_adaptive_temp=True,
                 use_spatial_attention=True,
                 adaptive_thresh=False,
                 min_area_thresh=0.008,
                 max_area_thresh=0.03,
                 use_mixed_precision=False):
        super().__init__()
        self.base_temp = base_temp
        self.small_temp = small_temp
        self.large_temp = large_temp
        self.small_weight = small_weight
        self.large_weight = large_weight
        self.small_area_thresh = small_area_thresh
        self.use_adaptive_temp = use_adaptive_temp
        self.use_spatial_attention = use_spatial_attention

        # 新增参数
        self.adaptive_thresh = adaptive_thresh
        self.min_area_thresh = min_area_thresh
        self.max_area_thresh = max_area_thresh
        self.use_mixed_precision = use_mixed_precision

        # 注册当前阈值缓冲区
        self.register_buffer('current_thresh', torch.tensor(small_area_thresh))

        # 初始化特征对齐层和温度适配器(惰性初始化)
        self.align_conv = None
        self.temp_adapter = None

        # 训练状态追踪
        self.total_samples = 0
        self.small_obj_samples = 0

    def forward(self, student_features_list, teacher_features_list, target_bboxes, img_metas):
        """前向传播计算蒸馏损失"""
        # 多尺度特征融合蒸馏
        distill_losses = {}

        # 更新当前批次的小目标数量统计
        if self.training and self.adaptive_thresh:
            self._update_small_obj_threshold(target_bboxes, img_metas)

        # 1. 全局特征蒸馏 - 为不同层级设置不同权重
        level_weights = [0.5, 0.3, 0.15, 0.05][:len(student_features_list)]
        level_weights = torch.tensor(level_weights, device=student_features_list[0].device)

        global_loss = 0.0
        for i, (student_feat, teacher_feat) in enumerate(zip(student_features_list, teacher_features_list)):
            # 对浅层特征使用更小的温度系数
            temp = self.small_temp if i == 0 else self.base_temp if i == 1 else self.large_temp

            # 基础蒸馏损失
            level_loss = self.calc_basic_distill_loss(student_feat, teacher_feat, temp)
            global_loss += level_weights[i] * level_loss

        distill_losses['loss_distill_global'] = global_loss

        # 2. 小目标区域增强蒸馏 - 只对P2和P3层应用
        if self.use_spatial_attention and len(student_features_list) >= 2:
            small_obj_loss = 0.0
            for i, (student_feat, teacher_feat) in enumerate(zip(student_features_list[:2], teacher_features_list[:2])):
                level_loss = self.calc_spatial_aware_distill_loss(
                    student_feat, teacher_feat, target_bboxes, img_metas)
                small_obj_loss += level_weights[i] * level_loss

            # 只有当小目标损失有效(大于0)时才返回
            if small_obj_loss > 0:
                distill_losses['loss_distill_small_obj'] = small_obj_loss * 2.0  # 加大权重

        # 3. ROI级别的动态蒸馏 - 专注于小目标
        if len(student_features_list) > 0:
            roi_loss = self.dynamic_roi_distillation(
                student_features_list[0],  # 使用P2层
                teacher_features_list[0],
                target_bboxes,
                img_metas
            )
            if roi_loss > 0:
                distill_losses['loss_distill_roi'] = roi_loss

        # 计算总损失
        total_loss = sum(distill_losses.values())

        # 打印当前阈值和小目标数量 - 修复分布式检查
        # if self.training and self.adaptive_thresh and self._is_main_process():
        #     print(f"小目标蒸馏 | 当前阈值: {self.current_thresh.item():.4f} | "
        #           f"小目标比例: {self.small_obj_samples}/{self.total_samples} "
        #           f"({100.0 * self.small_obj_samples / max(1, self.total_samples):.1f}%)")
        # 每10次迭代清理一次缓存
        if hasattr(self, '_forward_count'):
            self._forward_count += 1
        else:
            self._forward_count = 0

        if self._forward_count % 10 == 0:
            # 使用空操作清理缓存，不影响计算图
            torch.cuda.empty_cache()

        return total_loss

    def _is_main_process(self):
        """检查是否为主进程，适用于分布式和非分布式环境"""
        try:
            if torch.distributed.is_initialized():
                return torch.distributed.get_rank() == 0
            else:
                return True  # 非分布式环境下视为主进程
        except:
            return True  # 出错时默认为主进程

    def _update_small_obj_threshold(self, target_bboxes, img_metas):
        """更新小目标阈值"""
        all_areas = []
        self.total_samples = 0
        self.small_obj_samples = 0

        for b, boxes in enumerate(target_bboxes):
            if len(boxes) == 0:
                continue

            # 计算边界框面积
            img_h, img_w = img_metas[b]['img_shape'][:2]
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            normalized_areas = areas / (img_h * img_w)

            all_areas.append(normalized_areas)
            self.total_samples += len(normalized_areas)

        if all_areas:
            all_areas = torch.cat(all_areas)
            if len(all_areas) > 0:
                # 使用百分位数作为阈值(目标是选取约15-20%的小目标)
                # 注意使用clamp确保阈值在合理范围内
                new_thresh = torch.quantile(all_areas, 0.15).clamp(
                    min=self.min_area_thresh, max=self.max_area_thresh)

                # 使用指数移动平均平滑阈值变化
                momentum = 0.9
                self.current_thresh.copy_(
                    momentum * self.current_thresh + (1 - momentum) * new_thresh)

                # 统计当前小目标数量
                self.small_obj_samples = (all_areas < self.current_thresh).sum().item()

    def calc_basic_distill_loss(self, student_features, teacher_features, temp=2.0):
        """基础蒸馏损失计算"""
        # 验证输入特征形状
        assert student_features.dim() == 4 and teacher_features.dim() == 4, "特征必须为4D张量"
        assert student_features.size(0) == teacher_features.size(0), "Batch size不匹配"

        # 初始化特征对齐层
        if self.align_conv is None or self.align_conv.in_channels != teacher_features.size(1):
            in_channels = teacher_features.size(1)
            out_channels = student_features.size(1)
            self.align_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1
            ).to(teacher_features.device)

        # 初始化温度适配器
        if self.use_adaptive_temp and self.temp_adapter is None:
            self.temp_adapter = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                # nn.Linear(teacher_features.size(1), 1),
                nn.Linear(teacher_features.size(1), 64),  # 添加中间层
                nn.ReLU(inplace=True),  # 添加激活函数
                nn.Linear(64, 1),  # 最终输出层
                nn.Sigmoid()
            ).to(teacher_features.device)

        # 混合精度训练模式
        context = torch.autocast(device_type='cuda', dtype=torch.float16) if self.use_mixed_precision else nullcontext()

        with context:
            # 特征对齐
            aligned_teacher = self.align_conv(teacher_features)

            # 尺寸不匹配时进行插值
            if student_features.shape[-2:] != aligned_teacher.shape[-2:]:
                aligned_teacher = F.interpolate(
                    aligned_teacher,
                    size=student_features.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            # 温度自适应调整
            if self.use_adaptive_temp and self.temp_adapter is not None:
                # 动态计算温度系数
                temp_factor = 1.0 + 0.5 * self.temp_adapter(teacher_features).view(-1, 1, 1, 1)
                dynamic_temp = temp * temp_factor
            else:
                dynamic_temp = torch.tensor(temp, device=student_features.device)

            # 计算KL散度
            student_logits = student_features / dynamic_temp
            teacher_logits = aligned_teacher / dynamic_temp

            kl_div = F.kl_div(
                F.log_softmax(student_logits, dim=1),
                F.softmax(teacher_logits.detach(), dim=1),  # 只阻断教师梯度
                reduction='batchmean'
            ) * (dynamic_temp.mean() ** 2)

        return kl_div

    def calc_spatial_aware_distill_loss(self, student_features, teacher_features,
                                        target_bboxes, img_metas):
        """计算空间感知的蒸馏损失，关注小目标区域"""
        # 特征对齐
        if student_features.shape != teacher_features.shape:
            if student_features.dim() == 4:
                s_h, s_w = student_features.shape[2:]
                t_h, t_w = teacher_features.shape[2:]
                if s_h != t_h or s_w != t_w:
                    teacher_features = F.interpolate(
                        teacher_features, size=(s_h, s_w),
                        mode='bilinear', align_corners=False)

        # 创建小目标注意力图
        attention_map = self._create_small_object_attention_map(
            student_features, target_bboxes, img_metas)

        # 检查注意力图是否有效(全零表示无小目标)
        if attention_map.sum() < 1e-6:
            return torch.tensor(0.0, device=student_features.device)

        # 增强权重 - 小目标区域权重是3倍
        attention_weight = 1.0 + 2.0 * attention_map

        # 混合精度训练模式
        context = torch.autocast(device_type='cuda', dtype=torch.float16) if self.use_mixed_precision else nullcontext()

        with context:
            # 计算加权KL散度
            temp = self.small_temp  # 使用小目标温度
            student_logits = student_features / temp
            teacher_logits = teacher_features / temp

            kl_div = F.kl_div(
                F.log_softmax(student_logits, dim=1),
                F.softmax(teacher_logits, dim=1),
                reduction='none'
            )

            # 应用空间注意力权重
            weighted_kl_div = (kl_div.mean(dim=1, keepdim=True) * attention_weight).mean() * (temp * temp)

        return weighted_kl_div

    def _create_small_object_attention_map(self, features, target_bboxes, img_metas):
        """创建小目标注意力图"""
        batch_size = features.shape[0]
        h, w = features.shape[2:]
        device = features.device

        attention_map = torch.zeros((batch_size, 1, h, w), device=device)

        # 使用当前计算的阈值，而不是固定阈值
        thresh = self.current_thresh

        for b in range(batch_size):
            if b >= len(target_bboxes) or len(target_bboxes[b]) == 0:
                continue

            # 获取图像尺寸
            img_h, img_w = img_metas[b]['img_shape'][:2]

            # 计算边界框面积
            boxes = target_bboxes[b]
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            normalized_areas = areas / (img_h * img_w)

            # 找出小目标
            small_mask = normalized_areas < thresh
            if small_mask.sum() == 0:
                continue

            small_boxes = boxes[small_mask]

            # 在注意力图上标记小目标位置
            for box in small_boxes:
                x1, y1, x2, y2 = box
                # 缩放到特征图尺寸
                x1s, y1s = int(x1 * w / img_w), int(y1 * h / img_h)
                x2s, y2s = int(x2 * w / img_w), int(y2 * h / img_h)

                # 确保有效范围
                x1s, y1s = max(0, x1s), max(0, y1s)
                x2s, y2s = min(w - 1, x2s), min(h - 1, y2s)

                if x2s > x1s and y2s > y1s:
                    # 创建高斯核而不是硬边界
                    y_grid, x_grid = torch.meshgrid(
                        torch.arange(y1s, y2s, device=device),
                        torch.arange(x1s, x2s, device=device),
                        indexing='ij'
                    )

                    # 中心点
                    cx, cy = (x1s + x2s) / 2, (y1s + y2s) / 2

                    # 高斯权重
                    sigma = max((x2s - x1s), (y2s - y1s)) / 6  # 3sigma原则
                    gaussian = torch.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))

                    attention_map[b, 0, y1s:y2s, x1s:x2s] = torch.maximum(
                        attention_map[b, 0, y1s:y2s, x1s:x2s], gaussian)

        return attention_map

    def dynamic_roi_distillation(self, student_features, teacher_features, boxes, img_metas):
        """基于ROI的动态蒸馏"""
        batch_size = student_features.shape[0]
        device = student_features.device

        small_roi_features_student = []
        small_roi_features_teacher = []

        # 使用当前计算的阈值，而不是固定阈值
        thresh = self.current_thresh

        for b in range(batch_size):
            if b >= len(boxes) or len(boxes[b]) == 0:
                continue

            # 计算边界框面积
            img_h, img_w = img_metas[b]['img_shape'][:2]
            areas = (boxes[b][:, 2] - boxes[b][:, 0]) * (boxes[b][:, 3] - boxes[b][:, 1])
            normalized_areas = areas / (img_h * img_w)

            # 识别小目标
            small_mask = normalized_areas < thresh
            if small_mask.sum() == 0:
                continue

            small_boxes = boxes[b][small_mask]

            # 提取小目标ROI特征
            h, w = student_features.shape[2:]
            for box in small_boxes:
                x1, y1, x2, y2 = box
                # 缩放到特征图尺寸
                x1s, y1s = int(x1 * w / img_w), int(y1 * h / img_h)
                x2s, y2s = int(x2 * w / img_w), int(y2 * h / img_h)

                # 确保有效区域
                x1s, y1s = max(0, x1s), max(0, y1s)
                x2s, y2s = min(w - 1, x2s), min(h - 1, y2s)

                if x2s > x1s and y2s > y1s:
                    # 提取ROI特征
                    roi_feat_student = student_features[b, :, y1s:y2s, x1s:x2s]
                    roi_feat_teacher = teacher_features[b, :, y1s:y2s, x1s:x2s]

                    # 统一大小以便批处理
                    roi_feat_student = F.adaptive_avg_pool2d(roi_feat_student.unsqueeze(0), (7, 7))
                    roi_feat_teacher = F.adaptive_avg_pool2d(roi_feat_teacher.unsqueeze(0), (7, 7))

                    small_roi_features_student.append(roi_feat_student)
                    small_roi_features_teacher.append(roi_feat_teacher)

        # 计算小目标ROI蒸馏损失
        if small_roi_features_student:
            small_roi_student = torch.cat(small_roi_features_student)
            small_roi_teacher = torch.cat(small_roi_features_teacher)

            # 使用较低温度
            return self.calc_basic_distill_loss(small_roi_student, small_roi_teacher, self.small_temp)
        else:
            # 如果没有小目标，返回零损失
            return torch.tensor(0.0, device=device)


class nullcontext:
    """一个简单的上下文管理器，不执行任何操作"""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
