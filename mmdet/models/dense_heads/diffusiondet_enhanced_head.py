# from mmdet.registry import MODELS
# from mmdet.utils.lcm_modules import LCM
# import torch.nn as nn
# from projects.DiffusionDet.diffusiondet import DynamicDiffusionDetHead, SingleDiffusionDetHead
# from mmdet.models.losses import build_loss
#
#
# @MODELS.register_module()
# class EnhancedDiffusionDetHead(DynamicDiffusionDetHead):
#     """集成LAM-LCM的增强检测头"""
#
#     def __init__(self,
#                  num_classes=80,
#                  feat_channels=256,
#                  lcm_cfg=dict(num_blocks=3),
#                  lam_fpn=True,
#                  criterion=None,  # 添加criterion参数
#                  **kwargs):
#
#         # 更新 single_head 配置
#         if 'single_head' in kwargs:
#             kwargs['single_head'].update({
#                 'type': 'EnhancedSingleDiffusionDetHead',
#                 'lcm_cfg': lcm_cfg
#             })
#         # 确保正确调用父类的初始化
#         super().__init__(
#             num_classes=num_classes,
#             feat_channels=feat_channels,
#             **kwargs)
#
#         # 构建criterion
#         self.criterion = MODELS.build(criterion) if criterion is not None else None
#
#         # 特征精炼模块
#         self.lcm = self._build_lcm(lcm_cfg)
#
#         # 特征增强配置
#         self.lam_fpn = lam_fpn
#         if lam_fpn:
#             self.init_lam_fpn(feat_channels)
#
#     def _build_lcm(self, cfg):
#         """构建局部协作模块"""
#         return nn.ModuleList([
#             LCM(
#                 in_channels=self.feat_channels,
#                 num_blocks=cfg.get('num_blocks', 3)
#             ) for _ in range(cfg.get('num_heads', self.num_heads))
#         ])
#
#     def init_lam_fpn(self, in_channels):
#         """初始化LAM-FPN"""
#         from mmdet.models.necks.lam_fpn import LAM
#         self.lam = nn.ModuleList([
#             LAM(in_channels=in_channels)
#             for _ in range(self.num_heads)
#         ])
#
#     def loss(self, x, proposal_boxes, gt_instances, image_sizes, **kwargs):
#         """重写loss方法以支持criterion"""
#         # 获取预测结果
#         class_logits, pred_bboxes = self(x, proposal_boxes, **kwargs)
#
#         if self.criterion is not None:
#             # 使用criterion计算损失
#             return self.criterion(
#                 class_logits,
#                 pred_bboxes,
#                 proposal_boxes,
#                 gt_instances,
#                 image_sizes
#             )
#         # 如果没有criterion，使用父类的损失计算方式
#         return super().loss(x, proposal_boxes, gt_instances, image_sizes, **kwargs)
#
#     def forward(self, features, bboxes, init_t, init_features=None):
#         """改进的前向传播"""
#         # 1. 获取时间嵌入
#         time = self.time_mlp(init_t)
#
#         # 2. 应用LAM增强（如果启用）
#         if self.lam_fpn:
#             enhanced_features = []
#             for feat in features:
#                 # 对每个特征层应用LAM
#                 for lam_module in self.lam:
#                     feat = lam_module(feat)
#                 enhanced_features.append(feat)
#             features = enhanced_features
#
#         # 3. 调用父类的前向传播获取初始预测
#         class_logits, pred_bboxes = super().forward(
#             features, bboxes, init_t, init_features)
#
#         # 4. 应用LCM进行特征精炼
#         refined_class_logits = []
#         refined_pred_bboxes = []
#
#         for head_idx, (logits, boxes) in enumerate(zip(class_logits, pred_bboxes)):
#             feat = self.lcm[head_idx](boxes)  # 使用对应的LCM模块
#
#             # 更新预测结果
#             refined_class_logits.append(logits)
#             refined_pred_bboxes.append(feat)
#
#         # 5. 返回精炼后的预测结果
#         if self.deep_supervision:
#             return torch.stack(refined_class_logits), torch.stack(refined_pred_bboxes)
#         else:
#             return refined_class_logits[-1][None, ...], refined_pred_bboxes[-1][None, ...]
#
#
# @MODELS.register_module()
# class EnhancedSingleDiffusionDetHead(SingleDiffusionDetHead):
#     def __init__(self,
#                  num_classes=80,
#                  feat_channels=256,
#                  lcm_cfg=dict(num_blocks=3),
#                  **kwargs):
#         super().__init__(num_classes=num_classes,
#                          feat_channels=feat_channels,
#                          **kwargs)
#
#         # 添加 LCM 模块
#         self.lcm = LCM(
#             in_channels=feat_channels,
#             num_blocks=lcm_cfg.get('num_blocks', 3)
#         )
#
#     def forward(self, features, bboxes, pro_features, pooler, time_emb):
#         # 1. 获取原始特征
#         class_logits, pred_bboxes, obj_features = super().forward(
#             features, bboxes, pro_features, pooler, time_emb)
#
#         # 2. 使用 LCM 增强特征
#         enhanced_features = self.lcm(obj_features)
#
#         # 3. 基于增强特征更新预测
#         cls_feature = enhanced_features.transpose(0, 1).reshape(-1, self.feat_channels)
#         reg_feature = cls_feature.clone()
#
#         for cls_layer in self.cls_module:
#             cls_feature = cls_layer(cls_feature)
#         for reg_layer in self.reg_module:
#             reg_feature = reg_layer(reg_feature)
#
#         refined_class_logits = self.class_logits(cls_feature)
#         bboxes_deltas = self.bboxes_delta(reg_feature)
#         refined_pred_bboxes = self.apply_deltas(
#             bboxes_deltas, bboxes.view(-1, 4))
#
#         N, num_boxes = bboxes.shape[:2]
#         return (refined_class_logits.view(N, num_boxes, -1),
#                 refined_pred_bboxes.view(N, num_boxes, -1),
#                 enhanced_features)
