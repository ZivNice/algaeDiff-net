# # param_manager.py
# from mmengine.registry import HOOKS
# from mmengine.hooks import Hook
# from matplotlib import plt
# import os
# import pandas as pd
# import yaml
# import matplotlib.pyplot as plt
# import seaborn as sns
#
#
# @HOOKS.register_module()
# class ParamSensitivityHook(Hook):
#     """增强版参数敏感性分析Hook，支持密度分级和目标尺寸的细分指标"""
#
#     def __init__(self,
#                  param_ranges,
#                  save_dir,
#                  update_interval=1,
#                  warmup_epochs=5):
#         super().__init__()
#         self.param_ranges = param_ranges
#         self.save_dir = save_dir
#         self.update_interval = update_interval
#         self.warmup_epochs = warmup_epochs
#         self.results = []
#         self.current_param_set = None
#
#         # 确保保存目录存在
#         os.makedirs(save_dir, exist_ok=True)
#
#         # 初始化要监控的指标
#         self.metric_keys = [
#             # 整体性能指标
#             'bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75',
#             # 目标尺寸相关指标
#             'bbox_mAP_s', 'bbox_mAP_m', 'bbox_mAP_l',
#             # 密度分级指标
#             'bbox_mAP_sparse', 'bbox_mAP_50_sparse', 'bbox_mAP_75_sparse',
#             'bbox_mAP_normal', 'bbox_mAP_50_normal', 'bbox_mAP_75_normal',
#             'bbox_mAP_dense', 'bbox_mAP_50_dense', 'bbox_mAP_75_dense'
#         ]
#
#     def record_results(self, runner):
#         """记录当前参数组合的评估结果"""
#         if self.current_param_set is None:
#             return
#
#         # 获取评估结果
#         eval_results = runner.evaluation_results
#
#         # 构建指标字典
#         metrics = {}
#         for key in self.metric_keys:
#             # 处理COCO指标前缀
#             coco_key = f'coco/{key}' if not key.startswith('coco/') else key
#             metrics[key] = eval_results.get(coco_key, 0.0)
#
#         # 记录结果
#         result = {
#             'epoch': runner.epoch,
#             'params': self.current_param_set.copy(),
#             'metrics': metrics
#         }
#
#         self.results.append(result)
#         self.save_results()
#
#     def analyze_results(self):
#         """分析实验结果，针对不同维度的指标"""
#         analysis = {
#             'overall_sensitivity': {},
#             'size_sensitivity': {},
#             'density_sensitivity': {},
#             'best_combinations': {
#                 'overall': {},
#                 'small_objects': {},
#                 'dense_scenes': {}
#             }
#         }
#
#         # 将结果转换为DataFrame便于分析
#         df = pd.DataFrame([
#             {
#                 **r['params'],
#                 **{f'metrics_{k}': v for k, v in r['metrics'].items()}
#             }
#             for r in self.results
#         ])
#
#         # 1. 分析整体mAP的参数敏感度
#         for param in self.param_ranges.keys():
#             corr = df[param].corr(df['metrics_bbox_mAP'])
#             analysis['overall_sensitivity'][param] = float(corr)
#
#         # 2. 分析对小目标的影响
#         for param in self.param_ranges.keys():
#             corr = df[param].corr(df['metrics_bbox_mAP_s'])
#             analysis['size_sensitivity'][f'{param}_small'] = float(corr)
#
#         # 3. 分析对密集场景的影响
#         for param in self.param_ranges.keys():
#             corr = df[param].corr(df['metrics_bbox_mAP_dense'])
#             analysis['density_sensitivity'][f'{param}_dense'] = float(corr)
#
#         # 4. 找出最佳参数组合
#         # 整体最佳
#         best_idx = df['metrics_bbox_mAP'].idxmax()
#         analysis['best_combinations']['overall'] = df.loc[best_idx, self.param_ranges.keys()].to_dict()
#
#         # 小目标最佳
#         best_small_idx = df['metrics_bbox_mAP_s'].idxmax()
#         analysis['best_combinations']['small_objects'] = df.loc[best_small_idx, self.param_ranges.keys()].to_dict()
#
#         # 密集场景最佳
#         best_dense_idx = df['metrics_bbox_mAP_dense'].idxmax()
#         analysis['best_combinations']['dense_scenes'] = df.loc[best_dense_idx, self.param_ranges.keys()].to_dict()
#
#         return analysis
#
#     def save_final_results(self):
#         """保存最终结果和可视化"""
#         # 保存详细结果
#         self.save_results()
#
#         # 生成结果分析
#         analysis = self.analyze_results()
#
#         # 保存分析结果
#         analysis_path = os.path.join(self.save_dir, 'sensitivity_analysis.yaml')
#         with open(analysis_path, 'w') as f:
#             yaml.dump(analysis, f)
#
#         # 生成可视化结果
#         self._generate_visualizations()
#
#     def _generate_visualizations(self):
#         """生成可视化结果"""
#         df = pd.DataFrame(self.results)
#
#         # 1. 参数敏感度热力图
#         plt.figure(figsize=(15, 10))
#         sensitivity_data = []
#         for param in self.param_ranges.keys():
#             for metric in ['bbox_mAP', 'bbox_mAP_s', 'bbox_mAP_dense']:
#                 correlation = df[f'params.{param}'].corr(df[f'metrics.{metric}'])
#                 sensitivity_data.append({
#                     'Parameter': param,
#                     'Metric': metric,
#                     'Correlation': correlation
#                 })
#
#         sensitivity_df = pd.DataFrame(sensitivity_data)
#         sensitivity_matrix = sensitivity_df.pivot(
#             index='Parameter',
#             columns='Metric',
#             values='Correlation'
#         )
#
#         sns.heatmap(sensitivity_matrix, annot=True, cmap='RdYlBu_r', center=0)
#         plt.title('Parameter Sensitivity Analysis')
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.save_dir, 'sensitivity_heatmap.pdf'))
#         plt.close()
#
#         # 2. 性能趋势图
#         fig, axes = plt.subplots(3, 1, figsize=(15, 15))
#
#         # 整体mAP趋势
#         axes[0].plot(df['epoch'], df['metrics.bbox_mAP'], marker='o')
#         axes[0].set_title('Overall mAP Trend')
#         axes[0].set_xlabel('Epoch')
#         axes[0].set_ylabel('mAP')
#         axes[0].grid(True)
#
#         # 不同尺寸目标的mAP趋势
#         for size in ['s', 'm', 'l']:
#             axes[1].plot(df['epoch'], df[f'metrics.bbox_mAP_{size}'],
#                          marker='o', label=f'mAP_{size}')
#         axes[1].set_title('Size-specific mAP Trend')
#         axes[1].set_xlabel('Epoch')
#         axes[1].set_ylabel('mAP')
#         axes[1].legend()
#         axes[1].grid(True)
#
#         # 不同密度场景的mAP趋势
#         for density in ['sparse', 'normal', 'dense']:
#             axes[2].plot(df['epoch'], df[f'metrics.bbox_mAP_{density}'],
#                          marker='o', label=f'mAP_{density}')
#         axes[2].set_title('Density-specific mAP Trend')
#         axes[2].set_xlabel('Epoch')
#         axes[2].set_ylabel('mAP')
#         axes[2].legend()
#         axes[2].grid(True)
#
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.save_dir, 'performance_trends.pdf'))
#         plt.close()
#
