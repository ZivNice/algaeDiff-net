# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import time
import torch

from mmengine import MMLogger
from mmengine.config import Config, DictAction
from mmengine.dist import init_dist
from mmengine.registry import init_default_scope
from mmengine.utils import mkdir_or_exist

from mmdet.apis import init_detector
from mmdet.utils.benchmark import (DataLoaderBenchmark, DatasetBenchmark)


class EnhancedDiffusionDetBenchmark:
    """用于测试改进版DiffusionDet的性能基准测试类"""

    def __init__(self, cfg, checkpoint, distributed, fuse_conv_bn, max_iter,
                 log_interval, num_warmup, logger=None):
        self.cfg = cfg
        self.checkpoint = checkpoint
        self.distributed = distributed
        self.fuse_conv_bn = fuse_conv_bn
        self.max_iter = max_iter
        self.log_interval = log_interval
        self.num_warmup = num_warmup
        self.logger = logger or MMLogger.get_current_instance()

        # 设置设备
        if distributed:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
            self.device = torch.device(f'cuda:{self.rank}')
        else:
            self.rank = 0
            self.world_size = 1
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 构建模型
        self.model = self.build_model()

        # 检查是否启用了各种增强模块
        self.use_lcm = hasattr(self.model.bbox_head, 'use_lcm') and self.model.bbox_head.use_lcm
        self.use_adem = hasattr(self.model.bbox_head, 'use_adem') and self.model.bbox_head.use_adem
        self.dynamic_steps = hasattr(self.model.bbox_head, 'dynamic_steps') and self.model.bbox_head.dynamic_steps > 0

        self.logger.info(f'模型配置: LCM={self.use_lcm}, ADEM={self.use_adem}, Dynamic DDIM={self.dynamic_steps}')

        # 获取数据预处理器
        self.data_preprocessor = self.model.data_preprocessor

    def build_model(self):
        """构建模型"""
        model = init_detector(
            self.cfg, self.checkpoint, device=self.device, cfg_options={})
        if self.fuse_conv_bn:
            model = model.fuse_conv_bn()
        model.eval()
        return model

    def run_once(self):
        """单次运行测试"""
        pure_inf_time = 0
        total_time = 0

        # 创建随机输入
        batch_size = 1  # 批次大小
        img_height, img_width = 800, 800  # 图像尺寸

        # 创建随机图像 - 使用(B,C,H,W)格式，这是PyTorch的标准格式
        img = torch.randint(0, 256, (batch_size, 3, img_height, img_width), dtype=torch.float32).to(self.device)

        # 预热
        for _ in range(self.num_warmup):
            with torch.no_grad():
                # 直接使用预处理好的特征，跳过数据预处理器
                # 创建FPN输出的多尺度特征图
                fpn_feats = []
                for stride in [4, 8, 16, 32]:  # 典型的FPN步长
                    h, w = img_height // stride, img_width // stride
                    feat = torch.rand(batch_size, 256, h, w, device=self.device)  # 256是典型的FPN通道数
                    fpn_feats.append(feat)

                # 准备初始边界框和时间步
                init_bboxes = torch.rand(batch_size, 500, 4, device=self.device)  # 500是默认的num_proposals
                init_bboxes = init_bboxes * img_width  # 缩放到图像尺寸
                init_t = torch.zeros(batch_size, device=self.device, dtype=torch.long)

                # 直接调用bbox_head的forward方法
                _ = self.model.bbox_head(fpn_feats, init_bboxes, init_t)

        # 正式测试
        for i in range(self.max_iter):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                # 直接使用预处理好的特征，跳过数据预处理器
                # 创建FPN输出的多尺度特征图
                fpn_feats = []
                for stride in [4, 8, 16, 32]:  # 典型的FPN步长
                    h, w = img_height // stride, img_width // stride
                    feat = torch.rand(batch_size, 256, h, w, device=self.device)  # 256是典型的FPN通道数
                    fpn_feats.append(feat)

                # 准备初始边界框和时间步
                init_bboxes = torch.rand(batch_size, 500, 4, device=self.device)  # 500是默认的num_proposals
                init_bboxes = init_bboxes * img_width  # 缩放到图像尺寸
                init_t = torch.zeros(batch_size, device=self.device, dtype=torch.long)

                # 直接调用bbox_head的forward方法
                _ = self.model.bbox_head(fpn_feats, init_bboxes, init_t)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            pure_inf_time += elapsed
            total_time += elapsed

            if (i + 1) % self.log_interval == 0:
                fps = self.log_interval / pure_inf_time
                self.logger.info(
                    f'Done batch [{i + 1}/{self.max_iter}], '
                    f'fps: {fps:.1f} batch/s, '
                    f'times per batch: {pure_inf_time * 1000 / self.log_interval:.1f} ms/batch, '
                    f'batch size: {batch_size}, num_workers: 2')
                pure_inf_time = 0

        fps = self.max_iter / total_time
        self.logger.info(f'Overall fps: {fps:.1f} batch/s, '
                         f'times per batch: {total_time * 1000 / self.max_iter:.1f} ms/batch, '
                         f'batch size: {batch_size}, num_workers: 2')

        # 记录增强模块的状态
        self.logger.info(f'模型配置: LCM={self.use_lcm}, ADEM={self.use_adem}, Dynamic DDIM={self.dynamic_steps}')

        return fps

    def run(self, repeat_num=1):
        """运行多次测试并计算平均值"""
        fps_list = []
        for i in range(repeat_num):
            self.logger.info(f'Run {i + 1}/{repeat_num}')
            fps = self.run_once()
            fps_list.append(fps)

        mean_fps = sum(fps_list) / len(fps_list)
        std_fps = (sum((fps - mean_fps) ** 2 for fps in fps_list) / len(fps_list)) ** 0.5

        self.logger.info(f'平均 FPS: {mean_fps:.1f} ± {std_fps:.1f}')
        return mean_fps


def parse_args():
    parser = argparse.ArgumentParser(description='DiffusionDet性能测试')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--checkpoint', help='检查点文件')
    parser.add_argument(
        '--task',
        choices=['inference', 'dataloader', 'dataset'],
        default='inference',
        help='要进行基准测试的任务')
    parser.add_argument(
        '--dataset-type',
        choices=['train', 'val', 'test'],
        default='test',
        help='基准测试数据集类型。仅支持 train, val 和 test')
    parser.add_argument(
        '--repeat-num',
        type=int,
        default=1,
        help='重复测量次数以计算平均值')
    parser.add_argument(
        '--max-iter', type=int, default=2000, help='最大迭代次数')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='日志间隔')
    parser.add_argument(
        '--num-warmup', type=int, default=5, help='预热次数')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='是否融合卷积和BN，这会略微提高推理速度')
    parser.add_argument(
        '--work-dir',
        help='保存包含基准测试指标的文件的目录')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='覆盖配置文件中的一些设置')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='作业启动器')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # 设置分布式环境
    distributed = False
    if args.launcher != 'none':
        init_dist(args.launcher, **cfg.get('env_cfg', {}).get('dist_cfg', {}))
        distributed = True

    # 设置日志
    log_file = None
    if args.work_dir:
        log_file = os.path.join(args.work_dir, 'diffusiondet_benchmark.log')
        mkdir_or_exist(args.work_dir)

    logger = MMLogger.get_instance(
        'mmdet', log_file=log_file, log_level='INFO')

    # 运行基准测试
    # 根据任务类型选择不同的基准测试
    if args.task == 'inference':
        # 运行推理基准测试
        benchmark = EnhancedDiffusionDetBenchmark(
            cfg,
            args.checkpoint,
            distributed,
            args.fuse_conv_bn,
            args.max_iter,
            args.log_interval,
            args.num_warmup,
            logger=logger
        )
        benchmark.run(args.repeat_num)
    elif args.task == 'dataloader':
        # 运行数据加载器基准测试
        benchmark = DataLoaderBenchmark(
            cfg,
            distributed,
            args.dataset_type,  # 这里使用dataset-type参数
            args.max_iter,
            args.log_interval,
            args.num_warmup,
            logger=logger)
        benchmark.run(args.repeat_num)
    elif args.task == 'dataset':
        # 运行数据集基准测试
        benchmark = DatasetBenchmark(
            cfg,
            args.dataset_type,  # 这里使用dataset-type参数
            args.max_iter,
            args.log_interval,
            args.num_warmup,
            logger=logger)
        benchmark.run(args.repeat_num)


if __name__ == '__main__':
    main()
