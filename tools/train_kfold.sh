#!/bin/bash
set -e  # 遇到错误立即退出

CONFIG="SOTA/work_dirs/diffusiondet_r50_fpn_epoch_microalgeaOri/diffusiondet_r50_5kfold_fpn_epoch_microalgeaOri.py"
GPUS=1
MAX_EPOCHS=300
PATIENCE=20  # 早停耐心值

# 确认K折数据已存在，如果不存在则创建
DATA_ROOT="/media/ross/8TB/project/lsh/dataset/microAlgea/microAlgaeOri/"
KFOLD_DIR="${DATA_ROOT}/annotations/kfold"

if [ ! -d "$KFOLD_DIR" ]; then
    echo "未找到K折数据集目录，请先运行数据集划分脚本"
    exit 1
fi

# 训练所有折
for FOLD in 0 1 2 3 4
do
    echo "==============================================="
    echo "开始验证第 ${FOLD} 折数据集配置..."

    # 验证数据集配置
    python tools/verify_kfold_config.py --fold ${FOLD}

    echo "开始训练第 ${FOLD} 折..."
    FOLD_WORK_DIR="./work_dirs/diffusiondet_r50_5kfold_fpn_microalgea_fold${FOLD}"

    # 如果目录已存在，询问是否继续
    if [ -d "$FOLD_WORK_DIR" ]; then
        echo "警告: 工作目录 ${FOLD_WORK_DIR} 已存在"
        read -p "继续训练并覆盖现有结果? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "跳过第 ${FOLD} 折训练"
            continue
        fi
    fi

    # 训练模型
    CUDA_VISIBLE_DEVICES=0 python tools/train.py \
        ${CONFIG} \
        --cfg-options fold=${FOLD} \
        max_epochs=${MAX_EPOCHS} \
        default_hooks.early_stopping.patience=${PATIENCE} \
        work_dir=${FOLD_WORK_DIR}

    echo "第 ${FOLD} 折训练完成"
    echo "==============================================="
done

echo "5折交叉验证完成!"
echo "可以通过运行 python tools/analyze_kfold_results.py 分析结果"
