#!/usr/bin/env python
import os
import argparse
import json


def verify_kfold_config(fold):
    """验证指定fold的数据集配置是否正确"""
    data_root = '/media/ross/8TB/project/lsh/dataset/microAlgea/microAlgaeOri/'

    # 检查训练集和验证集文件是否存在
    train_ann = f"{data_root}/annotations/kfold/train_fold{fold}.json"
    val_ann = f"{data_root}/annotations/kfold/val_fold{fold}.json"

    if not os.path.exists(train_ann):
        print(f"错误: 找不到训练集标注文件 {train_ann}")
        return False

    if not os.path.exists(val_ann):
        print(f"错误: 找不到验证集标注文件 {val_ann}")
        return False

    # 检查训练集和验证集是否有数据
    with open(train_ann, 'r') as f:
        train_data = json.load(f)

    with open(val_ann, 'r') as f:
        val_data = json.load(f)

    if len(train_data.get('images', [])) == 0:
        print(f"错误: 训练集 fold{fold} 没有图片")
        return False

    if len(val_data.get('images', [])) == 0:
        print(f"错误: 验证集 fold{fold} 没有图片")
        return False

    # 检查训练集和验证集是否有重叠
    train_ids = {img['id'] for img in train_data.get('images', [])}
    val_ids = {img['id'] for img in val_data.get('images', [])}
    overlap = train_ids.intersection(val_ids)

    if overlap:
        print(f"警告: 训练集和验证集有 {len(overlap)} 张重叠图片")

    # 检查图片文件是否存在
    img_dir = f"{data_root}/train_val/"
    missing = 0
    for img in train_data.get('images', [])[:10]:  # 只检查前10张
        img_path = os.path.join(img_dir, os.path.basename(img['file_name']))
        if not os.path.exists(img_path):
            missing += 1
            print(f"找不到图片: {img_path}")

    if missing > 0:
        print(f"警告: 至少 {missing} 张训练图片不存在")

    print(f"验证通过: Fold {fold} 配置正确")
    print(f"  - 训练集: {len(train_data.get('images', []))} 张图片")
    print(f"  - 验证集: {len(val_data.get('images', []))} 张图片")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='验证K折交叉验证配置')
    parser.add_argument('--fold', type=int, required=True, help='要验证的fold索引')
    args = parser.parse_args()

    verify_kfold_config(args.fold)
