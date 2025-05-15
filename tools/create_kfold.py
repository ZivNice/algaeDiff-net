import json
import os
import random
from sklearn.model_selection import KFold
import numpy as np


def create_kfold_annotations(ann_file, output_dir, n_splits=5, seed=42):
    # 设置随机种子确保可重复性
    random.seed(seed)
    np.random.seed(seed)

    # 加载标注文件
    with open(ann_file, 'r') as f:
        data = json.load(f)

    # 获取所有图像ID
    image_ids = [img['id'] for img in data['images']]

    # 创建KFold对象
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 为每个fold生成训练和验证集
    for fold, (train_idx, val_idx) in enumerate(kf.split(image_ids)):
        train_ids = [image_ids[i] for i in train_idx]
        val_ids = [image_ids[i] for i in val_idx]

        # 创建训练集标注，确保ID唯一性
        train_images = [img for img in data['images'] if img['id'] in train_ids]
        train_annotations = []
        current_id = 0
        for ann in data['annotations']:
            if ann['image_id'] in train_ids:
                ann_copy = ann.copy()
                ann_copy['id'] = current_id
                train_annotations.append(ann_copy)
                current_id += 1

        train_data = {
            'info': data['info'],
            'licenses': data['licenses'],
            'categories': data['categories'],
            'images': train_images,
            'annotations': train_annotations
        }

        # 创建验证集标注，确保ID唯一性
        val_images = [img for img in data['images'] if img['id'] in val_ids]
        val_annotations = []
        current_id = 0
        for ann in data['annotations']:
            if ann['image_id'] in val_ids:
                ann_copy = ann.copy()
                ann_copy['id'] = current_id
                val_annotations.append(ann_copy)
                current_id += 1

        val_data = {
            'info': data['info'],
            'licenses': data['licenses'],
            'categories': data['categories'],
            'images': val_images,
            'annotations': val_annotations
        }

        # 保存标注文件
        with open(f'{output_dir}/train_fold{fold}.json', 'w') as f:
            json.dump(train_data, f)
        with open(f'{output_dir}/val_fold{fold}.json', 'w') as f:
            json.dump(val_data, f)

        print(f"Fold {fold}: 训练集 {len(train_images)} 张图片, {len(train_annotations)} 个标注")
        print(f"Fold {fold}: 验证集 {len(val_images)} 张图片, {len(val_annotations)} 个标注")


if __name__ == '__main__':
    data_root = '/media/ross/8TB/project/lsh/dataset/microAlgea/microAlgaeOri/'
    create_kfold_annotations(
        ann_file=os.path.join(data_root, 'annotations/instances_trainval.json'),
        output_dir=os.path.join(data_root, 'annotations/kfold'),
        n_splits=5
    )
