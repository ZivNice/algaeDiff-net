import json
import os


def verify_dataset(ann_file, data_root):
    with open(ann_file, 'r') as f:
        data = json.load(f)

    missing_files = []
    for img in data['images']:
        file_name = img['file_name']
        # 去除可能的前缀路径，只保留文件名
        base_name = os.path.basename(file_name)
        full_path = os.path.join(data_root, 'train_val', base_name)
        if not os.path.exists(full_path):
            missing_files.append(base_name)

    if missing_files:
        print(f"发现 {len(missing_files)} 个缺失的图片:")
        for f in missing_files[:10]:  # 只显示前10个
            print(f)
        if len(missing_files) > 10:
            print("...")
    else:
        print("所有图片路径都正确!")

    return len(missing_files) == 0


if __name__ == '__main__':
    data_root = '/media/ross/8TB/project/lsh/dataset/microAlgea/microAlgaeOri/'
    for fold in range(5):
        print(f"\n检查 fold {fold}:")
        print("训练集:")
        verify_dataset(
            f"{data_root}/annotations/kfold/train_fold{fold}.json",
            data_root
        )
        print("\n验证集:")
        verify_dataset(
            f"{data_root}/annotations/kfold/val_fold{fold}.json",
            data_root
        )
