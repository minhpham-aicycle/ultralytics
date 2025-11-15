import os
import yaml

LABELS_DIR = '../data/dataset100/labels'  # Cập nhật nếu cần

class_ids = set()

for filename in os.listdir(LABELS_DIR):
    if filename.endswith('.txt'):
        with open(os.path.join(LABELS_DIR, filename), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_ids.add(class_id)

max_class_id = max(class_ids)
names = [f'class_{i}' for i in range(max_class_id + 1)]

data_yaml = {
    'path': '../s../data/dataset100/',
    'train': 'images/train',
    'val': 'images/val',
    'nc': len(names),
    'names': names
}

with open('data.yaml', 'w') as f:
    yaml.dump(data_yaml, f, sort_keys=False)

print("Đã tạo file data.yaml với thông tin từ thư mục labels.")
