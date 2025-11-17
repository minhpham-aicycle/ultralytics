import os
import yaml

# ==== 1. Cập nhật data.yaml ====
data_yaml_path = "../../data/tungbui/dataset_tung_1k_clean_class/data.yaml"
keep_classes = [39, 40, 41, 42, 66, 69]

with open(data_yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

# Lọc tên class
names = data["names"]
new_names = [names[i] for i in keep_classes]

# Cập nhật số class và tên
data["nc"] = len(new_names)
data["names"] = new_names

# Ghi lại file data.yaml
with open("data_new.yaml", "w", encoding="utf-8") as f:
    yaml.dump(data, f, allow_unicode=True)

print(f"data.yaml updated: nc={data['nc']}, names={data['names']}")

# ==== 2. Cập nhật file label trong folders train/val/test ====
label_dirs = [
    os.path.join(data["path"], "labels", "train"),
    os.path.join(data["path"], "labels", "val"),
    os.path.join(data["path"], "labels", "test"),
]

# Map old class index -> new class index
class_map = {old: new for new, old in enumerate(keep_classes)}

for label_dir in label_dirs:
    if not os.path.exists(label_dir):
        print(f"Warning: folder not found: {label_dir}")
        continue
    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(label_dir, fname)
        new_lines = []
        with open(fpath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                if cls in class_map:
                    parts[0] = str(class_map[cls])
                    new_lines.append(" ".join(parts))
        # Ghi đè file label
        with open(fpath, "w") as f:
            f.write("\n".join(new_lines))
