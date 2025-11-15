
import os
import shutil
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split


# ============================
# CONFIG
# ============================
IMG_DIR = "../../data/tungbui/QA_pjico_test_set_17910_19519_team_AI_muon_new/images"
LBL_DIR = "../../data/tungbui/QA_pjico_test_set_17910_19519_team_AI_muon_new/labels"

VAL_RATIO = 0.5   # Test = phần còn lại


# ============================
# HÀM ĐẾM INSTANCE TRONG FILE LABEL YOLO POLYGON
# ============================
def count_instances(label_path):
    """
    Đếm số instance theo class trong một file label YOLO polygon.
    Mỗi dòng: class_id x1 y1 x2 y2 x3 y3 ...
    """
    class_count = defaultdict(int)
    if not os.path.exists(label_path):
        return class_count
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        try:
            cls_id = int(parts[0])
        except:
            continue
        class_count[cls_id] += 1
    return class_count

# ============================
# HÀM COPY ẢNH VÀ LABEL
# ============================
def copy_subset(img_list, img_dir, lbl_dir, subset):
    """
    Copy ảnh và label sang thư mục subset (val/test)
    """
    os.makedirs(os.path.join(img_dir, subset), exist_ok=True)
    os.makedirs(os.path.join(lbl_dir, subset), exist_ok=True)

    for img in img_list:
        shutil.copy(os.path.join(img_dir, img), os.path.join(img_dir, subset, img))
        lbl = os.path.splitext(img)[0] + ".txt"
        src_lbl = os.path.join(lbl_dir, lbl)
        dst_lbl = os.path.join(lbl_dir, subset, lbl)
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, dst_lbl)


def compute_distribution(label_dir, image_dir):
    """
    label_dir: thư mục labels/val hoặc labels/test
    image_dir: thư mục images/val hoặc images/test
    """
    total_counts = defaultdict(int)

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    for img in image_files:
        label_path = os.path.join(label_dir, img.rsplit(".",1)[0] + ".txt")
        instance_counts = count_instances(label_path)
        for cls, cnt in instance_counts.items():
            total_counts[cls] += cnt

    return total_counts


def print_distribution_1set(label_dir, img_dir):
    print("\n===== PHÂN BỐ INSTANCE =====")
    val_counts = compute_distribution(label_dir, img_dir)
    total_val = sum(val_counts.values())

    for cls in sorted(val_counts.keys()):
        count = val_counts[cls]
        ratio = count / total_val * 100 if total_val > 0 else 0
        print(f"  Class {cls}: {count} ({ratio:.2f}%)")


def print_distribution(val_label_dir, val_img_dir,
                       test_label_dir, test_img_dir):
    print("\n===== PHÂN BỐ INSTANCE TRONG VAL =====")
    val_counts = compute_distribution(val_label_dir, val_img_dir)
    total_val = sum(val_counts.values())

    for cls in sorted(val_counts.keys()):
        count = val_counts[cls]
        ratio = count / total_val * 100 if total_val > 0 else 0
        print(f"  Class {cls}: {count} ({ratio:.2f}%)")

    print("\n===== PHÂN BỐ INSTANCE TRONG TEST =====")
    test_counts = compute_distribution(test_label_dir, test_img_dir)
    total_test = sum(test_counts.values())

    for cls in sorted(test_counts.keys()):
        count = test_counts[cls]
        ratio = count / total_test * 100 if total_test > 0 else 0
        print(f"  Class {cls}: {count} ({ratio:.2f}%)")

    print("\n===== SO SÁNH VAL vs TEST =====")
    all_classes = sorted(set(val_counts.keys()) | set(test_counts.keys()))
    for cls in all_classes:
        v = val_counts.get(cls, 0)
        t = test_counts.get(cls, 0)
        print(f"Class {cls}: val={v}, test={t}, ratio val/test={v}/{t if t>0 else 1}")



# ============================
# HÀM CHÍNH SPLIT VAL/TEST
# ============================
def split_val_test(img_dir="images", lbl_dir="labels", val_ratio=0.2, random_state=42):
    random.seed(random_state)

    # Quét dataset
    image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    labeled_imgs = []
    labeled_major_classes = []
    unlabeled_imgs = []
    major_class_map = {}

    for img in image_files:
        label_file = os.path.splitext(img)[0] + ".txt"
        label_path = os.path.join(lbl_dir, label_file)
        instance_count = count_instances(label_path)
        if len(instance_count) == 0:
            unlabeled_imgs.append(img)
        else:
            major_class = max(instance_count, key=instance_count.get)
            labeled_imgs.append(img)
            labeled_major_classes.append(major_class)
            major_class_map[img] = major_class

    print("Ảnh có label:", len(labeled_imgs))
    print("Ảnh không có label:", len(unlabeled_imgs))

    # Tìm singleton (class chỉ xuất hiện 1 ảnh)
    class_image_counts = defaultdict(int)
    for cls in labeled_major_classes:
        class_image_counts[cls] += 1
    singleton_imgs = [img for img, cls in major_class_map.items() if class_image_counts[cls] == 1]

    remaining_labeled = [img for img in labeled_imgs if img not in singleton_imgs]
    remaining_labels = [major_class_map[img] for img in remaining_labeled]

    # Chia val/test cho ảnh labeled
    val_labeled = []
    test_labeled = []

    if len(remaining_labeled) > 0:
        train_remain, val_split, _, _ = train_test_split(
            remaining_labeled,
            remaining_labels,
            test_size=val_ratio,
            stratify=remaining_labels,
            random_state=random_state
        )
        val_labeled = val_split
        test_labeled = train_remain

    # Thêm singleton vào cả val và test
    val_labeled += singleton_imgs
    test_labeled += singleton_imgs

    # Chia ảnh không label
    random.shuffle(unlabeled_imgs)
    unlabeled_val_count = int(len(unlabeled_imgs) * val_ratio)
    val_unlabeled = unlabeled_imgs[:unlabeled_val_count]
    test_unlabeled = unlabeled_imgs[unlabeled_val_count:]

    # Ghép cuối cùng
    val_imgs = val_labeled + val_unlabeled
    test_imgs = test_labeled + test_unlabeled
    val_imgs = list(dict.fromkeys(val_imgs))
    test_imgs = list(dict.fromkeys(test_imgs))

    # Copy file
    copy_subset(val_imgs, img_dir, lbl_dir, "val")
    copy_subset(test_imgs, img_dir, lbl_dir, "test")

    # In thống kê
    print(f"Hoàn tất chia dữ liệu: val={len(val_imgs)}, test={len(test_imgs)}")
    print(f"Val có label: {sum(1 for img in val_imgs if os.path.exists(os.path.join(lbl_dir, os.path.splitext(img)[0]+'.txt')))}, "
          f"Val không label: {sum(1 for img in val_imgs if not os.path.exists(os.path.join(lbl_dir, os.path.splitext(img)[0]+'.txt')))}")
    print(f"Test có label: {sum(1 for img in test_imgs if os.path.exists(os.path.join(lbl_dir, os.path.splitext(img)[0]+'.txt')))}, "
          f"Test không label: {sum(1 for img in test_imgs if not os.path.exists(os.path.join(lbl_dir, os.path.splitext(img)[0]+'.txt')))}")

    return val_imgs, test_imgs


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    #val_imgs, test_imgs = split_val_test(img_dir=IMG_DIR, lbl_dir=LBL_DIR, val_ratio=VAL_RATIO, random_state=42)

    #print_distribution("../../data/tungbui/QA_pjico_test_set_17910_19519_team_AI_muon_new/labels/val", "../../data/tungbui/QA_pjico_test_set_17910_19519_team_AI_muon_new/images/val", "../../data/tungbui/QA_pjico_test_set_17910_19519_team_AI_muon_new/labels/test", "../../data/tungbui/QA_pjico_test_set_17910_19519_team_AI_muon_new/images/test")

    print_distribution_1set("../../data/tungbui/dataset_tung_1k/labels/train", "../../data/tungbui/dataset_tung_1k/images/train",)

