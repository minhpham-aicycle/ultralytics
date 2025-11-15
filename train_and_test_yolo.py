

from ultralytics import YOLO
from ultralytics import RTDETR

# Đường dẫn tới file cấu hình dữ liệu
DATA_YAML_PATH = 'data1k.yaml'  # Cập nhật đường dẫn nếu cần


import yaml
from pathlib import Path
from PIL import Image
import numpy as np

import torch
import os


# ---------------------------
# Các tiện ích chuyển đổi, tính IoU
# ---------------------------
def xywhn_to_xyxy(box, im_w, im_h):
    """Convert normalized xywh -> xyxy pixel coords."""
    x, y, w, h = box
    x *= im_w; y *= im_h; w *= im_w; h *= im_h
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.array([x1, y1, x2, y2], dtype=float)

def iou_xyxy(boxA, boxB):
    """IoU between two xyxy boxes."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

# ---------------------------
# Suy ra đường dẫn images/labels cho tập test
# ---------------------------
def resolve_paths_from_yaml(data_yaml_path):
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    root = Path(data.get('path', '.'))
    test_rel = data.get('test', None)
    if test_rel is None:
        raise ValueError("data.yaml không khai báo 'test:'")

    # test có thể là đường dẫn tương đối so với 'path' hoặc tuyệt đối.
    test_images_dir = (root / test_rel).resolve() if not Path(test_rel).is_absolute() else Path(test_rel)

    # SUY LUẬN labels: thay 'images' -> 'labels' ở cấp tương ứng.
    parts = list(test_images_dir.parts)
    try:
        idx = len(parts) - 1 - parts[::-1].index('images')
        parts[idx] = 'labels'
        test_labels_dir = Path(*parts)
    except ValueError:
        # nếu không có 'images' trong đường dẫn, mặc định labels là sibling 'labels'
        test_labels_dir = test_images_dir.parent / 'labels'

    return test_images_dir, test_labels_dir, data.get('names', None)

# ---------------------------
# Đọc nhãn YOLO từ .txt
# ---------------------------
def read_yolo_labels(label_path, im_w, im_h):
    """
    Đọc file .txt theo định dạng: class x y w h (normalized).
    Trả về list (cls_id, box_xyxy).
    """
    results = []
    if not label_path.exists():
        return results
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split()
            if len(vals) < 5:
                continue
            cls_id = int(float(vals[0]))
            x, y, w, h = map(float, vals[1:5])
            box_xyxy = xywhn_to_xyxy((x, y, w, h), im_w, im_h)
            results.append((cls_id, box_xyxy))
    return results

# ---------------------------
# Tính Precision/Recall cho một lớp ở conf=0.25 và IoU=0.5
# ---------------------------
def precision_recall_for_class(model, data_yaml_path, class_name_or_id, conf=0.25, iou_thr=0.5):
    test_images_dir, test_labels_dir, names = resolve_paths_from_yaml(data_yaml_path)

    # Map class name -> id
    if isinstance(class_name_or_id, str):
        if isinstance(names, dict):  # {id: name}
            inv = {v: k for k, v in names.items()}
            cls_id = inv.get(class_name_or_id)
        elif isinstance(names, list):  # [name0, name1, ...]
            cls_id = names.index(class_name_or_id)
        else:
            raise ValueError("Không tìm thấy 'names' trong data.yaml để map class name -> id.")
    else:
        cls_id = int(class_name_or_id)

    # Liệt kê ảnh  
    allowed = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_paths =  [p for p in test_images_dir.rglob('*') if p.is_file() and p.suffix.lower() in allowed]
    image_paths.sort()

    if not image_paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh test trong: {test_images_dir}")

    print(test_images_dir)
    print(len(image_paths))
    #print(image_paths)

    TP = 0
    FP = 0
    FN = 0

    for im_path in image_paths:
        # Kích thước ảnh để chuyển nhãn
        with Image.open(im_path) as im:
            im_w, im_h = im.size

        # Ground truth (chỉ lấy box thuộc cls_id)
        label_path = test_labels_dir / (im_path.stem + '.txt')
        gts = [b for (c, b) in read_yolo_labels(label_path, im_w, im_h) if c == cls_id]
        gt_used = np.zeros(len(gts), dtype=bool)

        # Dự đoán ở conf=0.25
        results = model.predict(source=str(im_path), conf=conf, iou=iou_thr, verbose=False)
        pred_boxes = []
        pred_confs = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            # Lọc theo lớp
            for b in r.boxes:
                c = int(b.cls.item())
                if c != cls_id:
                    continue
                xyxy = b.xyxy.cpu().numpy().reshape(-1)  # [x1,y1,x2,y2]
                pred_boxes.append(xyxy)
                pred_confs.append(float(b.conf.item()))

        # Sắp theo confidence giảm dần để match greedy
        order = np.argsort(-np.array(pred_confs)) if pred_confs else np.array([])
        pred_boxes = [pred_boxes[i] for i in order]

        # Ghép dự đoán với GT theo IoU >= iou_thr
        for pb in pred_boxes:
            best_iou = 0.0
            best_j = -1
            for j, gb in enumerate(gts):
                if gt_used[j]:
                    continue
                iou = iou_xyxy(pb, gb)
                if iou >= iou_thr and iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0:
                TP += 1
                gt_used[best_j] = True
            else:
                FP += 1

        FN += int((~gt_used).sum())

    precision = TP / (TP + FP) if (TP + FP) > 0 else float('nan')
    recall = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
    return precision, recall, {'TP': TP, 'FP': FP, 'FN': FN, 'cls_id': cls_id, 'conf': conf, 'iou_thr': iou_thr,
                               'num_images': len(image_paths)}


def train():
    #model = YOLO('yolov8m.pt')  # Có thể thay bằng yolov8s.pt, yolov8m.pt, v.v.
    #model = YOLO('yolov9c.pt')  # Có thể thay bằng 'yolov9c.pt yolo11m.pt yolov10n-seg.pt, yolov10s-seg.pt, yolov10l-seg.pt, v.v.
    #model = YOLO('yolov9c.pt')
    #model.info()
    #model = YOLO('yolo11m.pt')
    #model.info()
    #model = YOLO('yolo11m-seg.pt')
    #model.info()
    #model = YOLO('yolo12m.pt')
    #model.info()
    #model = RTDETR("rtdetr-l.pt")
    #model.info()
    
    model = YOLO("yolov9c.pt")  
    model.train(cfg="train_yolov9c_config.yml")

    #model.train(data=DATA_YAML_PATH, epochs=200, patience=20, imgsz=1024,     

        # ----- Augmentations -----
    #    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,   # tăng đa dạng màu sắc nhẹ
    #    degrees=5.0, translate=0.05, scale=0.5, shear=0.0, perspective=0.0,
    #    fliplr=0.5, flipud=0.0,
    #    mosaic=1.0, mixup=0.1, copy_paste=0.0,  # có thể tăng giảm theo GPU/dataset
    #    erasing=0.0,                            # Random Erasing (nếu cần)
    #    close_mosaic=10,                        # tắt mosaic ở 10 epoch cuối để mô hình “ổn định” trước khi kết thúc

        #iou=0.5,   
        #conf=0.01, 
    #    project='runs/segment',
    #    name='my_experiment'  # => runs/segment/my_experiment
    #)


def issue4_fn_test():
    #model = YOLO('../../data/tungbui/models/car_damage_segmentation/CarDamage_20250926.pt') 
    #CLASS_NAME = 42
    #p, r, info = precision_recall_for_class(model, DATA_YAML_PATH, CLASS_NAME, conf=0.2, iou_thr=0.3)
    #print(f"[{CLASS_NAME}] @conf=0.25, IoU=0.5 -> Precision={p:.3f}, Recall={r:.3f} | {info}")
    DATA_YAML_PATH = '../../data/tungbui/dataset1/data.yaml'


    # 1) Load model đã có sẵn
    #model = YOLO('../../data/tungbui/models/car_damage_segmentation/CarDamage_20250926.pt')
    model = YOLO('../../data/tungbui/outputs/train/car_damage_seg_resume_20250926/weights/best.pt')
    #model = YOLO('../../data/tungbui/outputs/train/car_damage_seg_resume_20250926_stage2_unfreeze/weights/best.pt')

    # 2) Khai báo nguồn ảnh và thư mục lưu kết quả
    source_dir = '../../data/tungbui/images'

    # Với model segmentation, nên đặt project vào nhánh 'segment' cho dễ quản lý
    project_dir = '../../data/tungbui/outputs/segment'   # thư mục gốc lưu kết quả
    run_name   = 'car_damage_20250926_2'                   # tên phiên chạy (sub-folder)

    # Tạo thư mục nếu chưa có
    os.makedirs(project_dir, exist_ok=True)

    # 3) Chạy suy luận (inference) và lưu ảnh kết quả
    results = model.predict(
        source=source_dir,        # thư mục chứa ảnh cần xử lý
        save=True,                # lưu ảnh đã vẽ box/mask
        project=project_dir,      # thư mục gốc lưu kết quả
        name=run_name,            # tên phiên chạy -> ảnh sẽ nằm trong <project>/<name>
        conf=0.1,                # ngưỡng confidence (tùy chỉnh theo nhu cầu)
        imgsz=1280,               # kích thước suy luận (tùy GPU/CPU)
        device=0 if torch.cuda.is_available() else 'cpu',  # ưu tiên GPU nếu có
        save_conf=True,           # lưu score lên ảnh
        save_txt=False,           # bật True nếu muốn lưu nhãn .txt (YOLO format)
        verbose=True,
        retina_masks=True         # (segmentation) mask sắc nét hơn
    )

    # 4) Thông báo đường dẫn đầu ra cho người dùng
    out_dir = os.path.join(project_dir, run_name)
    print(f'Ảnh kết quả đã được lưu tại: {out_dir}')


def issue4_fn_fix():

    # 1) Load checkpoint segmentation hiện có
    model = YOLO('../../data/tungbui/models/car_damage_segmentation/CarDamage_20250926.pt')

    # 2) Chỉ ra data.yaml trong dataset1
    data_cfg = '../../data/tungbui/dataset2/data.yaml'  # chỉnh lại cho đúng vị trí

    # 3) Cấu hình thư mục đầu ra (để dễ tìm)
    project_dir = '../../data/tungbui/outputs/train'
    run_name   = 'car_damage_seg_resume_20250926_2'
    os.makedirs(project_dir, exist_ok=True)

    # 4) Train tiếp (transfer learning từ checkpoint)
    #model.train(
    #    data=data_cfg,               # đường dẫn data.yaml (segmentation)
    #    project=project_dir,         # thư mục gốc lưu kết quả train
    #    name=run_name,               # tên phiên train
    #    epochs=10,                   # số epoch (tùy chỉnh)
    #    batch=8,                     # batch size (tùy GPU/CPU)
    #    imgsz=1024,                  # kích thước ảnh train
    #    device=0 if torch.cuda.is_available() else 'cpu',
    #    workers=4,
    #    lr0=0.001,                   # learning rate ban đầu (tùy chỉnh)
    #    cos_lr=True,                 # lịch LR cosine
    #    patience=5,                 # early stopping
    #    amp=True,                    # mixed precision (nhanh, tiết kiệm VRAM)
    #    cache=True,                  # cache dataset để train nhanh hơn
    #)

    # === GIAI ĐOẠN 1: freeze backbone để ổn định ===
    #model.train(
    #     data=data_cfg,
    #     project=project_dir,
    #     name=f"{run_name}_stage1_freeze",
    #     epochs=10,                  # 10 epoch đầu: chỉ train head
    #     freeze=10,                  # freeze ~backbone (10 layer đầu)
    #     batch=8,
    #     imgsz=1024,
    #     device=0 if torch.cuda.is_available() else 'cpu',
    #     lr0=0.001,                  # learning rate vừa phải
    #     cos_lr=True,
    #     patience=5,
    #     amp=True,
    #     cache=True,

    #     # ---- AUGMENTATION MẠNH ----
    #     hsv_h=0.015,
    #     hsv_s=0.7,
    #     hsv_v=0.4,
    #     degrees=5.0,
    #     translate=0.1,
    #     scale=0.8,
    #     shear=2.0,
    #     perspective=0.0005,
    #     flipud=0.1,
    #     fliplr=0.5,
    #     mosaic=1.0,
    #     mixup=0.2,
    #     copy_paste=0.3,
    # )

    # === GIAI ĐOẠN 2: unfreeze backbone để fine-tune toàn bộ ===
    #best_stage1 = Path(project_dir) / f"{run_name}_stage1_freeze" / "weights" / "best.pt"
    #model = YOLO(best_stage1)

    model.train(
        data=data_cfg,
        project=project_dir,
        name=run_name, 
        epochs=100,                      # 10–30 epoch là đủ
        patience=20,
        lr0=0.0003,                     # nhỏ hơn lần 1 (ví dụ 1/3)
        cos_lr=True,
        batch=8,
        imgsz=1024,
        device=0 if torch.cuda.is_available() else 'cpu',
        freeze=0,                       # không freeze để fine-tune toàn bộ
        mosaic=0.5,                     # nhẹ thôi, vì ảnh đã chuyên biệt
        mixup=0.1,
        copy_paste=0.3,
        fliplr=0.4,
        amp=True,
        cache=True,
    )

    # 5) In đường dẫn model sau khi train
    out_dir = os.path.join(project_dir, run_name, 'weights')
    print(f'Model sau khi train được lưu tại: {out_dir}')
    print('Trong thư mục này có các file: best.pt (tốt nhất) và last.pt (cuối cùng).')


def model_evaluation():
    #model = YOLO('../../data/tungbui/models/car_damage_segmentation/CarDamage_20250926.pt')

    model = YOLO('../../data/tungbui/outputs/train/car_damage_seg_resume_20250926/weights/best.pt')
    
    # Đánh giá trên toàn bộ dataset
    metrics = model.val(data='../../data/inference/data.yaml', split='val')

    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p:.4f}")
    print(f"Recall: {metrics.box.r:.4f}")
    print(f"All_ap: {metrics.box.all_ap:.4f}")
    print(f"All_ap: {metrics.box.ap_class_index:.4f}")

if __name__ == "__main__":
    #model_evaluation()
    #issue4_fn_test()
    #issue4_fn_fix()

    train()
    
    #CLASS_NAME = 42   
    #model = YOLO('runs/segment/my_experiment3/weights/best.pt') 
    #p, r, info = precision_recall_for_class(model, DATA_YAML_PATH, CLASS_NAME, conf=0.2, iou_thr=0.3)
    #print(f"[{CLASS_NAME}] @conf=0.25, IoU=0.5 -> Precision={p:.3f}, Recall={r:.3f} | {info}")




    # Đánh giá mô hình sau huấn luyện
    #best_model = YOLO('runs/segment/my_experiment/weights/best.pt')
    #best_model.val(split='test')


    #import torchvision
    #print(hasattr(torchvision.ops, "nms"))
