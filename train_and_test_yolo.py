
import numpy as np
from ultralytics import YOLO
from ultralytics import RTDETR
from ultralytics.utils import LOGGER
import torch

import argparse
import logging
#from loguru import logger
import re
import pty
import pandas as pd

from tqdm import tqdm
import yaml
import os
import glob
from pathlib import Path
from PIL import Image
import sys
import shutil
import cv2
import io
import math
from PIL import Image, ImageFile
import binascii
import albumentations as A
import boto3

# Đường dẫn tới file cấu hình dữ liệu
DATA_YAML_PATH = 'data1k.yaml'  # Cập nhật đường dẫn nếu cần

# Cho phép PIL đọc ảnh bị lỗi cụt (truncated) để kiểm tra sâu hơn
ImageFile.LOAD_TRUNCATED_IMAGES = True


# --- CẤU HÌNH MÀU SẮC HIỂN THỊ (Global) ---
CLASS_COLORS = {
    0: (0, 255, 0),   # Vết xước - Green
    1: (0, 0, 255),   # Vết móp - Red
    2: (255, 0, 0)    # Khác - Blue
}




def upload_to_s3(local_file = "mydata.zip", bucket = "aicycle-ai-advisor-train", 
                 s3_folder = "a100_uploads/",    
            aws_access_key_id = "aws_access_key_id", 
            aws_secret_access_key="aws_secret_access_key", region="ap-northeast-2"):
    # Tạo session S3 với key trực tiếp
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region
    )
    
    # Lấy tên file từ local_file và ghép với folder trên S3
    file_name = os.path.basename(local_file)
    s3_path = os.path.join(s3_folder, file_name)
    
    try:
        s3.upload_file(local_file, bucket, s3_path)
        print(f"Upload thành công: s3://{bucket}/{s3_path}")
    except Exception as e:
        print("Lỗi upload:", e)


def get_aug_pipeline():
    """Pipeline augmentation cho segmentation."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.HueSaturationValue(p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=0.1),
    ])


def yolo_to_masks(label_path, img_shape):
    """Đọc file YOLO polygon -> List Masks."""
    h, w = img_shape[:2]
    masks = []
    class_ids = []
    
    if not os.path.exists(label_path):
        return masks, class_ids

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i+1] * h)
            points.append([x, y])
        
        mask = np.zeros((h, w), dtype=np.uint8)
        points_array = np.array([points], dtype=np.int32)
        cv2.fillPoly(mask, points_array, 255)
        
        masks.append(mask)
        class_ids.append(class_id)
        
    return masks, class_ids

def masks_to_yolo(masks, class_ids, img_shape):
    """Chuyển List Masks -> YOLO polygon lines."""
    h, w = img_shape[:2]
    yolo_lines = []
    contours_to_draw = [] 
    classes_to_draw = []

    for mask, class_id in zip(masks, class_ids):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) < 50: continue # Lọc nhiễu nhỏ
            
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            if len(approx) < 3: continue

            normalized_coords = []
            for point in approx:
                x, y = point[0]
                normalized_coords.append(f"{x/w:.6f} {y/h:.6f}")
            
            line_str = f"{class_id} {' '.join(normalized_coords)}\n"
            yolo_lines.append(line_str)
            contours_to_draw.append(approx)
            classes_to_draw.append(class_id)

    return yolo_lines, contours_to_draw, classes_to_draw

def visualize_result(image, contours, class_ids):
    """Vẽ kết quả lên ảnh để review."""
    vis_img = image.copy()
    for cnt, cls_id in zip(contours, class_ids):
        color = CLASS_COLORS.get(cls_id, (0, 255, 255))
        cv2.drawContours(vis_img, [cnt], -1, color, 2)
        
        if len(cnt) > 0:
            first_point = tuple(cnt[0][0])
            cv2.putText(vis_img, f"ID:{cls_id}", first_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis_img


def run_interactive_augmentation(input_img_dir, input_label_dir, output_img_dir, output_label_dir, augment_times=3, image_ext="*.jpg"):
    """
    Hàm thực hiện augmentation có kiểm tra tương tác
    
    Args:
        input_img_dir (str): Đường dẫn folder ảnh gốc.
        input_label_dir (str): Đường dẫn folder nhãn gốc.
        output_img_dir (str): Đường dẫn folder lưu ảnh kết quả.
        output_label_dir (str): Đường dẫn folder lưu nhãn kết quả.
        augment_times (int): Số lượng ảnh biến thể muốn tạo ra từ 1 ảnh gốc.
        image_ext (str): Đuôi file ảnh cần quét (mặc định *.jpg).
    """
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Lấy danh sách ảnh
    transform = get_aug_pipeline()
    img_paths = glob.glob(os.path.join(input_img_dir, image_ext))
    img_paths.sort()

    if not img_paths:
        print(f"Cảnh báo: Không tìm thấy ảnh nào trong {input_img_dir}")
        return

    print(f"--- BẮT ĐẦU AUGMENTATION: {len(img_paths)} ảnh gốc ---")
    print(f"--- Cài đặt: {augment_times} biến thể/ảnh ---")
    print("HƯỚNG DẪN: Xem ảnh -> Gõ 'y' (Lưu), 'n' (Bỏ), 'q' (Thoát) -> Enter.")
    print("-" * 60)

    for img_idx, img_path in enumerate(img_paths):
        # 1. Load Resource
        image = cv2.imread(img_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        base_name = os.path.basename(img_path)
        name_only = os.path.splitext(base_name)[0]
        label_path = os.path.join(input_label_dir, name_only + ".txt")
        
        # 2. Convert YOLO -> Mask
        masks, class_ids = yolo_to_masks(label_path, (h, w))
        
        if not masks:
            print(f"[{img_idx+1}/{len(img_paths)}] Bỏ qua {base_name} (Không có label)")
            continue

        print(f"\n[{img_idx+1}/{len(img_paths)}] Đang xử lý: {base_name}")

        # 3. Loop Augmentation
        for i in range(augment_times):
            try:
                transformed = transform(image=image, masks=masks)
                aug_image = transformed['image']
                aug_masks = transformed['masks']
                
                # 4. Convert Mask -> YOLO & Visualization info
                new_lines, contours, new_class_ids = masks_to_yolo(aug_masks, class_ids, aug_image.shape)
                
                if not new_lines: continue # Mất hết object thì skip

                # 5. Visualization
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                vis_img = visualize_result(aug_image_bgr, contours, new_class_ids)
                
                # Resize hiển thị
                view_h, view_w = vis_img.shape[:2]
                if view_w > 1000:
                    scale = 1000 / view_w
                    vis_img = cv2.resize(vis_img, (int(view_w*scale), int(view_h*scale)))

                cv2.imshow("Review Augmentation (y/n/q)", vis_img)
                cv2.waitKey(100) 
                
                # 6. Human Check Loop
                while True:
                    user_input = input(f"  -> Biến thể {i+1}: Giữ? (y/n/q): ").strip().lower()
                    
                    if user_input == 'y':
                        save_name = f"{name_only}_aug_{i}"
                        # Lưu ảnh
                        cv2.imwrite(os.path.join(output_img_dir, save_name + ".jpg"), aug_image_bgr)
                        # Lưu label
                        with open(os.path.join(output_label_dir, save_name + ".txt"), 'w') as f:
                            f.writelines(new_lines)
                        print("     [Đã Lưu]")
                        break
                    elif user_input == 'n':
                        print("     [Bỏ qua]")
                        break
                    elif user_input == 'q':
                        print("Đã thoát chương trình theo yêu cầu.")
                        cv2.destroyAllWindows()
                        return 
                    
            except Exception as e:
                print(f"Lỗi: {e}")

    cv2.destroyAllWindows()
    print("\n--- HOÀN TẤT QUÁ TRÌNH ---")


def get_hex_signature(file_path, num_bytes=4):
    """Đọc các byte đầu tiên (Magic Number) để xác định định dạng thật"""
    try:
        with open(file_path, 'rb') as f:
            return binascii.hexlify(f.read(num_bytes)).decode('utf-8').upper()
    except:
        return "Error"

def analyze_image_header(file_path):
    """Trích xuất thông tin header chi tiết"""
    info = {}
    
    # 1. Kiểm tra kích thước file
    try:
        info['File Size'] = f"{os.path.getsize(file_path) / 1024:.2f} KB"
    except:
        info['File Size'] = "Not Found"
        return info

    # 2. Kiểm tra Magic Bytes (Chữ ký file)
    # FFD8FF... là JPG, 89504E47... là PNG
    info['Hex Signature'] = get_hex_signature(file_path)

    # 3. Phân tích bằng PILLOW
    try:
        with Image.open(file_path) as img:
            info['PIL Format'] = img.format  # Định dạng thực tế (JPEG, PNG...)
            info['PIL Mode'] = img.mode      # Hệ màu (RGB, CMYK, P, L...)
            info['Dimensions'] = img.size    # Kích thước ảnh
            
            # Kiểm tra thông tin phụ trong Header
            info['Is Progressive'] = img.info.get('progressive', False)
            info['Has ICC Profile'] = 'icc_profile' in img.info
            info['Has Adobe Marker'] = 'adobe' in img.info
            info['Has Exif'] = 'exif' in img.info
            
    except Exception as e:
        info['PIL Status'] = f"CRASH: {str(e)}"

    # 4. Phân tích bằng OPENCV (Cái mà CVAT dùng)
    try:
        img_cv = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img_cv is None:
            info['OpenCV Read'] = "FAILED (None)"
        else:
            info['OpenCV Read'] = "SUCCESS"
            info['OpenCV Shape'] = img_cv.shape # (Height, Width, Channels)
    except Exception as e:
        info['OpenCV Read'] = f"ERROR: {str(e)}"

    return info

def print_comparison(path_bad, path_good):
    print(f"{'ATTR':<20} | {'FILE LỖI (Ban đầu)':<30} | {'FILE SỬA (Resaved)':<30}")
    print("-" * 90)
    
    data_bad = analyze_image_header(path_bad)
    data_good = analyze_image_header(path_good)
    
    # Gộp tất cả các key để so sánh
    all_keys = sorted(list(set(data_bad.keys()) | set(data_good.keys())))
    
    # Ưu tiên hiển thị các thông số quan trọng trước
    priority_keys = ['OpenCV Read', 'PIL Mode', 'PIL Format', 'Hex Signature', 'OpenCV Shape']
    for k in priority_keys:
        if k in all_keys:
            all_keys.remove(k)
            all_keys.insert(0, k)

    for key in all_keys:
        val1 = str(data_bad.get(key, "N/A"))
        val2 = str(data_good.get(key, "N/A"))
        
        # Tô đậm sự khác biệt (bằng dấu *)
        diff_mark = "  "
        if val1 != val2:
            diff_mark = "->" 
        
        print(f"{diff_mark} {key:<18} | {val1:<30} | {val2:<30}")


def check_image_deeply(image_path):
    print(f"\nĐang kiểm tra kỹ thuật số file: {os.path.basename(image_path)}")
    
    if not os.path.exists(image_path):
        print("Lỗi: File không tồn tại.")
        return

    # --- BƯỚC 1: KIỂM TRA BẰNG PILLOW (PIL) ---
    # PIL giỏi hơn OpenCV trong việc phát hiện Mode màu (CMYK, P, L...) và định dạng gốc
    try:
        with Image.open(image_path) as img:
            print(f"   [PIL Info] Format: {img.format}")
            print(f"   [PIL Info] Mode: {img.mode}") # Đây là cái cần tìm (CMYK vs RGB)
            print(f"   [PIL Info] Size: {img.size}")
            
            # Kiểm tra CMYK
            if img.mode == 'CMYK':
                print("  CẢNH BÁO: Ảnh đang ở hệ màu CMYK!")
                print("   -> OpenCV thường gặp lỗi khi xử lý CMYK trực tiếp.")
                print("   -> Cần convert sang RGB.")
                return "CMYK_DETECTED"
            
            # Kiểm tra Mode lạ khác (P - Palette, L - Grayscale, RGBA)
            if img.mode not in ['RGB', 'BGR']:
                print(f" CẢNH BÁO: Ảnh không phải RGB chuẩn (Đang là {img.mode}).")
                return "NON_RGB_DETECTED"
                
            # Kiểm tra xem đuôi file có khớp định dạng thật không
            ext = os.path.splitext(image_path)[1].lower()
            if (img.format == 'JPEG' and ext not in ['.jpg', '.jpeg']) or \
               (img.format == 'PNG' and ext != '.png'):
                print(f"CẢNH BÁO: Đuôi file là {ext} nhưng định dạng thực là {img.format}.")
                return "FORMAT_MISMATCH"

            # Thử load toàn bộ dữ liệu để xem có bị cụt (truncated) không
            img.load()
            print("[PIL] Load dữ liệu ảnh thành công (Không bị truncated).")

    except OSError as e:
        print(f"[PIL LỖI] Không thể mở ảnh: {e}")
        if "broken data stream" in str(e):
             print("   -> Ảnh bị hỏng file (Corrupt/Truncated).")
        return "CORRUPT"
    except Exception as e:
        print(f"[PIL LỖI KHÁC] {e}")
        return "ERROR"

    # --- BƯỚC 2: KIỂM TRA LẠI BẰNG OPENCV ---
    # Để xem OpenCV trên máy này có đọc được không
    try:
        img_cv = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img_cv is None:
            print("[OpenCV] Trả về None (Không decode được).")
            return "OPENCV_FAIL"
        else:
            print(f"[OpenCV] Decode thành công. Shape: {img_cv.shape}")
            # Kiểm tra số kênh màu
            if len(img_cv.shape) == 3 and img_cv.shape[2] == 4:
                 print("Cảnh báo: Ảnh có 4 kênh màu (BGRA/Transparency).")
    except Exception as e:
        print(f"[OpenCV LỖI] {e}")
        return "OPENCV_ERROR"

    print("-> File có vẻ ổn định trên môi trường này.")
    return "OK"


def convert_to_rgb_safe(image_path):
    """Hàm tự động sửa ảnh về dạng RGB chuẩn"""
    try:
        output_path = image_path # Ghi đè luôn
        img = Image.open(image_path)
        
        # Nếu là CMYK hoặc RGBA, convert sang RGB
        if img.mode in ['CMYK', 'RGBA', 'P']:
            print(f"Đang convert {img.mode} sang RGB...")
            img = img.convert('RGB')
            img.save(output_path, "JPEG", quality=100)
            print("Đã lưu lại ảnh dưới dạng RGB chuẩn.")
        else:
            # Kể cả là RGB, lưu lại 1 lần để Pillow viết lại Header chuẩn
            print("Đang re-save để sửa header...")
            img.save(output_path, "JPEG", quality=100) # Ép về JPEG chuẩn
            print("Đã re-save ảnh.")
            
    except Exception as e:
        print(f"Không thể sửa ảnh: {e}")



def split_dataset_yolov8(root_folder, max_items=299):
    """
    Chia nhỏ dataset YOLOv8 thành các phần nhỏ hơn < 300 ảnh.
    Mỗi phần chỉ chứa 1 loại subset (train hoặc val hoặc test).
    """
    
    # Cấu hình tên thư mục gốc và suffix
    base_name = os.path.basename(os.path.normpath(root_folder))
    
    # Các subset cần duyệt theo thứ tự
    subsets = ['train', 'val', 'test']
    
    # Biến đếm số thứ tự dataset đầu ra (1, 2, 3...)
    dataset_counter = 1
    
    # Định dạng ảnh hỗ trợ
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    print(f"--- Bắt đầu xử lý thư mục: {root_folder} ---")

    for subset in subsets:
        # Đường dẫn tới thư mục ảnh và nhãn nguồn
        src_img_dir = os.path.join(root_folder, 'images', subset)
        src_lbl_dir = os.path.join(root_folder, 'labels', subset)

        # Kiểm tra nếu thư mục tồn tại
        if not os.path.exists(src_img_dir):
            print(f"Bỏ qua '{subset}' (Không tìm thấy thư mục images).")
            continue

        # Lấy danh sách file ảnh và sắp xếp Alphabet
        all_files = os.listdir(src_img_dir)
        image_files = sorted([f for f in all_files if Path(f).suffix.lower() in img_exts])
        
        total_images = len(image_files)
        if total_images == 0:
            print(f"Subset '{subset}' trống, bỏ qua.")
            continue

        print(f"\nĐang xử lý subset: '{subset}' - Tổng số ảnh: {total_images}")

        # Tính toán số lượng chunk
        num_chunks = math.ceil(total_images / max_items)

        for i in range(num_chunks):
            # Xác định khoảng index cho chunk hiện tại
            start_idx = i * max_items
            end_idx = min((i + 1) * max_items, total_images)
            current_batch = image_files[start_idx:end_idx]

            # Tạo tên thư mục đích: TênGốc_1, TênGốc_2...
            output_folder_name = f"{base_name}_{dataset_counter}"
            output_path = os.path.join(os.path.dirname(root_folder), output_folder_name)
            
            # Tạo cấu trúc thư mục đích chuẩn YOLO
            # Ví dụ: output/images/train và output/labels/train
            dst_img_path = os.path.join(output_path, 'images', subset)
            dst_lbl_path = os.path.join(output_path, 'labels', subset)
            
            os.makedirs(dst_img_path, exist_ok=True)
            os.makedirs(dst_lbl_path, exist_ok=True)

            print(f"  -> Đang tạo: {output_folder_name} ({len(current_batch)} ảnh thuộc {subset})")

            # Copy file
            success_count = 0
            for img_file in current_batch:
                # 1. Copy ảnh
                src_img = os.path.join(src_img_dir, img_file)
                dst_img = os.path.join(dst_img_path, img_file)
                shutil.copy2(src_img, dst_img)

                # 2. Copy nhãn (nếu có)
                name_no_ext = Path(img_file).stem
                lbl_file = name_no_ext + ".txt"
                src_lbl = os.path.join(src_lbl_dir, lbl_file)
                
                if os.path.exists(src_lbl):
                    dst_lbl = os.path.join(dst_lbl_path, lbl_file)
                    shutil.copy2(src_lbl, dst_lbl)
                
                success_count += 1

            # Tăng biến đếm folder lên 1 sau khi hoàn thành 1 chunk
            dataset_counter += 1

    print("\n=== HOÀN TẤT ===")
    print(f"Đã chia thành {dataset_counter - 1} thư mục con.")




def check_images_readable(path):
    bad_files = []
    total = 0
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(exts):
                total += 1
                file_path = os.path.join(root, f)

                # 2. Thử Đọc Bytes (Mô phỏng bước load file thô)
                try:
                    with open(file_path, "rb") as f:
                        image_bytes = f.read()
                    
                    if len(image_bytes) == 0:
                        print("LỖI: File có kích thước 0 bytes (File rỗng).")
                        bad_files.append(file_path)
                        continue
                except Exception as e:
                    print(f"LỖI: Không thể đọc bytes từ file. Chi tiết: {e}")
                    bad_files.append(file_path)
                    continue

                # 3. Thử Decode bằng OpenCV 
                try:
                    # Chuyển bytes thành numpy array
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    
                    # Cố gắng decode
                    # cv2.IMREAD_UNCHANGED: Đọc nguyên gốc (bao gồm cả kênh alpha nếu có)
                    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                    
                    if img_cv2 is None:
                        print("KẾT QUẢ: OpenCV trả về 'None'.")
                        print("-> KẾT LUẬN: Header của ảnh bị hỏng hoặc định dạng không được OpenCV hỗ trợ.")
                        bad_files.append(file_path)
                        continue
                    else:
                        h, w = img_cv2.shape[:2]
                        print(f"KẾT QUẢ: OpenCV decode thành công! (Kích thước: {w}x{h})")
                        
                except Exception as e:
                    print(f"LỖI EXCEPTION OpenCV: {e}")
                    bad_files.append(file_path)
                    continue

                # 4. Thử Decode bằng PIL (Để đối chiếu)
                print("\n[Thử nghiệm 2: PIL/Pillow Decode]")
                try:
                    img_pil = Image.open(io.BytesIO(image_bytes))
                    img_pil.verify() # Kiểm tra tính toàn vẹn file
                    print(f"KẾT QUẢ: PIL xác nhận file hợp lệ (Format: {img_pil.format})")
                    print("-> Nếu PIL đọc được mà OpenCV không đọc được: Ảnh có định dạng lạ mà OpenCV chưa hỗ trợ.")
                except Exception as e:
                    print(f"KẾT QUẢ: PIL cũng không đọc được file.")
                    print(f"-> Chi tiết lỗi PIL: {e}")
                    print("-> KẾT LUẬN: File ảnh đã bị hỏng hoàn toàn (corrupt).")
                    bad_files.append(file_path)
                    continue



    print(f"Tổng số file ảnh đã kiểm tra: {total}")
    print(f"Số file lỗi decode: {len(bad_files)}")

    if bad_files:
        print("Danh sách file lỗi:")
        for f in bad_files:
            print(f)

    return bad_files


def check_txt_for_images(path, img_file_list):
    missing = []

    # Đọc danh sách file ảnh
    with open(img_file_list, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Lấy tên file ảnh (vd: 1.jpg)
        img_name = os.path.basename(line)

        # Đổi đuôi sang .txt
        base, _ = os.path.splitext(img_name)
        txt_name = base + ".txt"

        # Đường dẫn tuyệt đối tới file txt cần kiểm tra trong thư mục path
        txt_path = os.path.join(path, txt_name)

        if not os.path.exists(txt_path):
            missing.append(txt_name)

    # In kết quả
    if missing:
        print("Các file .txt bị thiếu:", len(missing))
        for m in missing:
            print(" -", m)
    else:
        print("Tất cả file .txt đều tồn tại.")

    return missing


def write_image_list(path, prefix, output_name):
    # Lấy danh sách file trong thư mục
    files = os.listdir(path)

    # Lọc những file là ảnh (nếu muốn), còn nếu không cần thì bỏ bước này
    # extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    # files = [f for f in files if f.lower().endswith(extensions)]

    # Sắp xếp tên file để dễ theo dõi
    files.sort()

    # Ghi ra file output
    with open(output_name, "w", encoding="utf-8") as f:
        for filename in files:
            line = f"{prefix}{filename}"
            f.write(line + "\n")

    print(f"Đã ghi {len(files)} dòng vào file '{output_name}'.")


def zip_result(results_dir: str):
    """
    Zip toàn bộ thư mục kết quả train YOLO .

    Args:
        results_dir (str): Thư mục kết quả train YOLO (results.save_dir)
    Returns:
        str: Đường dẫn file zip tạo ra
    """
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Thư mục kết quả train không tồn tại: {results_dir}")

    # Lấy tên thư mục
    parent_dir, folder_name = os.path.split(results_dir)
    
    # Tạo zip file với tên giống thư mục
    zip_base = os.path.join(parent_dir, folder_name)
    zip_path = shutil.make_archive(zip_base, 'zip', results_dir)
    print(f"Đã tạo file zip: {zip_path}")

    return zip_path


def copy_to_onedrive(zip_path: str, onedrive_dir: str):
    """
    Copy file zip vào OneDrive.

    Args:
        zip_path (str): File zip cần copy
        onedrive_dir (str): Thư mục OneDrive trên máy
    Returns:
        str: Đường dẫn file zip trong OneDrive
    """

    # Copy zip vào OneDrive
    dest_path = os.path.join(onedrive_dir, os.path.basename(zip_path))
    shutil.copy(zip_path, dest_path)
    print(f"Đã copy file zip vào: {dest_path}")

    return dest_path



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


def train(cfg_path='train_yolov9c_config.yml'):
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
    
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    project = cfg.get("project", "runs/train")
    name = cfg.get("name", "exp")

    expected_save_dir  = os.path.join(project, name)

    print("===== Training Started (Logging Enabled) =====")
    model = YOLO(cfg.get("model", "yolov9c.pt"))
    #cls_weight = np.array([   0.016953 ,   0.098252 ,   0.072657 ,  0.0042909 ,   0.087335  ,   0.72051])
    #cls_weight = cls_weight.astype(np.float32)
    #model.loss_fn = model.loss_fn.clone_with_cls_weight(cls_weight)
    results = model.train(cfg=cfg_path)

    actual_save_dir = results.save_dir
    print(f"Expected save_dir: {expected_save_dir}")
    print(f"Actual save_dir: {actual_save_dir}")
    print("===== Training Finished =====")

    zip_file_path = zip_result(results.save_dir)
    #onedrive_path = "/home/minhpt"  # thay bằng đường dẫn OneDrive thực tế
    #copy_to_onedrive(zip_file_path, onedrive_path)
    bucket = "aicycle-ai-advisor-train"          # bucket S3
    s3_folder = "a100_uploads"                         # folder trên S3
    aws_access_key_id = "AKIAxxxxxxxxxxxx"        # access key của bạn
    aws_secret_access_key = "xxxxxxxxxxxxxxxxx"   # secret key của bạn

    upload_to_s3(zip_file_path, bucket, s3_folder, aws_access_key_id, aws_secret_access_key)
    
    #os.system("sudo shutdown -h now")  # Nếu dùng Ubuntu

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


def issue4_fn_test(model_name = '../../result/yolov9c_baseline_best.pt', conf = 0.1, source_dir='../../data/tungbui/images',
                   results_dir='../../data/tungbui/outputs/segment', classes_arg = None):
    #model = YOLO('../../data/tungbui/models/car_damage_segmentation/CarDamage_20250926.pt') 
    #CLASS_NAME = 42
    #p, r, info = precision_recall_for_class(model, DATA_YAML_PATH, CLASS_NAME, conf=0.2, iou_thr=0.3)
    #print(f"[{CLASS_NAME}] @conf=0.25, IoU=0.5 -> Precision={p:.3f}, Recall={r:.3f} | {info}")
    #DATA_YAML_PATH = '../../data/tungbui/dataset1/data.yaml'

    #model = YOLO('../../data/tungbui/models/car_damage_segmentation/CarDamage_20250926.pt')
    #model = YOLO('../../data/tungbui/outputs/train/car_damage_seg_resume_20250926/weights/best.pt')
    #model = YOLO('../../data/tungbui/outputs/train/car_damage_seg_resume_20250926_stage2_unfreeze/weights/best.pt')

    model = YOLO(model_name)
    source_dir = source_dir
    results_dir = results_dir
    run_name   = img_name = os.path.basename(model_name)

    os.makedirs(results_dir, exist_ok=True)

    results = model.predict(
        source=source_dir,        # thư mục chứa ảnh cần xử lý
        save=True,                # lưu ảnh đã vẽ box/mask
        project=results_dir,      # thư mục gốc lưu kết quả
        name=run_name,            # tên phiên chạy -> ảnh sẽ nằm trong <project>/<name>
        conf=conf,                # ngưỡng confidence (tùy chỉnh theo nhu cầu)
        imgsz=1024,               # kích thước suy luận (tùy GPU/CPU)
        device=0 if torch.cuda.is_available() else 'cpu',  # ưu tiên GPU nếu có
        save_conf=True,           # lưu score lên ảnh
        save_txt=False,           # bật True nếu muốn lưu nhãn .txt (YOLO format)
        verbose=True,
        retina_masks=True,         # (segmentation) mask sắc nét hơn
        classes=classes_arg
    )

    out_dir = os.path.join(results_dir, run_name)
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




def model_evaluation(model_file='../../result/yolov9c_baseline_best.pt',
                          data_file='../../data/tungbui/dataset_tung_1k/data.yaml',
                          split_name='val',
                          imgsz=1024,       # resize khi predict
                          batch_size=4,   # batch size khi predict
                          device='0'      # 'cpu' hoặc '0', '0,1' nếu có GPU
                         ):
    """
    Phiên bản tối ưu đánh giá mô hình YOLOv9
    - Dùng model.predict toàn bộ ảnh cùng lúc (batch) để nhanh hơn
    - Tính mAP, Precision, Recall, F1, Confidence (min/max/mean/median)
    - AP từng class
    """

    # Load model
    model = YOLO(model_file)

    # 1. Đánh giá chuẩn YOLO (mAP, Precision, Recall, F1)
    metrics = model.val(data=data_file, split=split_name)
    map50 = metrics.box.map50
    map50_95 = metrics.box.map

    stats = {}
    for name, arr in zip(['Precision','Recall','F1'], [metrics.box.p, metrics.box.r, metrics.box.f1]):
        arr_np = np.array(arr)  # chuyển list sang numpy array
        stats[name] = {
            'min': float(np.min(arr_np)),
            'max': float(np.max(arr_np)),
            'mean': float(np.mean(arr_np)),
            'median': float(np.median(arr_np))
        }

    # 2. Lấy confidence từ toàn bộ ảnh cùng lúc
    with open(data_file) as f:
        data_yaml = yaml.safe_load(f)
    split_path = os.path.join(os.path.dirname(data_file), data_yaml[split_name])

    class_names_yaml = data_yaml['names']  # list tiếng Việt, đúng thứ tự index

    print(f"Lấy ảnh từ: {split_path}")
    # Lấy tất cả file ảnh trong folder
    img_files = glob.glob(os.path.join(split_path, '*.*'))  # jpg, png, ...

    conf_list = []
    for img in tqdm(img_files, desc="Predicting for confidence"):
        results = model.predict(img, verbose=False, conf=0.001)  # trả về list chứa Result objects
        for r in results:
            # r.boxes.conf → tensor confidence từng detection
            conf_list.extend(r.boxes.conf.cpu().numpy().flatten())

    conf_array = np.array(conf_list)
    stats['Confidence'] = {
        'min': float(np.min(conf_array)),
        'max': float(np.max(conf_array)),
        'mean': float(np.mean(conf_array)),
        'median': float(np.median(conf_array))
    }

    # 3. DataFrame thống kê đẹp
    df_stats = pd.DataFrame(stats).T

    # 4. AP từng class
    # metrics.box.ap: AP của các class xuất hiện
    # metrics.box.ap_class_index: index của class tương ứng với mỗi AP
    ap_values = np.zeros(len(model.names), dtype=float)  # mặc định 0
    for idx, ap in zip(metrics.box.ap_class_index, metrics.box.ap):
        ap_values[idx] = float(ap)

    class_ap_list = ap_values.tolist()

    # Kiểm tra chiều dài
    assert len(class_ap_list) == len(model.names), f"{len(class_ap_list)} != {len(model.names)}"

    df_ap = pd.DataFrame({
        'Class_index': range(len(class_names_yaml)),  # số thứ tự 0,1,2,...
        'Class_name': class_names_yaml,               # tên class từ yaml
        'AP': class_ap_list
    })

    # 5. In kết quả
    print("=== Tổng quan mAP ===")
    print(f"mAP50: {map50:.4f}")
    print(f"mAP50-95: {map50_95:.4f}\n")

    print("=== Thống kê chi tiết Precision / Recall / F1 / Confidence ===")
    print(df_stats)

    print("\n=== AP theo từng class ===")
    print(df_ap)

    model_name = os.path.basename(model_file)  # 'yolov9c_baseline_best.pt'
    output_file = model_name.replace('.pt', '_eval.txt')

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== Tổng quan mAP ===\n")
        f.write(f"mAP50: {map50:.4f}\n")
        f.write(f"mAP50-95: {map50_95:.4f}\n\n")

        f.write("=== Thống kê chi tiết Precision / Recall / F1 / Confidence ===\n")
        f.write(df_stats.to_string())
        f.write("\n\n")

        f.write("=== AP theo từng class ===\n")
        f.write(df_ap.to_string())
        f.write("\n")

    print(f"Kết quả đã được lưu vào {output_file}")

    return df_stats, df_ap



def compute_class_weight():
    # số lượng mẫu từng class
    samples = np.array([510, 88, 119, 2015, 99, 12])

    # tính weight ngược với tần suất
    cls_weight = 1 / (samples / samples.sum())
    cls_weight = cls_weight / cls_weight.sum()  # chuẩn hóa tổng =1

    print(cls_weight)
    # Output: [   0.016953    0.098252    0.072657   0.0042909    0.087335     0.72051]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument(
        "cfg_path",
        nargs="?",
        default="train_yolov9c_config.yml",
        help="Đường dẫn file config YAML"
    )
    args = parser.parse_args()

    train(cfg_path=args.cfg_path)    

    #train('train_yolov9c_config_baseline.yml')
    #train('train_yolov9c_config_clean_class.yml')
    #train('train_yolov9c_config_clean_class.yml')

    #model_evaluation()
    #issue4_fn_test(model_name='../../result/yolov9c_baseline_best.pt', conf=0.005, source_dir='../../data/tungbui/images',
    #               results_dir='../../data/tungbui/outputs/segment', classes_arg = [39])
    #issue4_fn_fix()
    #compute_class_weight()
    
    #split_dataset_yolov8(root_folder='../../data/tungbui/dataset_tung_1k_clean_class_cvat_all_copy_2_1_2_2_1_2_1_2', max_items=1)
    #write_image_list(path='../../data/tungbui/dataset_tung_1k_clean_class_cvat_all_copy_2_1_2_2_1_2_1_2/images/train', prefix='images/train/', output_name='train.txt')

    #check_txt_for_images(path='../../data/tungbui/dataset_tung_1k_clean_class_cvat/labels/test', img_file_list='../../data/tungbui/dataset_tung_1k_clean_class_cvat/test.txt')
    #check_images_readable(path='../../data/tungbui/dataset_tung_1k_clean_class_cvat_all_copy_2_1_2_2_1_2_1_2_2/images/train')

    #check_image_deeply('../../data/tungbui/dataset_tung_1k_clean_class_cvat_all_copy_2_1_2_2_1_2_1_2_2/images/train/91027d6cd4d64b3f9cca9c6376deecb1262dfa86e52b4c189a47a501b8a854ec1722818012.jpg')
    #convert_to_rgb_safe('../../data/tungbui/dataset_tung_1k_clean_class_cvat_all_copy_failed/images/train/a.jpg')
    #print_comparison('../../data/tungbui/dataset_tung_1k_clean_class_cvat_all_copy_2_1_2_2_1_2_1_2_2/images/train/91027d6cd4d64b3f9cca9c6376deecb1262dfa86e52b4c189a47a501b8a854ec1722818012.jpg', '../../data/tungbui/dataset_tung_1k_clean_class_cvat_all_copy_failed/images/train/a.jpg')
    #print_comparison('../../data/tungbui/dataset_tung_1k_clean_class_cvat_all_copy_1/images/train/0a91ce8b-03dd-4c38-96b3-20fb57d3c03c.jpg', '../../data/tungbui/dataset_tung_1k_clean_class_cvat_all_copy_failed/images/train/a.jpg')

    #run_interactive_augmentation(
    #    input_img_dir='../../data/tungbui/dataset_tung_1k_clean_class_cvat_all_copy_failed/images/train/',
    #    input_label_dir='../../data/tungbui/dataset_tung_1k_clean_class_cvat_all_copy_failed/labels/train/',
    #    output_img_dir='../../data/tungbui/dataset_tung_1k_clean_class_cvat_all_copy_failed/output_aug/images/',
    #    output_label_dir='../../data/tungbui/dataset_tung_1k_clean_class_cvat_all_copy_failed/output_aug/labels/',
    #    augment_times=3,     # Số ảnh tạo ra từ 1 ảnh gốc
    #    image_ext="*.jpg"    # Định dạng ảnh
    #)

    #model_evaluation(model_file='../../result/yolov9c_baseline_best.pt', data_file='../../data/tungbui/dataset_tung_1k/data.yaml', split_name='val')
    #model_evaluation(model_file='../../result/yolov9e_baseline_best.pt', data_file='../../data/tungbui/dataset_tung_1k/data.yaml', split_name='val')
    #model_evaluation(model_file='../../result/yolo12m_baseline_best.pt', data_file='../../data/tungbui/dataset_tung_1k/data.yaml', split_name='val')    
    #model_evaluation(model_file='../../result/yolov9c_clean_class_best.pt', data_file='../../data/tungbui/dataset_tung_1k_clean_class/data.yaml', split_name='val')
    #model_evaluation(model_file='../../result/yolov9c_baseline_best.pt', data_file='../../data/tungbui/dataset_tung_1k/data.yaml', split_name='val')
    
    #CLASS_NAME = 42   
    #model = YOLO('runs/segment/my_experiment3/weights/best.pt') 
    #p, r, info = precision_recall_for_class(model, DATA_YAML_PATH, CLASS_NAME, conf=0.2, iou_thr=0.3)
    #print(f"[{CLASS_NAME}] @conf=0.25, IoU=0.5 -> Precision={p:.3f}, Recall={r:.3f} | {info}")

    # Đánh giá mô hình sau huấn luyện
    #best_model = YOLO('runs/segment/my_experiment/weights/best.pt')
    #best_model.val(split='test')

    #import torchvision
    #print(hasattr(torchvision.ops, "nms"))
