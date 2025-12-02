
import numpy as np
from ultralytics import YOLO
from ultralytics import RTDETR
from ultralytics.utils import LOGGER
import torch

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
import argparse
import boto3
from train_and_test_yolo import upload_to_s3


def clean_yolo_log(input_path, output_path):
    """
    Làm sạch file log YOLO: xóa ANSI escape codes, ký tự màu, ký tự ghi-đè dòng.
    Giữ nguyên format xuống dòng như hiển thị trong terminal.
    """
    # Đọc nội dung file log gốc
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Xóa ANSI escape codes (màu sắc, di chuyển con trỏ, v.v.)
    cleaned = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', content)

    # Chuyển carriage return (\r) thành newline thực sự
    cleaned = cleaned.replace('\r', '\n')

    # Gom nhiều dòng trống liên tiếp thành 1 dòng
    cleaned = re.sub(r'\n+', '\n', cleaned)

    # Ghi file sạch ra output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    print(f"Đã làm sạch log và lưu tại: {output_path}")



def train_terminal_log(cfg_path='train_yolov9c_config.yml', file_log_name='terminal_out.txt'):
        
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    project = cfg.get("project", "runs/train")
    name = cfg.get("name", "exp")

    file_log_cleaned_name = name + "_" + file_log_name
    clean_yolo_log(input_path = file_log_name, output_path = file_log_cleaned_name)

    zip_base = os.path.splitext(file_log_cleaned_name)[0]
    zip_path = shutil.make_archive(zip_base, 'zip', root_dir='.', base_dir=file_log_cleaned_name)
    print(f"Đã tạo file zip: {zip_path}")
    #onedrive_path = "/home/minhpt"  # thay bằng đường dẫn OneDrive thực tế
    #copy_to_onedrive(zip_file_path, onedrive_path)

    bucket = "aicycle-ai-advisor-train"          # bucket S3
    s3_folder = "a100_uploads"                         # folder trên S3
    aws_access_key_id = "AKIAxxxxxxxxxxxx"        # access key của bạn
    aws_secret_access_key = "xxxxxxxxxxxxxxxxx"   # secret key của bạn
    upload_to_s3(zip_path, bucket, s3_folder, aws_access_key_id, aws_secret_access_key)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train YOLO log cleaner")

    parser.add_argument(
        "cfg_path",
        nargs="?",
        default="train_yolov9c_config.yml",
        help="Đường dẫn đến file YAML config"
    )

    args = parser.parse_args()

    train_terminal_log(cfg_path=args.cfg_path)
