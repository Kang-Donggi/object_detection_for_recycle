import os
import pandas as pd
import numpy as np
import cv2
import shutil
import yaml
import warnings
warnings.filterwarnings("ignore")

from ultralytics import RTDETR
from glob import glob
from tqdm import tqdm
from IPython.display import clear_output
from sklearn.model_selection import train_test_split

SEED = 42
BATCH_SIZE = 8
MODEL = "train_detr"


model = RTDETR("/home/kangdg22/meta_Assignment/boostcamp/detection/rt-detr_code/rtdetr-x.pt")
results = model.train(
    data="/home/kangdg22/meta_Assignment/boostcamp/detection/yolo_dataset/split/trash_yolo.yaml",
    imgsz=512,
    epochs=200,
    batch=BATCH_SIZE,
    patience=15,
    workers=16,
    device=0,
    exist_ok=True,
    project=f"{MODEL}",
    name="train",
    seed=SEED,
    pretrained=True,
    resume=False,
    optimizer="SGD",
    lr0=1e-3,
    val=True,
    cache=True,
    # weight_decay=0.0005,
    # fliplr=0.2,
    # mixup = 0.3   # use mix-up Augmentation
    )

def get_test_image_paths(test_image_paths):
    for i in range(0, len(test_image_paths), BATCH_SIZE):
        yield test_image_paths[i:i+BATCH_SIZE]

model = RTDETR(f"{MODEL}/train/weights/best.pt")
# model = YOLO(f"{MODEL}/train/weights/last.pt")
test_image_paths = sorted(glob("/home/kangdg22/meta_Assignment/boostcamp/detection/yolo_dataset/split/test/*.jpg"))

for i, image in tqdm(enumerate(get_test_image_paths(test_image_paths)), total=int(len(test_image_paths)/BATCH_SIZE)):
    model.predict(image, imgsz=(1024, 1024), iou=0.5, conf=0.5, save_conf=True, save=False, save_txt=True, project=f"{MODEL}", name="predict",
                  exist_ok=True, device=0, augment=True, verbose=False)
    if i % 5 == 0:
        clear_output(wait=True)
