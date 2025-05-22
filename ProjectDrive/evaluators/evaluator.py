import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from src.model import UNet
from src.train import BDD100Data

# ----- CONFIG -----
LANE_MODEL_PATH = "../models/lane_net_best.pth"
DATA_DIR = "../../BDD100K"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_WORKERS = 4
YOLO_MODEL_PATH = "../models/best_YOLO.pt"


def load_lane_model():
    print(f"Loading model on {DEVICE}")
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    checkpoint = torch.load(LANE_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def run_yolo_eval(split):
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL_PATH)

    metrics = model.val(
        data="../../data.yaml",
        split=split,
        imgsz=640,
        batch=8,
        verbose=False
    )

    return {
        "mAP@0.5": metrics.box.map50,
        "mAP@0.5:0.95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr
    }

def evaluate_lane_model(model, dataloader, tag="val"):
    model.eval()
    total_iou, total_acc, count = 0.0, 0.0, 0

    for images, masks in tqdm(dataloader, desc=f"Evaluating {tag}"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        with torch.no_grad():
            preds = model(images)
            preds_bin = (preds > 0.5).float()

        intersection = (preds_bin * masks).sum(dim=(1, 2, 3))
        union = ((preds_bin + masks) >= 1).float().sum(dim=(1, 2, 3))
        iou = (intersection / union.clamp(min=1e-6)).mean().item()
        acc = (preds_bin == masks).float().mean().item()

        total_iou += iou
        total_acc += acc
        count += 1

    return total_iou / count, total_acc / count

if __name__ == "__main__":
    model = load_lane_model()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_data = BDD100Data(
        images_dir=f"{DATA_DIR}/images/test",
        masks_dir=f"{DATA_DIR}/labels/lane/test",
        transform=transform
    )


    train_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)


    print("\nEvaluating Lane Detection Model...")
    train_iou, train_acc = evaluate_lane_model(model, train_loader, tag="train")

    print(f"Lane Test IOU: {train_iou:.4f}, Acc: {train_acc:.4f}")

    print("\n Evaluating YOLOv5 Model...")
    yolo_train = run_yolo_eval("test")
    print(f" YOLO Test → mAP@0.5: {yolo_train['mAP@0.5']:.4f}, P: {yolo_train['precision']:.4f}, R: {yolo_train['recall']:.4f}")
