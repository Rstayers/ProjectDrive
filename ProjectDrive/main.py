import os
import cv2
from inference.lane_inference import LaneInference
from inference.yolo_inference import YOLOInference
from inference.run_video import run_combined_inference
import argparse

def batch_image_inference(image_dir, lane_model_path, yolo_model_path):
    lane_model = LaneInference(lane_model_path)
    yolo_model = YOLOInference(yolo_model_path)

    stitched_images = []

    for fname in sorted(os.listdir(image_dir)):
        if not fname.endswith(".jpg"):
            continue
        img_path = os.path.join(image_dir, fname)
        image = cv2.imread(img_path)

        # Lane detection
        lane_mask = lane_model.predict(image)
        lane_overlay = lane_model.visualize_overlay(image, lane_mask)

        # YOLO detection
        combined = yolo_model.predict_and_visualize(lane_overlay)

        # Resize
        combined_resized = cv2.resize(combined, (640, 360))
        stitched_images.append(combined_resized)

    # Stitch all together horizontally in batches of 3
    row_images = []
    for i in range(0, len(stitched_images), 3):
        row = stitched_images[i:i+3]
        if len(row) < 3:
            pad = [cv2.copyMakeBorder(img, 0, 0, 0, 640 * (3 - len(row)), cv2.BORDER_CONSTANT, value=(0, 0, 0)) for img in row]
            row_images.append(cv2.hconcat(pad))
        else:
            row_images.append(cv2.hconcat(row))

    final_img = cv2.vconcat(row_images)

    cv2.imshow("Batch Combined Lane + YOLO Inference", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def batch_video_inference(videos_dir, lane_model_path, yolo_model_path, output_dir):
    for fname in os.listdir(videos_dir):
        if not fname.endswith(".mp4"):
            continue
        input_path = os.path.join(videos_dir, fname)
        output_path = os.path.join(output_dir, f"processed_{fname}")
        run_combined_inference(
            video_path=input_path,
            output_path=output_path,
            lane_model_path=lane_model_path,
            yolo_model_path=yolo_model_path
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run image or video inference")
    parser.add_argument("--mode", choices=["image", "video"], default="image",
                        help="Choose to run 'image' inference on Chosen Test Samples or 'video' processing on videos/")
    args = parser.parse_args()

    lane_model_path = "models/lane_net_best.pth"
    yolo_model_path = "models/best_YOLO.pt"
    dir  = "Chosen Test Samples"
    if args.mode == "image":
        batch_image_inference(dir, lane_model_path, yolo_model_path)

    elif args.mode == "video":
        output_video_dir = "videos/processed"
        batch_video_inference(dir, lane_model_path, yolo_model_path, output_video_dir)