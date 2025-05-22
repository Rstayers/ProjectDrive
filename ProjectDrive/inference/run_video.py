import cv2
import os
from inference.lane_inference import LaneInference
from inference.yolo_inference import YOLOInference

def run_combined_inference(video_path, output_path,
                           lane_model_path,
                           yolo_model_path):
    # Initialize models
    lane_detector = LaneInference(model_path=lane_model_path)
    yolo_detector = YOLOInference(model_path=yolo_model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Lane detection overlay
        lane_mask = lane_detector.predict(frame)
        lane_overlay = lane_detector.visualize_overlay(frame, lane_mask)

        # YOLO detection overlay
        combined = yolo_detector.predict_and_visualize(lane_overlay)

        out.write(combined)
        cv2.imshow("Preview", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done! Output saved to: {output_path}")


if __name__ == "__main__":
    input_video = "..\Chosen Test Samples\Highway.mp4"
    output_video = "../videos/final_lane_yolo_overlay_Unet.mp4"
    run_combined_inference(input_video,lane_model_path='../models/lane_net_best.pth', yolo_model_path='../models/best_YOLO.pt', output_path=output_video)

