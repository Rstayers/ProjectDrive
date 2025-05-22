import os
from glob import glob
from PIL import Image
import numpy as np
import json

from feature_extractor import FeatureExtractor
from preprocessing import FramePreprocessor
from tqdm import tqdm
from feature_extractor import FeatureExtractor
from preprocessing import FramePreprocessor
import pickle

def extract_feature_vector_lane(feature_dict):
    """
    Concat HOG and Hough features.
    """
    hog = feature_dict['hog_features']
    hough_lines = feature_dict['hough_lines']
    hough_feature = [0] if hough_lines is None else [len(hough_lines)]
    return np.concatenate((hog, hough_feature))

def convert_to_yolo(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2 / img_w
    y_center = (y1 + y2) / 2 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return [x_center, y_center, width, height]
def serialize_keypoints(keypoints):
    """
    Converts cv2.KeyPoint objects to a serializable format.
    """
    return [(
        kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id
    ) for kp in keypoints]

def process_image(img_path):
    """
    Runs the current processing pipeline for feature extraction
    :param img_path: path to image
    :return: dictionary of feature vectors
    """
    # preprocess
    preprocessor = FramePreprocessor(remove_hood=False)
    preprocessor.preprocess_frame(img_path)

    # feature extraction
    extractor = FeatureExtractor(preprocessor=preprocessor)
    extractor.extract_features()

    # convert non-serializable items
    features = extractor.results.copy()
    features['keypoints'] = serialize_keypoints(features['keypoints'])

    return features


def process_dataset(dataset_path, output_path):
    """
    Process all images in the dataset directory and save extracted features.
    """
    os.makedirs(output_path, exist_ok=True)

    for root, dirs, files in os.walk(dataset_path):
        for file in tqdm(files):
            if file.lower().endswith('.jpg'):
                img_path = os.path.join(root, file)
                rel_path = os.path.relpath(img_path, dataset_path)
                out_path = os.path.join(output_path, rel_path).replace(".jpg", "")
                out_dir = os.path.dirname(out_path)
                os.makedirs(out_dir, exist_ok=True)

                try:
                    features = process_image(img_path)
                    with open(out_path + ".pkl", 'wb') as f:
                        pickle.dump(features, f)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

def process_labels_to_YOLO(json_dir, image_dir, category_map, output_dir):

    for json_path in tqdm(glob(os.path.join(json_dir, "*.json"))):
        with open(json_path, "r") as f:
            data = json.load(f)

        frame = data["frames"][0]
        image_name = os.path.splitext(os.path.basename(json_path))[0] + ".jpg"
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"[!] Image not found: {image_path}")
            continue

        width, height = Image.open(image_path).size
        label_lines = []

        for obj in frame["objects"]:
            category = obj.get("category")
            if category not in category_map or "box2d" not in obj:
                continue

            class_id = category_map[category]
            box = obj["box2d"]
            yolo_box = convert_to_yolo([box["x1"], box["y1"], box["x2"], box["y2"]], width, height)
            label_lines.append(f"{class_id} {' '.join(f'{x:.6f}' for x in yolo_box)}")

        # Save label file
        if label_lines:
            label_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".txt")
            with open(label_path, "w") as out_f:
                out_f.write("\n".join(label_lines))

def extract_feature_vector_car(feature_dict, max_sift_features=100):
    """
    Combines HOG and SIFT descriptors into a single feature vector.

    Returns: feature vector
    """
    hog = feature_dict.get('hog_features', None)
    sift = feature_dict.get('descriptors', None)

    if hog is None:
        raise ValueError("Missing HOG features")

    if sift is None:
        sift = np.zeros((max_sift_features, 128))
    else:
        if sift.shape[0] > max_sift_features:
            sift = sift[:max_sift_features]
        elif sift.shape[0] < max_sift_features:
            padding = np.zeros((max_sift_features - sift.shape[0], 128))
            sift = np.vstack((sift, padding))

    sift_flat = sift.flatten()

    # Final feature vector: [HOG | SIFT]
    return np.concatenate((hog, sift_flat))
