import os
import pickle
import json
from glob import glob
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from joblib import dump
import numpy as np
from src.utils_custom import extract_feature_vector_lane
from tqdm import tqdm

# Directories
FEATURE_DIR = "../../BDD100K/features_vectors/train"
LABEL_DIR = "../../BDD100K/labels/train"
MODEL_OUT = "svm_lane_classifier_pca.joblib"
N_COMPONENTS = 0.95


def has_lane_polyline(json_path):
    """
    Returns True if any object in the JSON file has a 'lane/' category and poly2d.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    for frame in data.get('frames', []):
        for obj in frame.get('objects', []):
            if obj.get('category', '').startswith('lane/') and obj.get('poly2d'):
                return True
    return False

def load_features_and_labels():
    features = []
    labels = []

    for pkl_path in tqdm(glob(os.path.join(FEATURE_DIR, '**', '*.pkl'), recursive=True),  desc="Loading features"):
        base_name = os.path.basename(pkl_path).replace('.pkl', '')

        json_path = os.path.join(LABEL_DIR, base_name + '.json')
        if not os.path.exists(json_path):
            continue

        with open(pkl_path, 'rb') as f:
            feature_dict = pickle.load(f)
            try:
                feature_vector = extract_feature_vector_lane(feature_dict)
                features.append(feature_vector)
                labels.append(1 if has_lane_polyline(json_path) else 0)

            except Exception as e:
                print(f"Error with {pkl_path}: {e}")

    return np.array(features), np.array(labels)

def main():
    print("Loading data...")
    X, y = load_features_and_labels()
    print(f"Loaded {len(X)} samples with shape {X.shape}.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline: PCA + SVM
    pipeline = Pipeline([
        ('pca', PCA(n_components=N_COMPONENTS)),
        ('svm', svm.SVC(kernel='rbf', probability=True))
    ])

    print("Training classifier...")
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["no_lane", "lane"]))

    print(f"Saving model with PCA to {MODEL_OUT}")
    dump(pipeline, MODEL_OUT)

if __name__ == "__main__":
    main()
