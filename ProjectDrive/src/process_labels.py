import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def draw_lane_polylines(poly2d_list, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    for poly in poly2d_list:
        points = np.array([[int(p[0]), int(p[1])] for p in poly], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(mask, [points], isClosed=False, color=255, thickness=3)
    return mask

if __name__ == '__main__':
    json_dir = '../../BDD100K/labels/test'
    output_mask_dir = '../../BDD100K/labels/lane/test'
    images_dir = '../../BDD100K/images/test'
    image_shape = (720, 1280)

    os.makedirs(output_mask_dir, exist_ok=True)

    for fname in tqdm(os.listdir(json_dir), desc="Processing lane labels"):
        if not fname.endswith('.json'):
            continue

        base_name = os.path.splitext(fname)[0]
        image_path = os.path.join(images_dir, base_name + '.jpg')


        if not os.path.exists(image_path):
            continue

        out_name = base_name + '.png'
        out_path = os.path.join(output_mask_dir, out_name)

        if os.path.exists(out_path):
            continue

        json_path = os.path.join(json_dir, fname)
        with open(json_path, 'r') as f:
            data = json.load(f)

        poly2d_list = []
        for frame in data.get('frames', []):
            for obj in frame.get('objects', []):
                if obj.get('category', '').startswith('lane/'):
                    poly = obj.get('poly2d', [])
                    if poly:
                        poly2d_list.append(poly)

        mask = draw_lane_polylines(poly2d_list, image_shape)
        cv2.imwrite(out_path, mask)
