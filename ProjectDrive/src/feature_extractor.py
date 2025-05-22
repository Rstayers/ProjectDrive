import cv2
import numpy as np
from preprocessing import FramePreprocessor
from skimage import feature, exposure
class FeatureExtractor:
    def __init__(self, preprocessor, canny_thresholds=(100, 200), hough_params=None):
        self.img = preprocessor.results['gray']
        self.feature_detector = cv2.SIFT_create()

        if hough_params is None:
            self.hough_params = {
                'rho': 6,
                'theta': np.pi / 60,
                'threshold': 160,
                'minLineLength': 40,
                'maxLineGap': 25
            }
        else:
            self.hough_params = hough_params
        self.canny_thresholds = canny_thresholds
        self.results = {}

    def calculate_hog(self):
        fd, hog_image = feature.hog(self.img, orientations=8, pixels_per_cell=(16, 16),
                                    cells_per_block=(1, 1), visualize=True)

        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        return fd, hog_image_rescaled

    def apply_hough_transform(self, edges, image):
        """
        Applies the Hough transform to detect lines and draws them on the image.
        """
        lines = cv2.HoughLinesP(edges,
                                rho=self.hough_params['rho'],
                                theta=self.hough_params['theta'],
                                threshold=self.hough_params['threshold'],
                                minLineLength=self.hough_params['minLineLength'],
                                maxLineGap=self.hough_params['maxLineGap'])
        line_image = np.copy(image)
        if len(line_image.shape) == 2:
            line_image = cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return line_image, lines

    def extract_features(self):
        """
        Extracts features using Canny edge detection, Hough transform, SIFT, and HOG.
        """
        # Filter out horizontal lines for lane detection
        sobel_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        direction = np.arctan2(np.abs(sobel_y), np.abs(sobel_x))
        mask = (direction > np.pi / 6) & (direction < 2 * np.pi / 3)

        # Canny edge detection
        edges = cv2.Canny(self.img, self.canny_thresholds[0], self.canny_thresholds[1])
        edges_filtered = cv2.bitwise_and(edges, edges, mask=mask.astype(np.uint8))

        # Apply Hough transform
        hough_image, hough_lines = self.apply_hough_transform(edges_filtered, self.img)

        # Feature detection: SIFT
        keypoints, descriptors = self.feature_detector.detectAndCompute(self.img, None)

        # HOG feature extraction
        hog_features, hog_image = self.calculate_hog()


        self.results =  {
            'edges': edges,
            'hough_image': hough_image,
            'hough_lines': hough_lines,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'hog_features': hog_features,
            'hog_image': hog_image,
        }

def visualize_debug(feature_extractor):
    """
    Debug visualization of extractor output
    """
    feature_extractor.extract_features()

    # Draw keypoints
    keypoints_img = cv2.drawKeypoints(feature_extractor.img, feature_extractor.results['keypoints'], None,
                                      color=(0, 255, 0),
                                      flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

    edges_bgr = cv2.cvtColor(feature_extractor.results['edges'], cv2.COLOR_GRAY2BGR)
    hough_img = feature_extractor.results['hough_image'] if feature_extractor.results['hough_image'] is not None else np.zeros_like(feature_extractor.img)
    hog_img = feature_extractor.results['hog_image'] if feature_extractor.results['hog_image'] is not None else np.zeros_like(feature_extractor.img)

    # Normalize HOG image
    if len(hog_img.shape) == 2:
        hog_img = cv2.normalize(hog_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hog_img = cv2.cvtColor(hog_img, cv2.COLOR_GRAY2BGR)

    # resize
    height, width = feature_extractor.img.shape[:2]
    edges_bgr = cv2.resize(edges_bgr, (width, height))
    hough_img = cv2.resize(hough_img, (width, height))
    hog_img = cv2.resize(hog_img, (width, height))
    keypoints_img = cv2.resize(keypoints_img, (width, height))

    top_row = np.hstack((edges_bgr, hough_img))
    bottom_row = np.hstack((hog_img, keypoints_img))
    combined = np.vstack((top_row, bottom_row))

    # Display the combined image
    cv2.imshow("Debug: Edges | Hough | HOG | Keypoints", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_img_path = r"C:\Users\soupy\PycharmProjects\CV_SP25\BDD100K/images/train/0a0a0b1a-7c39d841.jpg"
    preprocessor = FramePreprocessor()
    preprocessor.preprocess_frame(test_img_path)
    extractor = FeatureExtractor(preprocessor=preprocessor)
    visualize_debug(extractor)
