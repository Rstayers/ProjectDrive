import cv2
import numpy as np

class FramePreprocessor:
    def __init__(self, aspect=(640, 360), denoise=True, remove_hood=True,
                 hood_crop_ratio=0.15):
        self.resize_shape = aspect
        self.denoise = denoise
        self.hood_crop_ratio = hood_crop_ratio
        self.remove_hood = remove_hood
        self.results = {}

    def preprocess_frame(self, img_path):
        """
        Reads, resizes, and denoises image
        """
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        # crop
        if self.remove_hood:
            img = self.crop_car_hood(img)
        # resize
        img = cv2.resize(img, self.resize_shape)
        # denoise
        if self.denoise:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        # grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.results =  {
            'original': img,
            'gray': gray
        }
    def crop_car_hood(self, image):
        """
        Crops out the bottom portion of the image that is assumed to be the car hood.
        """
        h, w = image.shape[:2]
        crop_y = int(h * (1 - self.hood_crop_ratio))
        cropped = image[0:crop_y, :]
        return cropped



def visualize_debug(preprocessor, img_path):
    """
    debug visualization of preprocessing output
    """
    result = preprocessor.preprocess_frame(img_path)

    # Prepare images for side-by-side visualization
    original = preprocessor.results['original']
    gray_bgr = cv2.cvtColor(preprocessor.results['gray'], cv2.COLOR_GRAY2BGR)

    combined = np.hstack((original, gray_bgr))

    cv2.imshow("Debug: Original | Gray", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    preprocessor = FramePreprocessor()
    test_img_path = "/BDD100K/images/train/0a0a0b1a-7c39d841.jpg"
    visualize_debug(preprocessor, test_img_path)