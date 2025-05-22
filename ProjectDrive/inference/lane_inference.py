import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from src.model import UNet


class LaneInference:
    def __init__(self, model_path, input_size=(256, 256), device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = UNet(in_channels=1, out_channels=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(input_size),
            transforms.ToTensor()
        ])

    def predict(self, image: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor).squeeze().cpu().numpy()

        mask = cv2.resize(output, (image.shape[1], image.shape[0]))
        return mask

    def visualize_overlay(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        red_overlay = np.zeros_like(image)
        red_overlay[:, :, 1] = 255  # Red

        mask_3d = np.stack([mask] * 3, axis=-1)
        red_mask = (mask_3d * red_overlay).astype(np.uint8)

        return cv2.addWeighted(image, 1.0, red_mask, alpha, 0)


if __name__ == "__main__":
    img_path = "../../BDD100K\images\\test\\cd828461-8956189e.jpg"
    model_path = "../models\lane_net_best.pth"

    img = cv2.imread(img_path)
    lane_model = LaneInference(model_path)
    mask = lane_model.predict(img)
    overlay = lane_model.visualize_overlay(img, mask)

    cv2.imshow("Lane Detection (LaneCNN)", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
