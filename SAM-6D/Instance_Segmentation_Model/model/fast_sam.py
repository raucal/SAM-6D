"""FastSAM-x proposals with Ultralytics 8.0.135 and coordinates in the source image."""
from pathlib import Path
import torch
import torch.nn.functional as F
from ultralytics import YOLO

class FastSAM:
    def __init__(self, checkpoint_path, config, segmentor_width_size=640, device=None):
        self.model = YOLO(str(Path(checkpoint_path)))
        self.current_device = torch.device(device or 'cpu')
        self.segmentor_width_size = segmentor_width_size
        self.iou_threshold = float(config.iou_threshold)
        self.conf_threshold = float(config.conf_threshold)
        self.max_det = int(config.max_det)

    @torch.no_grad()
    def generate_masks(self, image):
        h, w = image.shape[:2]
        result = self.model.predict(source=image[:, :, ::-1].copy(), device=str(self.current_device),
            imgsz=self.segmentor_width_size, conf=self.conf_threshold, iou=self.iou_threshold,
            max_det=self.max_det, retina_masks=True, verbose=False)[0]
        if result.masks is None:
            return {'masks': torch.empty((0,h,w),device=self.current_device),
                    'boxes': torch.empty((0,4),device=self.current_device)}
        masks = result.masks.data.to(self.current_device)
        if tuple(masks.shape[-2:]) != (h,w):
            masks = F.interpolate(masks[:,None].float(),size=(h,w),mode='nearest')[:,0]
        return {'masks': masks, 'boxes': result.boxes.xyxy.to(self.current_device)}
