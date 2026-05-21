from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from PIL import Image as PILImage
from abc import ABC, abstractmethod
from ultralytics import YOLO, RTDETR
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

COCO_TO_VISDRONE = {0: 1, 1: 3, 2: 4, 3: 10, 5: 9, 7: 6}

VISDRONE_CLASSES = {
    1: "pedestrian", 2: "people", 3: "bicycle",
    4: "car", 5: "van", 6: "truck", 7: "tricycle",
    8: "awning-tricycle", 9: "bus", 10: "motor"
}

TEXT_TO_VISDRONE = {
    "pedestrian": 1, "people": 2, "bicycle": 3,
    "car": 4, "van": 5, "truck": 6, "tricycle": 7,
    "awning tricycle": 8, "bus": 9, "motor": 10
}

@dataclass
class Detection:
    box: np.ndarray
    class_id: int
    class_name: str
    confidence: float
    
class BaseDetector(ABC):

    name: str
    
    @abstractmethod
    def predict(self, image:np.ndarray):
        pass
    
    @abstractmethod
    def load(self):
        pass
    
    
class YOLODetector(BaseDetector):
    name= "yolo"
    
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        
        self.confidence = config.get('models',{}).get('yolo_model',{}).get('confidence_threshold',0.5)
        self.weights_path = config.get('models',{}).get('yolo_model',{}).get('weights_path',"yolov8m.pt")
        self.device = config.get('models',{}).get('device', 'cpu')
        
    def load(self):
        self.model = YOLO(self.weights_path)
        
    def predict(self, image:np.ndarray) -> list[Detection]:
        results = self.model(image, conf=self.confidence, verbose= False)
        result = results[0]
        
        detections = []
        
        for box in result.boxes:
            norm_xyxy = box.xyxyn
            coco_cls = int(box.cls)
            conf = float(box.conf)
            
            if coco_cls in COCO_TO_VISDRONE:
                visdrone_cls = COCO_TO_VISDRONE[coco_cls]
                visdrone_label = VISDRONE_CLASSES[visdrone_cls]
                np_box = norm_xyxy[0].cpu().numpy()
                detections.append(Detection(box= np_box, class_id= visdrone_cls, class_name= visdrone_label, confidence= conf))
           
        return detections
    
class RTDETRDetector(BaseDetector):
    name = "rtdetr"
    
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        
        self.confidence = config.get('models',{}).get('rt_detr_model',{}).get('confidence_threshold',0.5)
        self.weights_path = config.get('models',{}).get('rt_detr_model',{}).get('weights_path',"rtdetr-l.pt")
        self.device = config.get('models',{}).get('device', 'cpu')
        
    def load(self):
        self.model = RTDETR(self.weights_path)
        
    def predict(self, image:np.ndarray) -> list[Detection]:
        results = self.model(image, conf=self.confidence, verbose= False)
        result = results[0]
        
        detections = []
        
        for box in result.boxes:
            norm_xyxy = box.xyxyn
            coco_cls = int(box.cls)
            conf = float(box.conf)
            
            if coco_cls in COCO_TO_VISDRONE:
                visdrone_cls = COCO_TO_VISDRONE[coco_cls]
                visdrone_label = VISDRONE_CLASSES[visdrone_cls]
                np_box = norm_xyxy[0].cpu().numpy()
                detections.append(Detection(box= np_box, class_id= visdrone_cls, class_name= visdrone_label, confidence= conf))
           
        return detections
    

class GroundingDINODetector(BaseDetector):
    name= "grounding_dino"
    
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        
        self.model_id = config.get('models',{}).get('grounding_dino_model',{}).get('model_id', "IDEA-Research/grounding-dino-tiny")
        
        self.text_prompt = config.get('models',{}).get('grounding_dino_model',{}).get('text_prompt', "")
        
        self.confidence = config.get('models',{}).get('grounding_dino_model',{}).get('confidence_threshold', 0.5)
        
        self.device = config.get('models',{}).get('device', 'cpu')
    
    def load(self):
        # Loading the processor
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
    
    def predict(self, image: np.ndarray):
        # Loading the image to PIL
        H, W = image.shape[:2]
        image = PILImage.fromarray(image)
        
        # Processing the image using the processor and the text prompt
        inputs = self.processor(images=image, text= self.text_prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        results = self.processor.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold= self.confidence, text_threshold=0.3, target_sizes =[(H,W)])
        
        result = results[0]
        
        detections = []
        for box, label, score in zip(result['boxes'], result['labels'], result['scores']):
                      
            visdrone_id = TEXT_TO_VISDRONE.get(label.lower().strip())
            if visdrone_id is None:
                continue
            
            x1,y1,x2,y2 = box.cpu().numpy()
            norm_box = np.array([x1/W,y1/H,x2/W,y2/H])
            score = float(score)
            detections.append(Detection(box=norm_box, class_id=visdrone_id, class_name=VISDRONE_CLASSES[visdrone_id],confidence=score))
            
        return detections
            
            
            