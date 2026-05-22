import ray
from sqlalchemy.orm import Session
from pathlib import Path

from .frame_sampler import FrameSampler
from .detectors import YOLODetector, RTDETRDetector, GroundingDINODetector
from .consensus import ConsensusEngine
from .writer import TFRecordWriter
from .db import build_engine, Video, Frame, Annotation

@ray.remote
class ETLWorker:
    def __init__(self, config: dict):
        self.sampler = FrameSampler(config= config)
        self.detectors = [YOLODetector(config=config), RTDETRDetector(config= config), GroundingDINODetector(config= config)]
        for detector in self.detectors:
            detector.load()
        
        self.consensus = ConsensusEngine(config= config)
        self.writer = TFRecordWriter(config= config)
        engine = build_engine(url= config.get('database', {}).get('url'))
        self.session = Session(engine)
    
    def process_video(self, video_path: str):       
        sampled_frames, metadata = self.sampler.sample(video_path=video_path)
        
        video_record = Video(source_file= video_path)
        video_record.filename = Path(video_path).name
        video_record.duration= metadata['duration']
        video_record.fps = metadata['fps']
        video_record.total_frames= metadata['total_frames']
        video_record.height = metadata['height']
        video_record.width = metadata['width']
        
        self.session.add(video_record)
        self.session.flush()
        
        # Opening the writer
        self.writer.open(str(video_record.id))
        
        # Running all detection models
        for frame in sampled_frames:
            detections_per_model = {detector.name: detector.predict(frame.image) for detector in self.detectors}
            consensus_annotations = self.consensus.compute(detections_per_model= detections_per_model)
            if not consensus_annotations:
                continue
            
            self.writer.write(frame.image, consensus_annotations, str(video_record.id),video_path)
            sampled_frame = Frame(video_id= video_record.id, frame_index=frame.frame_index, width=frame.width, height=frame.height, timestamp_s= frame.timestamp_s, scene_change_score= frame.scene_change_score)
            self.session.add(sampled_frame)
            self.session.flush()
            
            # Creating the annotations
            for annotation in consensus_annotations:
                x1,y1,x2,y2 = annotation.box
                class_id = annotation.class_id
                votes = annotation.votes
                class_name= annotation.class_name
                consensus_score = annotation.consensus_score
                model_confidences = annotation.model_confidence
                
                annotation = Annotation(
                    frame_id= sampled_frame.id,
                    x1=x1, 
                    y1=y1, 
                    x2=x2, 
                    y2=y2, 
                    class_id= class_id, 
                    class_name= class_name, 
                    votes= votes,
                    consensus_score= consensus_score,
                    model_confidences= model_confidences
                )
                
                self.session.add(annotation)
                self.session.flush()
                
        self.session.commit()
        self.writer.close()