from sqlalchemy.orm import Session
from .db import Dataset, Box

class DatasetStatsLedger:
    def __init__(self, engine):
        self.session = Session(engine)
    
    def get_or_create_dataset(self, name, split, root_path, config_path= None):
        dataset = self.session.query(Dataset).filter_by(name=name, split=split).one_or_none()
        if not dataset:
            
            dataset = Dataset(name=name, split=split, root_path= root_path, config_path=config_path)
            self.session.add(dataset)
        else:
            # Dataset exists so need to update the record in the database
            if root_path != dataset.root_path:
                dataset.root_path = root_path
                
            if config_path != dataset.config_path:
                dataset.config_path = config_path
        
        self.session.commit()
        return dataset
        
    def add_boxes(self, dataset_id, box_rows: list[dict]):
        boxes = []
        for box in box_rows:
            box_row = Box(dataset_id=dataset_id, **box)
            boxes.append(box_row)
            
        if len(boxes) != 0:
            self.session.add_all(boxes)
            self.session.commit()
            
    def clear_boxes(self, dataset_id):
        
        # Deleting the box from the database
        self.session.query(Box).filter_by(dataset_id=dataset_id).delete()
        self.session.commit()
        
    def finalize_dataset(self, dataset_id, num_images, num_boxes):
        # Finalizing all the changes in the dataset
        self.session.query(Dataset).filter_by(id=dataset_id).update({'num_images':num_images, 'num_boxes': num_boxes})
        self.session.commit()
        
    def close(self):
        self.session.close()