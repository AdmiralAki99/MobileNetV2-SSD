from .writer import TFRecordWriter
from ..dataset_stats.stats import compute_box_stats
from ..dataset_stats.ledger import DatasetStatsLedger

def shard_split(dataset, output_dir: str, split_name: str, ledger: DatasetStatsLedger, dataset_id: str, shard_size=500, stats_only=False):
    writer = None
    if not stats_only:
        # Stats are not wanted
        writer = TFRecordWriter(output_dir=output_dir, shard_size=shard_size)
        writer.open(split_name=split_name)
        
    box_batch = []
    num_images = 0
    num_boxes = 0
    
    for sample in dataset:
        if not stats_only:
            writer.write(sample=sample)
        num_images = num_images + 1
                
        H, W = sample.orig_size
        for box, label in zip(sample.boxes, sample.labels):
            w, h, w_norm, h_norm = compute_box_stats(box, (H,W))
            box_batch.append({
                "image_id": sample.image_id,
                "image_height": H,
                "image_width": W,
                "box_width": w,
                "box_height": h,
                "norm_width": w_norm,
                "norm_height": h_norm,
                "class_label": int(label)
            })
            num_boxes = num_boxes + 1
            
        if len(box_batch) >= shard_size:
            ledger.add_boxes(dataset_id=dataset_id, box_rows= box_batch)
            box_batch = []
            
    if not stats_only:
        writer.close()
    if box_batch:
        ledger.add_boxes(dataset_id=dataset_id, box_rows= box_batch) # Dealing with the leftover boxes
        
    return len(dataset), num_boxes
        