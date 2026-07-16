import os
from pathlib import Path

from ..dataset_stats.db import build_engine
from ..dataset_stats.ledger import DatasetStatsLedger
from datasets.base import create_dataset_from_config
from .pipeline import shard_split

from ....config import ETL_DB_URL, PROJECT_ROOT

def _dataset_prefix(root: str) -> str:
    data_root = os.getenv("DATA_ROOT", "datasets")
    if not os.path.isabs(data_root):
        data_root = str((PROJECT_ROOT / data_root).resolve())
    return str(Path(root).resolve().relative_to(data_root))

def run_tfrecord_job(config, config_path, stats_only):
    dataset_name = config["data"]["dataset_name"]
    dataset_bucket = config["infrastructure"]["storage"]["data_bucket"]
    dataset_prefix = _dataset_prefix(config["data"]["root"])
    output_dir = f"{dataset_bucket}/{dataset_prefix}"
    
    engine = build_engine(ETL_DB_URL)
    ledger = DatasetStatsLedger(engine=engine)
    
    for split_key in ("train_split","val_split"):
        split = config['data'][split_key]
        dataset = create_dataset_from_config(config=config, split=split)
        dataset_row = ledger.get_or_create_dataset(name=dataset_name, split=dataset.split, root_path=output_dir, config_path=config_path)
        ledger.clear_boxes(dataset_id=dataset_row.id)
        
        num_images, num_boxes = shard_split(dataset=dataset, output_dir=output_dir, split_name=dataset.split, ledger=ledger, dataset_id=dataset_row.id, stats_only=stats_only)
        ledger.finalize_dataset(dataset_id=dataset_row.id, num_images=num_images, num_boxes=num_boxes)
    
    # Close the ledger
    ledger.close()