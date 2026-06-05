import sys
from pathlib import Path
from ..config import DYNAMODB_TABLE, REGION

SRC_DIR = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from infrastructure.dynamodb_ledger import ExperimentLedger


def get_ledger():
    # Need to create the experiment ledger
    dynamodb_ledger = ExperimentLedger(table_name=DYNAMODB_TABLE, region=REGION)
    return dynamodb_ledger
