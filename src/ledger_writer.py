import os
import boto3
from datetime import datetime, timezone


def _table():
    return boto3.resource("dynamodb", region_name=os.environ["AWS_DEFAULT_REGION"]).Table(os.environ["DYNAMODB_TABLE"])


def _key():
    return {
        "experiment_id": os.environ["EXPERIMENT_ID"],
        "fingerprint": os.environ["FINGERPRINT"],
    }


def write_metric(metric: float, metric_name: str):
    _table().update_item(
        Key=_key(),
        UpdateExpression="SET best_metric = :m, metric_name = :mn",
        ExpressionAttributeValues={":m": str(metric), ":mn": metric_name},
    )


def write_checkpoint(s3_path: str):
    _table().update_item(
        Key=_key(),
        UpdateExpression="SET checkpoint_s3_path = :p",
        ExpressionAttributeValues={":p": s3_path},
    )


def write_status(status: str, reason: str | None = None):
    expr = "SET #s = :s, completed_at = :t"
    names = {"#s": "status"}
    values = {":s": status, ":t": datetime.now(timezone.utc).isoformat()}
    if reason:
        expr += ", failure_reason = :r"
        values[":r"] = reason
    _table().update_item(
        Key=_key(),
        UpdateExpression=expr,
        ExpressionAttributeNames=names,
        ExpressionAttributeValues=values,
    )
