import boto3
from ..config import CHECKPOINT_BUCKET, ARTIFACTS_BUCKET, REGION


def _s3_client():
    return boto3.client("s3", region_name=REGION)


def list_checkpoints(experiment_id: str):
    response = _s3_client().list_objects_v2(Bucket=CHECKPOINT_BUCKET, Prefix=f"runs/{experiment_id}/")
    return [
        {
            "Key": contents["Key"],
            "Size": contents["Size"],
            "LastModified": contents["LastModified"].isoformat(),
        }
        for contents in response.get("Contents", [])
    ]


def check_artifacts(experiment_id: str):

    result = {"saved_model": False, "onnx": False, "priors": False}
    try:
        response = _s3_client().list_objects_v2(Bucket=ARTIFACTS_BUCKET, Prefix=f"{experiment_id}/")
        contents = response.get("Contents", [])
        for content in contents:
            if "saved_model" in content["Key"]:
                result["saved_model"] = True
            if content["Key"].endswith(".onnx"):
                result["onnx"] = True
            if "priors" in content["Key"]:
                result["priors"] = True
    except Exception:
        return result

    return result
