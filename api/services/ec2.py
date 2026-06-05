from ..config import REGION
import boto3


def _client():
    return boto3.client("ec2", region_name=REGION)


def _ssm_client():
    return boto3.client("ssm", region_name=REGION)


def describe_instance(instance_id: str):
    # Getting the description of the instance
    response = _client().describe_instances(InstanceIds=[instance_id])

    if len(response["Reservations"]) == 0:
        return None

    # There is a reserved instance running
    instances = response["Reservations"][0]["Instances"][0]

    # Getting the name
    name = instances["State"]["Name"]
    instance_type = instances.get("InstanceType")
    architecture = instances["Architecture"]
    public_ip = instances.get("PublicIpAddress", None)
    launch_time = instances["LaunchTime"].isoformat()
    tensorboard_url = f"http://{public_ip}:6006" if public_ip is not None else None

    return {
        "name": name,
        "architecture": architecture,
        "instance_type": instance_type,
        "public_ip": public_ip,
        "launch_time": launch_time,
        "tensorboard_url": tensorboard_url,
    }


def stop_training(instance_id: str):
    # Calling the inbuilt ssm command in the EC2 container to stop the training
    _ssm_client().send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": ["docker stop ml-training"]},
    )

    return True
