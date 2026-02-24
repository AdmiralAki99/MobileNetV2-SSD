import os
import urllib.request
from datetime import datetime, timezone
import boto3
from typing import Any
from botocore.exceptions import ClientError
import decimal

class ExperimentLedger:
    def __init__(self, table_name: str, region: str, logger= None):
        dynamodb_table = boto3.resource('dynamodb',region_name=region)
        self._table = dynamodb_table.Table(table_name)
        self._logger = logger
        
    def get_experiment_state(self, experiment_id: str, fingerprint: str):
        # Function to get the state of the experiment for ledger (Reader)
        resp = self._table.get_item(
            Key= {'experiment_id': experiment_id, 'fingerprint': fingerprint},
            ConsistentRead= True
        )
        
        return resp.get('Item')
    
    def claim_experiment(self, experiment_id: str, fingerprint: str, timestamp, instance_id):
        try:
            
            self._table.update_item(
               Key= {'experiment_id': experiment_id, 'fingerprint': fingerprint},
               UpdateExpression= "SET #s = :running, claimed = :now, ec2_instance = :instance_id, run_timestamp = :timestamp REMOVE failure_reason",
               ConditionExpression= "#s IN (:pending, :failed)",
               ExpressionAttributeNames= {'#s': 'status'},
               ExpressionAttributeValues= {
                    ':running': 'running',
                    ':pending': 'pending',
                    ':failed': 'failed',
                    ':now': datetime.now(timezone.utc).isoformat(),
                    ':instance_id': instance_id,
                    ':timestamp': timestamp,
                }
            )
            
            # It succeded
            return True
        except ClientError as err:
            if err.response['Error']['Code'] == "ConditionalCheckFailedException":
                return False
            
            raise
        
    def update_checkpoint_pointer(self, experiment_id: str, fingerprint: str, checkpoint_s3_path: str, step: int):
        try:
            self._table.update_item(
                Key={'experiment_id': experiment_id, 'fingerprint': fingerprint},
                UpdateExpression= "SET checkpoint_s3_path = :path, total_steps = :step",
                ExpressionAttributeValues= {
                    ':path': checkpoint_s3_path,
                    ':step': step
                }
            )
        except Exception as err:
            if self._logger: 
                self._logger.warning(f"Failed to update checkpoint pointer: {err}")
    
    def mark_success(self, experiment_id: str, fingerprint: str, checkpoint_s3_path: str, artifact_s3_path: str, best_epoch: int, total_steps: int, best_metric: float):
        try:
            
            self._table.update_item(
               Key= {'experiment_id': experiment_id, 'fingerprint': fingerprint},
               UpdateExpression= "SET #s = :success, completed_at = :now, artifact_s3_path = :artfctpath, checkpoint_s3_path = :chkptpath, total_steps = :steps, best_metric = :metric, best_epoch = :epoch",
               ConditionExpression= "#s = :running",
               ExpressionAttributeNames= {'#s': 'status'},
               ExpressionAttributeValues= {
                    ':running': 'running',
                    ':success': 'success',
                    ':now': datetime.now(timezone.utc).isoformat(),
                    ':chkptpath': checkpoint_s3_path,
                    ':artfctpath': artifact_s3_path,
                    ':metric': decimal.Decimal(str(best_metric)),
                    ':epoch': best_epoch,
                    ':steps': total_steps
                }
            )
            
            # It succeded
            return True
        except ClientError as err:
            if err.response['Error']['Code'] == "ConditionalCheckFailedException":
                if self._logger: 
                    self._logger.warning(f"Failed to update mark success for experiment id {experiment_id}: {err}")
                return
            
            raise
        
    def mark_failure(self, experiment_id: str, fingerprint: str, checkpoint_s3_path: str | None, total_steps: int, reason: str):
        try:
            
            update_expression = "SET #s = :failed, completed_at = :now, failure_reason = :reason, total_steps = :steps"
            
            attribute_values = {
                ':running': 'running',
                ':failed': 'failed',
                ':now': datetime.now(timezone.utc).isoformat(),
                ':reason': reason,
                ':steps': total_steps
            }
            
            if checkpoint_s3_path:
                update_expression = update_expression + ', checkpoint_s3_path = :chkptpath'
                attribute_values[':chkptpath'] = checkpoint_s3_path
            
            self._table.update_item(
               Key= {'experiment_id': experiment_id, 'fingerprint': fingerprint},
               UpdateExpression= update_expression,
               ConditionExpression= "#s = :running",
               ExpressionAttributeNames= {'#s': 'status'},
               ExpressionAttributeValues= attribute_values
            )
            
            # It succeded
            return True
        except ClientError as err:
            if err.response['Error']['Code'] == "ConditionalCheckFailedException":
                if self._logger: 
                    self._logger.warning(f"Failed to update checkpoint pointer: {err}")
                return
            raise
        
    def register_experiment(self, item: dict[str, Any]):
        # This function is to create the experiment in the ledger
        try:
            self._table.put_item(
                Item= item,
                ConditionExpression= "attribute_not_exists(experiment_id) AND attribute_not_exists(fingerprint)"
            )
            return True
        except ClientError as err:
            if err.response['Error']['Code'] == "ConditionalCheckFailedException":
                return False # Expected since the experiment exists
            
    def list_experiments(self):
        # Function to get all the experiments in the ledger
        resp= self._table.scan()
        items= resp.get('Items',[])
        while 'LastEvaluatedKey' in resp:
            # There could be more stuff to scan in the page
            resp= self._table.scan(ExclusiveStartKey= resp['LastEvaluatedKey'])
            items.extend(resp.get('Items',[])) # Concat all of them into one big list
            
        return items
    
    def reset_failed(self, experiment_id: str):
        # Resting the failed experiement to pending status so that it can be started by another run
        items= self.list_experiments()
        reset_counter = 0 
        # Now need to go through the ones where the status is failed
        for item in items:
            if item['experiment_id'] == experiment_id and item['status'] == 'failed':
                try:
                    # Resetting the experiment
                    self._table.update_item(
                        Key={'experiment_id': item['experiment_id'], 'fingerprint': item['fingerprint']},
                        UpdateExpression="SET #s = :pending REMOVE failure_reason, claimed, ec2_instance, run_timestamp",
                        ConditionExpression= "#s = :failed",
                        ExpressionAttributeNames= {'#s': 'status'},
                        ExpressionAttributeValues={
                            ':pending': 'pending',
                            ':failed': 'failed'
                        }
                    )
                
                    reset_counter= reset_counter + 1
                except ClientError as err:
                    if err.response['Error']['Code'] == "ConditionalCheckFailedException":
                       return
                   
                    raise
                    
        return reset_counter
                
                
def get_ec2_instance_id():
    # Inside the AWS EC2 instance the id can be got by making a request to a URL
    try:
        token = urllib.request.Request(url= "http://169.254.169.254/latest/api/token", method="PUT", headers={'X-aws-ec2-metadata-token-ttl-seconds': 21600})
        
        with urllib.request.urlopen(token, timeout= 2) as resp:
            resp_token = resp.read().decode()
            
        id_req = urllib.request.Request(url=  "http://169.254.169.254/latest/meta-data/instance-id", headers={"X-aws-ec2-metadata-token": resp_token})
        
        with urllib.request.urlopen(id_req, timeout= 2) as resp:
            return resp.read().decode() # This is the instance id
        
    except Exception:
        return None # In the case it doesnt work, the EC2 ID doesnt change much in the grand scheme of things
        
def build_dynamodb_ledger(config: dict[str, Any], logger = None):
    table_name= config.get('infrastructure',{}).get('dynamodb_table', None)
    
    if not table_name:
        if logger:
            logger.info("No DynamoDB table configured. Ledger disabled.")
            
        return None

    # Getting the region
    region= config.get('infrastructure', {}).get('region', 'us-east-1')
    
    try:
        ledger= ExperimentLedger(table_name= table_name, region= region, logger= logger)
        return ledger
    except Exception as err:
        if logger:
            logger.warning(f"DynamoDB ledger unavailable: {err}. Running without ledger.")
            
        return None