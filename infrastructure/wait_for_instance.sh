#!/usr/bin/env bash

set -euo pipefail

FLEET_ID=$1
REGION=$2

echo "Waiting for fleet $FLEET_ID to launch instance..."
for i in $(seq 1 60); do
  INSTANCE_ID=$(aws ec2 describe-fleet-instances \
    --fleet-id "$FLEET_ID" \
    --region "$REGION" \
    --query 'ActiveInstances[0].InstanceId' \
    --output text 2>/dev/null || true)
  if [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "None" ]; then
    echo "Instance $INSTANCE_ID found, waiting for running state..."
    aws ec2 wait instance-running \
      --instance-ids "$INSTANCE_ID" \
      --region "$REGION"
    echo "Instance ready"
    exit 0
  fi
  echo "Attempt $i/60: no instance yet, retrying in 10s..."
  sleep 10
done

echo "ERROR: Timeout after 600s waiting for fleet instance"
exit 1
