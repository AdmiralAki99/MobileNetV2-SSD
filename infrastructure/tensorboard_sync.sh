#!/bin/bash
# TensorBoard S3 sync script
# Continuously syncs training logs from S3 to local directory for TensorBoard

set -e

S3_BUCKET="${S3_BUCKET:-}"
S3_PREFIX="${S3_PREFIX:-}"
LOCAL_DIR="${LOCAL_DIR:-/tensorboard/logs}"
SYNC_INTERVAL="${SYNC_INTERVAL:-60}"

if [ -z "$S3_BUCKET" ]; then
    echo "ERROR: S3_BUCKET environment variable not set"
    exit 1
fi

echo "Starting TensorBoard S3 sync..."
echo "  S3: s3://${S3_BUCKET}/${S3_PREFIX}"
echo "  Local: ${LOCAL_DIR}"
echo "  Interval: ${SYNC_INTERVAL}s"

mkdir -p "$LOCAL_DIR"

while true; do
    echo "[$(date)] Syncing from S3..."
    aws s3 sync "s3://${S3_BUCKET}/${S3_PREFIX}" "$LOCAL_DIR" --quiet
    echo "[$(date)] Sync complete. Waiting ${SYNC_INTERVAL}s..."
    sleep "$SYNC_INTERVAL"
done
