import pytest
import tensorflow as tf
import numpy as np

from training.metrics import MeanAveragePrecision, MetricsCollection, convert_batch_images_to_metric_format, build_metrics_from_config

@pytest.mark.unit
def test_map_perfect_predictions():
    metric = MeanAveragePrecision(
        num_classes=3,
        iou_thresholds=[0.5],
        style="voc",
    )
    
    # Ground truth: 2 boxes of class 1, 1 box of class 2
    ground_truths = [
        {
            "image_id": "img1",
            "boxes": np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=np.float32),
            "labels": np.array([1, 1], dtype=np.int32),
        },
        {
            "image_id": "img2",
            "boxes": np.array([[20, 20, 60, 60]], dtype=np.float32),
            "labels": np.array([2], dtype=np.int32),
        },
    ]
    
    # Predictions: exact matches with high confidence
    predictions = [
        {
            "image_id": "img1",
            "boxes": np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=np.float32),
            "scores": np.array([0.9, 0.8], dtype=np.float32),
            "labels": np.array([1, 1], dtype=np.int32),
        },
        {
            "image_id": "img2",
            "boxes": np.array([[20, 20, 60, 60]], dtype=np.float32),
            "scores": np.array([0.95], dtype=np.float32),
            "labels": np.array([2], dtype=np.int32),
        },
    ]
    
    metric.update(predictions, ground_truths)
    results = metric.compute()

    assert "mAP@0.50" in results
    assert np.isclose(results["mAP@0.50"], 1.0), f"Expected 1.0, got {results['mAP@0.50']}"

@pytest.mark.unit    
def test_map_no_predictions():
    """No predictions should give mAP = 0.0"""
    metric = MeanAveragePrecision(
        num_classes=3,
        iou_thresholds=[0.5],
        style="voc",
    )
    
    ground_truths = [
        {
            "image_id": "img1",
            "boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        },
    ]
    
    predictions = [
        {
            "image_id": "img1",
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int32),
        },
    ]
    
    metric.update(predictions, ground_truths)
    results = metric.compute()
    
    assert results["mAP@0.50"] == 0.0

@pytest.mark.unit    
def test_map_no_ground_truth():
    metric = MeanAveragePrecision(
        num_classes=3,
        iou_thresholds=[0.5],
        style="voc",
    )
    
    ground_truths = []
    predictions = []
    
    metric.update(predictions, ground_truths)
    results = metric.compute()

    assert results["mAP@0.50"] == 0.0

@pytest.mark.unit  
def test_map_wrong_class():
    metric = MeanAveragePrecision(
        num_classes=3,
        iou_thresholds=[0.5],
        style="voc",
    )
    
    ground_truths = [
        {
            "image_id": "img1",
            "boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),  # Class 1
        },
    ]
    
    predictions = [
        {
            "image_id": "img1",
            "boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),  # Same box
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([2], dtype=np.int32),  # Wrong class!
        },
    ]
    
    metric.update(predictions, ground_truths)
    results = metric.compute()
    
    # Class 1 has GT but no predictions -> AP = 0
    # Class 2 has predictions but no GT -> ignored
    assert results["mAP@0.50"] == 0.0

@pytest.mark.unit    
def test_map_low_iou():

    metric = MeanAveragePrecision(
        num_classes=2,
        iou_thresholds=[0.5],
        style="voc",
    )
    
    ground_truths = [
        {
            "image_id": "img1",
            "boxes": np.array([[0, 0, 100, 100]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        },
    ]
    
    # Prediction only overlaps a tiny bit (IoU < 0.5)
    predictions = [
        {
            "image_id": "img1",
            "boxes": np.array([[90, 90, 150, 150]], dtype=np.float32),  # Small overlap
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        },
    ]
    
    metric.update(predictions, ground_truths)
    results = metric.compute()
    
    # IoU is too low, so it's a false positive
    assert results["mAP@0.50"] == 0.0

@pytest.mark.unit    
def test_map_multiple_iou_thresholds():
    
    metric = MeanAveragePrecision(
        num_classes=2,
        iou_thresholds=[0.5, 0.75],
        style="voc",
    )
    
    ground_truths = [
        {
            "image_id": "img1",
            "boxes": np.array([[0, 0, 100, 100]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        },
    ]
    
    # Prediction with ~60% IoU (passes 0.5, fails 0.75)
    # Box [0,0,100,100] vs [20,20,120,120]
    # Intersection: [20,20,100,100] = 80x80 = 6400
    # Union: 10000 + 10000 - 6400 = 13600
    # IoU = 6400/13600 ≈ 0.47 -> fails both, let's adjust
    
    # Better: [0,0,100,100] vs [10,10,110,110]
    # Intersection: [10,10,100,100] = 90x90 = 8100
    # Union: 10000 + 10000 - 8100 = 11900
    # IoU = 8100/11900 ≈ 0.68 -> passes 0.5, fails 0.75
    predictions = [
        {
            "image_id": "img1",
            "boxes": np.array([[10, 10, 110, 110]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        },
    ]
    
    metric.update(predictions, ground_truths)
    results = metric.compute()
    
    assert "mAP@0.50" in results
    assert "mAP@0.75" in results
    assert results["mAP@0.50"] == 1.0, f"Expected 1.0 at IoU=0.5, got {results['mAP@0.50']}"
    assert results["mAP@0.75"] == 0.0, f"Expected 0.0 at IoU=0.75, got {results['mAP@0.75']}"

@pytest.mark.unit    
def test_map_duplicate_detections():

    metric = MeanAveragePrecision(
        num_classes=2,
        iou_thresholds=[0.5],
        style="voc",
    )
    
    # Need TWO GT boxes to see recall change
    ground_truths = [
        {
            "image_id": "img1",
            "boxes": np.array([
                [10, 10, 50, 50],
                [100, 100, 150, 150],
            ], dtype=np.float32),
            "labels": np.array([1, 1], dtype=np.int32),
        },
    ]
    
    # Three predictions: first matches GT1, second is duplicate, third matches GT2
    predictions = [
        {
            "image_id": "img1",
            "boxes": np.array([
                [10, 10, 50, 50],      # TP (matches GT box 1)
                [10, 10, 50, 50],      # FP (duplicate - GT box 1 already matched)
                [100, 100, 150, 150],  # TP (matches GT box 2)
            ], dtype=np.float32),
            "scores": np.array([0.95, 0.9, 0.8], dtype=np.float32),
            "labels": np.array([1, 1, 1], dtype=np.int32),
        },
    ]
    
    # Sorted by score: [0.95, 0.9, 0.8]
    # Pred 1 (0.95): TP, recall=0.5, precision=1.0
    # Pred 2 (0.9):  FP (duplicate), recall=0.5, precision=0.5  
    # Pred 3 (0.8):  TP, recall=1.0, precision=0.67
    # AP ≈ 0.83 (area under interpolated P-R curve)
    
    metric.update(predictions, ground_truths)
    results = metric.compute()
    
    assert results["mAP@0.50"] < 1.0
 
@pytest.mark.unit   
def test_map_reset():

    metric = MeanAveragePrecision(
        num_classes=2,
        iou_thresholds=[0.5],
        style="voc",
    )
    
    ground_truths = [
        {
            "image_id": "img1",
            "boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        },
    ]
    predictions = [
        {
            "image_id": "img1",
            "boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        },
    ]
    
    metric.update(predictions, ground_truths)
    result1 = metric.compute()
    assert result1["mAP@0.50"] == 1.0
    
    metric.reset()
    result2 = metric.compute()
    assert result2["mAP@0.50"] == 0.0  # No data after reset

@pytest.mark.unit    
def test_map_coco_style():

    metric = MeanAveragePrecision(
        num_classes=2,
        iou_thresholds=[0.5, 0.75],
        style="coco",
    )
    
    ground_truths = [
        {
            "image_id": "img1",
            "boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        },
    ]
    predictions = [
        {
            "image_id": "img1",
            "boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        },
    ]
    
    metric.update(predictions, ground_truths)
    results = metric.compute()

    print(f"DEBUG: results = {results}")
    
    # COCO style reports mAP@[0.50:0.75]
    assert "mAP@[0.50:0.75]" in results
    assert "AP@0.50" in results
    assert "AP@0.75" in results

@pytest.mark.unit    
def test_metrics_collection():

    metrics = MetricsCollection({
        "voc": MeanAveragePrecision(num_classes=2, iou_thresholds=[0.5], style="voc"),
        "coco": MeanAveragePrecision(num_classes=2, iou_thresholds=[0.5, 0.75], style="coco"),
    })
    
    ground_truths = [
        {
            "image_id": "img1",
            "boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        },
    ]
    predictions = [
        {
            "image_id": "img1",
            "boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        },
    ]
    
    metrics.update(predictions, ground_truths)
    results = metrics.compute()
    
    # Should have prefixed keys
    assert "voc/mAP@0.50" in results
    assert "coco/mAP@[0.50:0.75]" in results

@pytest.mark.unit    
def test_convert_batch_to_metric_format():

    B = 2  # batch size
    N = 5  # predictions per image
    M = 3  # GT boxes per image
    
    pred_boxes = tf.random.uniform((B, N, 4), 0, 100)
    pred_scores = tf.random.uniform((B, N), 0, 1)
    pred_labels = tf.random.uniform((B, N), 1, 3, dtype=tf.int32)
    
    gt_boxes = tf.random.uniform((B, M, 4), 0, 100)
    gt_labels = tf.random.uniform((B, M), 1, 3, dtype=tf.int32)
    gt_mask = tf.constant([[True, True, False], [True, False, False]])  # 2 valid, 1 valid
    
    image_ids = tf.constant(["img1", "img2"])
    
    preds, gts = convert_batch_images_to_metric_format(
        pred_boxes=pred_boxes,
        pred_scores=pred_scores,
        pred_labels=pred_labels,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        gt_mask=gt_mask,
        image_ids=image_ids,
    )
    
    assert len(preds) == B
    assert len(gts) == B
    
    # Check structure
    assert preds[0]["image_id"] == "img1"
    assert preds[0]["boxes"].shape == (N, 4)
    assert preds[0]["scores"].shape == (N,)
    assert preds[0]["labels"].shape == (N,)
    
    # Check GT masking worked
    assert gts[0]["boxes"].shape == (2, 4)  # 2 valid boxes
    assert gts[1]["boxes"].shape == (1, 4)  # 1 valid box

@pytest.mark.unit    
def test_build_metrics_from_config():

    config = {
        "model": {
            "num_classes": 21,
        },
        "eval": {
            "metrics": {
                "voc_ap_50": {
                    "type": "voc_ap",
                    "iou_thresholds": [0.5],
                },
                "coco_map": {
                    "type": "coco_map",
                    "iou_thresholds": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                },
            },
        },
    }
    
    metrics = build_metrics_from_config(config)
    
    assert isinstance(metrics, MetricsCollection)
    assert "voc_ap_50" in metrics.metrics
    assert "coco_map" in metrics.metrics
    assert metrics.metrics["voc_ap_50"].style == "voc"
    assert metrics.metrics["coco_map"].style == "coco"