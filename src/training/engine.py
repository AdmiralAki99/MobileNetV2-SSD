import tensorflow as tf
import numpy as np
from typing import Any

from training.metrics import convert_batch_images_to_metric_format
from mobilenetv2ssd.core.precision_config import PrecisionConfig
from mobilenetv2ssd.core.logger import Logger
from training.checkpoints import CheckpointManager

from mobilenetv2ssd.models.ssd.orchestration.targets_orch import building_training_targets
from mobilenetv2ssd.models.ssd.orchestration.conf_loss_orch import build_conf_loss
from mobilenetv2ssd.models.ssd.orchestration.hard_neg_orch import select_hard_negatives
from mobilenetv2ssd.models.ssd.orchestration.loss_orch import calculate_final_loss
from mobilenetv2ssd.models.ssd.orchestration.post_process_orch import build_decoded_boxes
from mobilenetv2ssd.models.factory import build_ssd_model

from mobilenetv2ssd.core.utils import ssd_get_prior_stats, calculate_model_prediction_health, calculate_nms_health_scores, calculate_gt_health_scores, calculate_pred_health_metrics, verify_pred_boxes_sanity, gt_box_range, calculate_iou_sanity_top1, prediction_box_bad_frac, ground_truth_box_bad_frac, inference_function
from mobilenetv2ssd.core.exceptions import GracefulShutdownException

from training.amp import AMPContext
from training.ema import EMA
from training.metrics import convert_batch_images_to_metric_format, MetricsCollection
from training.shutdown import ShutdownHandler

from infrastructure.s3_sync import S3SyncClient

def training_step(config: dict[str,Any],model: tf.keras.Model, priors_cxcywh: tf.Tensor, batch: dict[str, Any], precision_config: PrecisionConfig, logger: Logger):
    
    # First get the batch elements from the dataset
    image, boxes, labels, gt_mask = batch['image'], batch['boxes'], batch['labels'], batch['gt_mask']

    tf.debugging.assert_equal(tf.rank(image), tf.constant(4, dtype = tf.int32), message = f"The image has rank : {tf.rank(image)}, expected: 4")
    
    # Building the training targets
    localization_targets, classification_targets, positive_mask, negative_mask, ignore_mask, diagnostics = building_training_targets(config = config, priors_cxcywh = priors_cxcywh, gt_labels= labels, gt_boxes_xyxy= boxes, gt_valid_mask= gt_mask, precision_config= precision_config)
    
    tf.debugging.assert_equal(tf.shape(classification_targets)[:2], tf.shape(localization_targets)[:2], message = f"The shapes between the are different between classification targets:{tf.shape(classification_targets)[:2]}, but expected {tf.shape(localization_targets)[:2]}")
    
    predicted_offsets, predicted_logits = model(image, training = True)

    tf.debugging.assert_equal(tf.shape(predicted_logits)[:2], tf.shape(localization_targets)[:2], message = f"The shapes between the are different between localization logits :{tf.shape(predicted_logits)[:2]}, but expected {tf.shape(localization_targets)[:2]}")

    tf.debugging.assert_equal(tf.shape(localization_targets), tf.shape(predicted_offsets), message=f"The shapes between the are different between localization targets:{tf.shape(localization_targets)}, and predicted_offsets:{tf.shape(predicted_offsets)}")

    # Calculating the confidence loss
    conf_loss, candidate_negative_mask = build_conf_loss(config = config, predicted_logits = predicted_logits, classification_targets = classification_targets, pos_mask = positive_mask, neg_mask = negative_mask, ignore_mask = ignore_mask, precision_config = precision_config)
    
    # Perform Hard negative mining
    selected_negative_mask = select_hard_negatives(config = config, conf_loss = conf_loss, positive_mask = positive_mask, negative_mask = candidate_negative_mask)

    # Logging the value
    priors_stats = ssd_get_prior_stats(positive_mask, selected_negative_mask)
    
    # Calculate the loss
    loss_dict = calculate_final_loss(config = config, predicted_offsets = predicted_offsets, predicted_logits = predicted_logits, localization_targets = localization_targets, classification_targets = classification_targets, positive_mask = positive_mask, negative_mask = selected_negative_mask, precision_config = precision_config)
    
    return loss_dict, priors_stats # Checking if the boxes are correctly formatted

def train_one_epoch(config: dict[str, Any], epoch: int, model: tf.keras.Model, train_dataset: tf.data.Dataset, optimizer: tf.keras.optimizers.Optimizer, priors_cxcywh: tf.Tensor, precision_config: PrecisionConfig, ema : EMA, amp: AMPContext, logger: Logger, global_step_offset: int = 0, log_every: int = 5, max_steps: int|None = None, shutdown_handler: ShutdownHandler = None):
    # TODO: If num_pos is 0 the skip the loc loss and only do a safe cls loss OR skip the update completely
    # Running counter of the loss value
    loss_meter = tf.keras.metrics.Mean(name="loss")

    global_step = global_step_offset
    
    for step, batch in enumerate(train_dataset):
        # Use Gradient Tape to get the losses
        if shutdown_handler and shutdown_handler.is_requested():
            signal_number = shutdown_handler.signal_number or 15
            
            raise GracefulShutdownException(signal_number= signal_number)
            
        with tf.GradientTape() as tape:
            with amp.autocast():
                loss_dict, prior_stats = training_step(config = config, model = model, priors_cxcywh = priors_cxcywh, batch = batch, precision_config = precision_config, logger= logger)
                total_loss = loss_dict['total_loss']

                # Making sure the optimizer operation is guarded since it can be disabled
                if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                    scaled_loss = optimizer.scale_loss(total_loss)
                else:
                    scaled_loss = total_loss

            
        gradients = tape.gradient(scaled_loss, model.trainable_variables)

        # Only changing the values for the stuff that is not None
        grads_and_vars = [(g, v) for g, v in zip(gradients, model.trainable_variables) if g is not None]
        if not grads_and_vars:
            global_step += 1
            continue

        optimizer.apply_gradients(grads_and_vars)

        # Updating the EMA based on the predefined conditions
        ema.update(global_step)

        # Adding to the running counter
        loss_meter.update_state(total_loss)

        if step % log_every == 0:
            logger.metric(f"Epoch {epoch}, Step {step}, Loss {float(total_loss.numpy())}, Num Pos: {int(loss_dict['num_pos'].numpy())}")
            logger.log_scalar(tag= "train/loss", value= total_loss.numpy(), step= global_step)
            logger.metric(f"Number of Positive Priors: {prior_stats['num_pos']}")
            logger.metric(f"Number of Negative Priors: {prior_stats['num_neg']}")
            logger.metric(f"Number of Negative to Positive Ratio: {prior_stats['neg_pos_ratio']}")
            logger.metric(f"Number of Zero Positive Priors Ratio: {prior_stats['zero_pos_frac']}")
            logger.metric(f"Number of Min Positive Priors: {prior_stats['pos_min']}")
            logger.metric(f"Number of Mean Positive Priors: {prior_stats['pos_mean']}")
            logger.metric(f"Number of Max Positive Priors: {prior_stats['pos_max']}")
            logger.log_scalars(tag= "train", values= prior_stats, step= step)
            if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                logger.log_scalar(tag= "train/lr", value= optimizer.inner_optimizer.learning_rate.numpy(), step= global_step)
            else:
                logger.log_scalar(tag= "train/lr", value= optimizer.learning_rate.numpy(), step= global_step)
                
            logger.flush()
            
        global_step += 1
        
        # Safe guard from pushing past a certain limit
        if max_steps is not None and step + 1 >= max_steps:
            break

    global_step_end = global_step_offset + (step + 1)
    
    return loss_meter.result(), global_step_end

def evaluate_step(config: dict[str,Any],model: tf.keras.Model, priors_cxcywh: tf.Tensor, batch: dict[str, Any], precision_config: PrecisionConfig, logger: Logger):
    
    image, boxes, labels, gt_mask = batch['image'], batch['boxes'], batch['labels'], batch['gt_mask']

    tf.debugging.assert_equal(tf.rank(image), tf.constant(4, dtype = tf.int32), message = f"The image has rank : {tf.rank(image)}, expected: 4")

    # Forward pass
    predicted_offsets, predicted_logits = model(image, training = False)

    calculate_model_prediction_health(predicted_logits, predicted_offsets, logger)

    # Postprocess
    nmsed_boxes, nmsed_scores, nmsed_classes, decoded_classes, classes, valid_detections = build_decoded_boxes(config = config, predicted_offsets = predicted_offsets, predicted_logits = predicted_logits, priors = priors_cxcywh, precision_config = precision_config)

    # Decoding the GT Boxes back to image space
    image_shape = tf.shape(image)
    
    H = image_shape[1]
    W = image_shape[2]

    x1, y1, x2, y2 = tf.split(boxes, num_or_size_splits = 4, axis = -1)

    x1 = x1 * tf.cast(W, dtype = boxes.dtype)
    y1 = y1 * tf.cast(H, dtype = boxes.dtype)
    x2 = x2 * tf.cast(W, dtype = boxes.dtype)
    y2 = y2 * tf.cast(H, dtype = boxes.dtype)

    boxes = tf.concat([x1,y1,x2,y2], axis = -1)
    
    # Output the stuff in a dict
    return {
        'pred_boxes': nmsed_boxes,
        'pred_scores': nmsed_scores,
        'pred_classes': nmsed_classes,
        'gt_boxes': boxes,
        'gt_labels': labels,
        'gt_mask': gt_mask,
        'valid_detections': valid_detections
    }

def evaluate(config: dict[str, Any], model: tf.keras.Model, priors_cxcywh: tf.Tensor, val_dataset: tf.data.Dataset, metrics_manager: MetricsCollection, precision_config: PrecisionConfig, ema: EMA, logger: Logger = None, max_steps: int| None = None, log_every: int = 1, heavy_log_every: int = 100, shutdown_handler: ShutdownHandler = None):
    # Reset the metrics manager
    metrics_manager.reset()
    
    # Checking if the EMA is up
    with ema.eval_context(model):
        for step, batch in enumerate(val_dataset):
        
            if shutdown_handler and shutdown_handler.is_requested():
                signal_number = shutdown_handler.signal_number or 15
            
                raise GracefulShutdownException(signal_number= signal_number)
            # Evaluating step
            evaluation_output = evaluate_step(config = config, model = model, priors_cxcywh = priors_cxcywh, batch = batch, precision_config = precision_config, logger = logger)

            # Compute the metrics
            predictions, ground_truths = convert_batch_images_to_metric_format(pred_boxes = evaluation_output['pred_boxes'], pred_scores = evaluation_output['pred_scores'], pred_labels = evaluation_output['pred_classes'], gt_boxes = evaluation_output['gt_boxes'], gt_labels = evaluation_output['gt_labels'], gt_mask = evaluation_output['gt_mask'], image_ids = batch['image_id'])
        
            # Log the metrics
            metrics_manager.update(predictions, ground_truths)

            if max_steps is not None and step + 1 >= max_steps:
                break

            if step % heavy_log_every == 0:
                nms_health_metrics = calculate_nms_health_scores(evaluation_output['pred_scores'], evaluation_output['valid_detections'])
                ground_truth_health_metrics = calculate_gt_health_scores(batch['boxes'], batch['labels'], batch['gt_mask'])
                gt_box_sanity = gt_box_range(evaluation_output['gt_boxes'],  evaluation_output['gt_mask'])
                pred_health_metrics = calculate_pred_health_metrics(evaluation_output['pred_scores'], evaluation_output['pred_classes'], evaluation_output['valid_detections'])
                pred_box_sanity = verify_pred_boxes_sanity(evaluation_output['pred_boxes'], evaluation_output['valid_detections'])
            
                logger.metric(f"Number of Valid Detections: {evaluation_output['valid_detections']}")
            
                logger.metric(f"Mean Top1 Scores Including 0 detections:{nms_health_metrics['mean_top1_incl0']}")
            
                logger.metric(f"Average Ground Truth Count Per Image: {ground_truth_health_metrics['avg_ground_truth_boxes_per_image']}")
                logger.metric(f"Ground Truth Top Classes: {ground_truth_health_metrics['top_gt_classes']}")
                logger.metric(f"Ground Truth Top Class Counts: {ground_truth_health_metrics['top_gt_class_counts']}")
            
                logger.metric(f"Ground Truth Boxes Min Coordinate : {gt_box_sanity['min_coordinates']}")
                logger.metric(f"Ground Truth Boxes Max Coordinate : {gt_box_sanity['max_coordinates']}")
            
                logger.metric(f"Pred Top Classes : {pred_health_metrics['top_classes']}")
                logger.metric(f"Pred Top Class Counts : {pred_health_metrics['top_class_counts']}")
            
                logger.metric(f"Pred Boxes Min Coordinate : {pred_box_sanity['min_coordinates']}")
                logger.metric(f"Pred Boxes Max Coordinate : {pred_box_sanity['max_coordinates']}")
            
            if (step + 1) % log_every == 0:
            
                nms_health_metrics = calculate_nms_health_scores(evaluation_output['pred_scores'], evaluation_output['valid_detections'])
                ground_truth_health_metrics = calculate_gt_health_scores(batch['boxes'], batch['labels'], batch['gt_mask'])
                pred_health_metrics = calculate_pred_health_metrics(evaluation_output['pred_scores'], evaluation_output['pred_classes'], evaluation_output['valid_detections'])
                pred_box_sanity = verify_pred_boxes_sanity(evaluation_output['pred_boxes'], evaluation_output['valid_detections'])
                gt_box_sanity = gt_box_range(evaluation_output['gt_boxes'],  evaluation_output['gt_mask'])
                iou_sanity = calculate_iou_sanity_top1(evaluation_output['pred_boxes'], evaluation_output['pred_scores'], evaluation_output['valid_detections'], evaluation_output['gt_boxes'],  evaluation_output['gt_mask']) 
                pred_bad_boxes_ratio = prediction_box_bad_frac(evaluation_output['pred_boxes'], evaluation_output['valid_detections'])
                ground_truth_bad_box_ratio = ground_truth_box_bad_frac(evaluation_output['gt_boxes'],  evaluation_output['gt_mask'])
            
                logger.metric(f"Min Validations In Batch: {np.min(evaluation_output['valid_detections'])}")
                logger.metric(f"Mean Validations In Batch: {np.mean(evaluation_output['valid_detections'])}")
                logger.metric(f"Max Validations In Batch: {np.max(evaluation_output['valid_detections'])}")
            

                # Model NMS health metrics
                logger.metric(f"Num of Valid Scores:{nms_health_metrics['num_valid']}")
                logger.metric(f"Min Valid Scores:{nms_health_metrics['min_valid']}")
                logger.metric(f"Mean Valid Scores:{nms_health_metrics['mean_valid']}")
                logger.metric(f"Max Valid Scores:{nms_health_metrics['max_valid']}")
                logger.metric(f"Num of Valid Scores < 0.9:{nms_health_metrics['below_thresh_scores']}")
                logger.metric(f"Average Valid Detections:{nms_health_metrics['average_valid_det']}")
                logger.metric(f"Zero Valid Detection Ratio:{nms_health_metrics['zero_valid_det']}")
                logger.metric(f"Mean Top1 Scores:{nms_health_metrics['mean_top1']}")
            
            
                logger.log_scalar(tag="val/nms_num_valid_scores", value= nms_health_metrics['num_valid'], step= step)
                logger.log_scalar(tag="val/nms_min_valid_scores", value= nms_health_metrics['min_valid'], step= step)
                logger.log_scalar(tag="val/nms_mean_valid_scores", value= nms_health_metrics['mean_valid'], step= step)
                logger.log_scalar(tag="val/nms_max_valid_scores", value= nms_health_metrics['max_valid'], step= step)
                logger.log_scalar(tag="val/nms_num_valid_scores_less_than_0.9", value= nms_health_metrics['below_thresh_scores'], step= step)
                logger.log_scalar(tag="val/nms_average_valid_detections", value= nms_health_metrics['average_valid_det'], step= step)
                logger.log_scalar(tag="val/nms_zero_valid_detections_ratio", value= nms_health_metrics['zero_valid_det'], step= step)
                logger.log_scalar(tag="val/nms_mean_top1_scores", value= nms_health_metrics['mean_top1'], step= step)
                logger.log_histogram(tag="val/nms_top1_scores_incl0_detections", values= nms_health_metrics['top1_incl0'], step= step)
                logger.log_scalar(tag="val/nms_mean_top1_scores_incl0_detections", value= nms_health_metrics['mean_top1_incl0'], step= step)

                # GT Metrics
                logger.metric(f"Ground Truth Count Per Image:{ground_truth_health_metrics['ground_truth_count']}")
                logger.metric(f"Zero Ground Truth Ratio Per Batch: {ground_truth_health_metrics['zero_ground_truth_ratio']}")



                logger.metric(f"Ground Truth Boxes Bad Box Ratio : {ground_truth_bad_box_ratio}")
                logger.log_histogram(tag="val/gt_top_class_dist", values = ground_truth_health_metrics['top_gt_class_distribution'], step=step)

                logger.log_histogram(tag="val/gt_count_per_image", values= ground_truth_health_metrics['ground_truth_count'], step= step)
                logger.log_scalar(tag="val/gt_avg_count_per_image", value= ground_truth_health_metrics['avg_ground_truth_boxes_per_image'], step= step)
                logger.log_scalar(tag="val/gt_zero_ratio_per_batch", value= ground_truth_health_metrics['zero_ground_truth_ratio'], step= step)
                # logger.log_scalar(tag="val/gt_top_classes", value= ground_truth_health_metrics['top_gt_classes'], step= step)
                # logger.log_histogram(tag="val/gt_top_classes_counts", value= ground_truth_health_metrics['top_gt_class_counts'])
                # logger.log_scalar(tag="val/gt_top_classes_counts", value= ground_truth_health_metrics['top_gt_class_counts'])

                # Pred Metrics
                logger.metric(f"Pred Boxes Bad Ratio : {pred_bad_boxes_ratio}")

                logger.log_histogram(tag="val/pred_top_class_dist", values = pred_health_metrics['top_class_distribution'], step=step)
                logger.log_scalar(tag="val/pred_boxes_bad_ratio", value= pred_bad_boxes_ratio, step= step)

                # IoU Metrics
                logger.metric(f"Mean Top1 IoU Only Detection:{iou_sanity['mean_iou_top1_only_det']}")
                logger.metric(f"Mean Top1 IoU Incl0:{iou_sanity['mean_iou_top1_incl0']}")


                # mAP Metrics
                logger.log_scalars(tag= "val",values= metrics_manager.compute(), step= step)
    
    
    inference_function(config= config, dataset_batch= batch, model_prediction= evaluation_output, logger= logger, global_step= 300)
    
    return metrics_manager.compute()    

def fit(config: dict[str,Any], model: tf.keras.Model, priors_cxcywh: tf.Tensor, train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset, optimizer: tf.keras.optimizers.Optimizer, precision_config: PrecisionConfig, metrics_manager: MetricsCollection, logger: Logger, checkpoint_manager: CheckpointManager, ema: EMA, amp: AMPContext, start_epoch: int = 0, global_step: int = 0, max_epochs: int | None = None, best_metric: float | None = None, shutdown_handler: ShutdownHandler = None, s3_sync: S3SyncClient = None):
    # Initialize overarching variables
    # 1. Epoch, 2. eval_every, 3. log_every, 4. heavy_log_every, 5. save_every, 6. save_best, 7. best_metric, 8. global_step
    epochs = max_epochs if max_epochs is not None else int(config['train']['epochs'])
    eval_every = int(config['train'].get('eval_every', 1))
    train_log_every = int(config['logging'].get('log_interval_steps', 10))
    eval_log_every = int(config['logging'].get('log_interval_steps', 10))

    if best_metric is None:
        best_metric = float("-inf")

    primary_metric = config['eval'].get('main_metric', 'voc_ap_50')

    logger.metric(f"Starting fit: epochs={epochs}, start_epoch={start_epoch}, global_step={global_step}")
    
    # Loop over the epochs:
        # Train one epoch
        # Log Training loss, learning_rate_at_epoch_end
        # Check if evaluate necessary:
            # Evaluate model
            # Log Metrics
            # Decide if to save best model
                # Save best
        # Save checkpoint at the end (model weights, optimizer state, epoch, global step, best_metric, EMA weights)
    try:
        
        for epoch in range(start_epoch, epochs):
        
            # Checking in with shutdown handler
            if shutdown_handler and shutdown_handler.is_requested():
                signal_number = shutdown_handler.signal_number or 15
            
                raise GracefulShutdownException(signal_number= signal_number)
        
            logger.metric(f"Epoch {epoch+1}/{epochs} starting")

            # Training over one epoch
            train_loss, global_step = train_one_epoch(config, epoch, model, train_dataset, optimizer, priors_cxcywh, precision_config, ema = ema, amp = amp, logger = logger, log_every= train_log_every, shutdown_handler= shutdown_handler)

            # Logging the scalar
            logger.log_scalar("train/loss_epoch_mean", float(train_loss.numpy()), step=global_step)

            if epoch % eval_every == 0 or epoch == epochs - 1:
                # Evaluate the model
                eval_metrics = evaluate(config, model, priors_cxcywh, validation_dataset, metrics_manager = metrics_manager, precision_config = precision_config, logger = logger,ema = ema, log_every= eval_log_every, shutdown_handler= shutdown_handler)
                logger.log_scalars(tag = "val", values= eval_metrics, step= global_step)

                # Checking for the best metric
                score = float(eval_metrics.get(primary_metric, float("-inf")))
                if score > best_metric:
                    best_metric = score
                    logger.metric(f"New Best {primary_metric}: {best_metric}")

                    if checkpoint_manager is not None:
                        save_path = checkpoint_manager.save_best(epoch= epoch, global_step= global_step, metric= best_metric)
                        
                        # Checking if the path exists or if there is a s3 client
                        if save_path and s3_sync:
                            # Saving the S3 sync
                            s3_sync.upload_directory(local_dir= save_path, s3_sub_prefix= str(checkpoint_manager.best_directory))

            # Checkpointing the last model at the end of the epoch
            if checkpoint_manager is not None:
                save_path = checkpoint_manager.save_last(epoch= epoch, global_step= global_step)
                
                # Checking if the path exists or if there is a s3 client
                if save_path and s3_sync:
                    # Saving the S3 sync
                    s3_sync.upload_directory(local_dir= save_path, s3_sub_prefix= str(checkpoint_manager.last_directory))
                

            # Logging the end of the model
            logger.metric(f"Epoch {epoch + 1} done. best_{primary_metric}={best_metric}")
            
    except GracefulShutdownException:
        logger.warning("Shutdown signal received, saving emergency checkpoint...")
        if checkpoint_manager is not None:
            checkpoint_manager.save_last(epoch= epoch, global_step= global_step)
            
        raise # Raising it again so it propagates to the top

    # Return Training Summary [final_epoch_metrics, best_metric, checkpoint_path]
    return {
        'best_metric': best_metric,
        'primary_metric': primary_metric,
        'global_step': global_step
    }   
    
