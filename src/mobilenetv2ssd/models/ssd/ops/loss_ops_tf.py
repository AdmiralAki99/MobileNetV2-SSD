import tensorflow as tf

from mobilenetv2ssd.core.precision_config import PrecisionConfig, should_force_fp32

def smooth_l1_loss(predicted_values, target, beta, reduction: str = "sum"):

    # Calculate the difference between the two values
    difference = predicted_values - target
    absolute_difference = tf.math.abs(difference)

    # Masking which values require L1 and L2
    small_mask = absolute_difference < beta
    large_mask = tf.logical_not(small_mask)
    
    # Calculate where the formula needs to change
    errors = tf.where(small_mask, 0.5*(difference**2)/beta,  tf.zeros_like(difference))
    errors = tf.where(large_mask, absolute_difference - (0.5*beta),errors)

    # Sum over the four coordinates
    errors = tf.reduce_sum(errors,axis=-1)

    # Reduction strategy
    if reduction == "sum":
        loss = tf.reduce_sum(errors)
    elif reduction == "max":
        loss = tf.reduce_max(errors)
    elif reduction == "mean":
        loss = tf.reduce_mean(errors)
    else:
        loss = errors
    
    return loss

def l1_loss(predicted_values,target,reduction: str = "sum"):

    # Calculate the difference between the pred and the actual values
    difference = predicted_values - target
    absolute_difference = tf.math.abs(difference)

    # Sum over the four coordinates
    errors = tf.reduce_sum(absolute_difference,axis=-1)

    # Reduction strategy
    if reduction == "sum":
        loss = tf.reduce_sum(errors)
    elif reduction == "max":
        loss = tf.reduce_max(errors)
    elif reduction == "mean":
        loss = tf.reduce_mean(errors)
    else:
        loss = errors
    
    return loss

def l2_loss(predicted_values, target, reduction= "sum"):

    # Calculate the difference between the pred and the actual values
    difference = predicted_values - target
    squared_difference = tf.square(difference)

    # Sum over the four coordinates
    errors = tf.reduce_sum(squared_difference,axis=-1)

    # Reduction strategy
    if reduction == "sum":
        loss = tf.reduce_sum(errors)
    elif reduction == "max":
        loss = tf.reduce_max(errors)
    elif reduction == "mean":
        loss = tf.reduce_mean(errors)
    else:
        loss = errors
    
    return loss

def softmax_cross_entropy_loss(logits: tf.Tensor, labels: tf.Tensor, reduction: str ="none"):

    logits = tf.cast(logits, tf.float32)
    labels = tf.cast(labels, tf.int32)

    # Calculating the shifted values
    max_per_row = tf.reduce_max(logits, axis=-1,keepdims = True)

    # Shifted logits from stopping exponential values from going extremely large
    shifted_logits = logits - max_per_row
    shifted_logits = tf.cast(shifted_logits,tf.float32)
    
    # Exponential Values
    exponential_shifted_logits = tf.math.exp(shifted_logits)
    sum_exponential_shifted_logits = tf.reduce_sum(exponential_shifted_logits, axis= -1, keepdims = True)

    # Calculating probabilities
    log_probabilities = tf.math.log(sum_exponential_shifted_logits)
    log_probabilities = shifted_logits - log_probabilities

    # Calculating for the labels
    rows = tf.range(tf.shape(logits)[0])
    index = tf.stack([rows,labels], axis=1)
    probs = tf.gather_nd(log_probabilities, index)
    per_class_loss = -probs

    if reduction == "sum":
        return tf.reduce_sum(per_class_loss)
    elif reduction == "mean":
        num = tf.cast(tf.size(per_class_loss),tf.float32)
        sum_probs = tf.reduce_sum(per_class_loss)
        return tf.math.divide_no_nan(sum_probs, num)
    else:
        return per_class_loss
    
def multibox_loss(predicted_offsets: tf.Tensor, predicted_logits: tf.Tensor, target_offsets: tf.Tensor, target_labels: tf.Tensor, positive_mask: tf.Tensor, negative_mask: tf.Tensor, localization_weight: float, classification_weight: float,beta: float|None ,cls_loss_type: str ="softmax_ce", loc_loss_type: str = "smooth_l1", normalize_denom: str = "num_pos", reduction: str = "sum", precision_config: PrecisionConfig | None = None):
    # Calculate the mask for classification of anchors
    classification_mask = tf.logical_or(positive_mask,negative_mask)

    # Calculate the number of positives and number of negative boxes
    number_of_positives = tf.reduce_sum(tf.cast(positive_mask,tf.int32))
    number_of_negatives = tf.reduce_sum(tf.cast(negative_mask,tf.int32))

    # Calculating Safe values
    raw_number_of_positive = number_of_positives
    raw_number_of_negative = number_of_negatives
    number_of_positives = tf.maximum(1,number_of_positives)
    number_of_negatives = tf.maximum(1,number_of_negatives)
    number_of_classifications = number_of_positives + number_of_negatives

    # Flattening the masks
    B = tf.shape(positive_mask)[0]
    N = tf.shape(positive_mask)[-1]

    positive_mask_flattened = tf.reshape(positive_mask,[-1])
    negative_mask_flattened = tf.reshape(negative_mask,[-1])
    classification_mask_flattened = tf.reshape(classification_mask, [-1])

    # Flattened Offsets
    predicted_offsets_flattened = tf.reshape(predicted_offsets,[-1,4])
    target_offsets_flattened = tf.reshape(target_offsets,[-1,4])

    # Flattening the Logits
    C = tf.shape(predicted_logits)[-1]
    predicted_logits_flattened = tf.reshape(predicted_logits,[-1,C])

    # Flattening the Labels
    labels_flattened = tf.reshape(target_labels,[-1])

    # Masking the offsets
    positive_offsets_flattened = tf.boolean_mask(predicted_offsets_flattened,positive_mask_flattened)
    negative_offsets_flattened = tf.boolean_mask(predicted_offsets_flattened,negative_mask_flattened)
    positive_targets_flattened = tf.boolean_mask(target_offsets_flattened,positive_mask_flattened)

    # Masking the logits
    positive_logits_flattened = tf.boolean_mask(predicted_logits_flattened, positive_mask_flattened)
    negative_logits_flattened = tf.boolean_mask(predicted_logits_flattened, negative_mask_flattened)
    
    # Selecting the anchors that are used in the classification task
    selected_prediction_logits = tf.boolean_mask(predicted_logits_flattened,classification_mask_flattened)
    selected_prediction_targets = tf.boolean_mask(labels_flattened,classification_mask_flattened)

    # Calculating the losses for the model (Localization + Classification)

    # The classification loss looks ath both the positive and negative anchors in the model
    if cls_loss_type == "softmax_ce":
        classification_raw = softmax_cross_entropy_loss(selected_prediction_logits,selected_prediction_targets,reduction=reduction)

    
    # The localization loss only looks at the positive 
    if loc_loss_type == "smooth_l1":
        if beta != None:
            if should_force_fp32("loss_reduction", precision_config):
                positive_offsets_flattened = tf.cast(positive_offsets_flattened, tf.float32)
                positive_targets_flattened = tf.cast(positive_targets_flattened, tf.float32)
                
            localization_raw = smooth_l1_loss(positive_offsets_flattened,positive_targets_flattened,beta = beta, reduction=reduction)
        else:
            if should_force_fp32("loss_reduction", precision_config):
                positive_offsets_flattened = tf.cast(positive_offsets_flattened, tf.float32)
                positive_targets_flattened = tf.cast(positive_targets_flattened, tf.float32)
                
            localization_raw = smooth_l1_loss(positive_offsets_flattened,positive_targets_flattened,beta = 1.0, reduction=reduction)
        
    elif loc_loss_type == "l1_loss":
        if should_force_fp32("loss_reduction", precision_config):
            positive_offsets_flattened = tf.cast(positive_offsets_flattened, tf.float32)
            positive_targets_flattened = tf.cast(positive_targets_flattened, tf.float32)
            
        localization_raw = l1_loss(positive_offsets_flattened,positive_targets_flattened,reduction=reduction)
    elif loc_loss_type == "l2_loss":
        if should_force_fp32("loss_reduction", precision_config):
            positive_offsets_flattened = tf.cast(positive_offsets_flattened, tf.float32)
            positive_targets_flattened = tf.cast(positive_targets_flattened, tf.float32)
            
        localization_raw = l2_loss(positive_offsets_flattened,positive_targets_flattened,reduction=reduction)

    # Normalize the losses
    # Localization loss looks at the number of positives
    if normalize_denom == "num_neg":
        localization_loss = tf.math.divide(localization_raw,tf.cast(number_of_negatives,dtype=tf.float32))
        classification_loss = tf.math.divide(classification_raw,tf.cast(number_of_negatives,dtype=tf.float32))
    elif normalize_denom == "num_cls":
        localization_loss = tf.math.divide(localization_raw,tf.cast(number_of_classifications,dtype=tf.float32))
        classification_loss = tf.math.divide(classification_raw,tf.cast(number_of_classifications,dtype=tf.float32))
    elif normalize_denom == "num_batch":
        localization_loss = tf.math.divide(localization_raw,tf.cast(B,dtype=tf.float32))
        classification_loss = tf.math.divide(classification_raw,tf.cast(B,dtype=tf.float32))
    else:
        localization_loss = tf.math.divide(localization_raw,tf.cast(number_of_positives,dtype=tf.float32))
        classification_loss = tf.math.divide(classification_raw,tf.cast(number_of_positives,dtype=tf.float32))

    # Adding the losses using the weights
    multibox_loss = (localization_weight * localization_loss) + (classification_weight * classification_loss)
    
    return {
        'total_loss': multibox_loss,
        'loc_loss': localization_loss,
        'cls_loss': classification_loss,
        'num_pos': number_of_positives,
        'raw_num_pos': raw_number_of_positive,
        'raw_num_negative': raw_number_of_negative,
        'num_negative': number_of_negatives
    }
    