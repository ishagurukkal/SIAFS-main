import numpy as np
import tensorflow as tf
import cv2

def compute_gradcam(model, img_array, layer_name='out_relu'):
    """
    Computes a forensic Grad-CAM heatmap highlighting generative or tampering artifacts.
    Handles nested backbones and prevents zero-gradient collapse.
    """
    # 1. Resolve layer reference across flat or nested backbones
    target_layer = None
    try:
        target_layer = model.get_layer(layer_name)
    except (ValueError, AttributeError):
        # Search inside nested sub-models (e.g., base_model within the classifier)
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                try:
                    target_layer = layer.get_layer(layer_name)
                    break
                except ValueError:
                    continue

    # Fallback: dynamically bind to the deepest 4D convolutional feature tensor
    if target_layer is None:
        for layer in reversed(model.layers):
            if len(layer.output.shape) == 4:
                target_layer = layer
                break

    # 2. Build grad model (pass model.inputs directly, no outer list brackets)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    # 3. Compute gradients directed at generative anomaly features
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        score = preds[:, 0]
        # If the image is leaning Fake (score < 0.54), drive gradients toward the fake class
        loss = tf.where(score < 0.54, 1.0 - score, score)

    grads = tape.gradient(loss, conv_outputs)
    
    # Safety check: if gradients are vanishing, prevent division by zero
    if grads is None:
        return np.zeros((img_array.shape[1], img_array.shape[2]), dtype=np.float32)

    # 4. Global average pooling of gradients across spatial dims (H, W)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Weight feature maps by gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Apply ReLU and normalize between 0.0 and 1.0
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    
    if max_val > 0:
        heatmap = heatmap / max_val
    else:
        heatmap = tf.zeros_like(heatmap)

    return heatmap.numpy()

def overlay_heatmap(heatmap, original_img, alpha=0.4):
    """
    Resizes the heatmap to match original image dimensions and creates a smooth overlay.
    """
    if original_img is None:
        return None

    h, w = original_img.shape[:2]
    
    # Resize activation map directly to the original upload resolution
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255.0 * heatmap_resized)
    
    # Convert single-channel intensity into JET color spectrum
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Blend overlay with authentic photo
    output = cv2.addWeighted(original_img, 1.0 - alpha, heatmap_color, alpha, 0)
    return output