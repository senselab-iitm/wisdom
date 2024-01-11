import tensorflow as tf
import tensorflow_model_optimization as tfmot


'''
:params
    - layer: a Keras layer

:returns
    - layer: returns the 'layer' given as param with quantization annotation, so that QAT is performed. But skips the annotation if the layer is Batch Normalization.
'''
def apply_quantization_to_non_bn(layer):
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer
