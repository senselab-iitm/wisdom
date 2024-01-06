import tf_lite_convert_utils as utils

import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot

from tensorflow_model_optimization.python.core.clustering.keras.experimental import cluster


def no_opt(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    return tflite_model


def pruning(model, X_train, Y_train, X_val, Y_val, epochs, model_folder, model_name):

    # Setting pruning parameters
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, begin_step=0, frequency=100)
    }
    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # Compiling the model
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model_for_pruning.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Fine Tuning
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
    ]
    batch_size = 16
    model_for_pruning.fit(
        X_train, Y_train,
        batch_size=batch_size, epochs=epochs,
        validation_data=(X_val, Y_val),
        callbacks=callbacks
    )

    # Saving the keras model
    stripped_pruned_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    export_dir = '{}/{}_sparse'.format(model_folder, model_name)
    stripped_pruned_model.save(export_dir)

    # Converting to TFLite model
    converter = tf.lite.TFLiteConverter.from_keras_model(stripped_pruned_model)
    converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
    tflite_model = converter.convert()

    return tflite_model


def clustering(model, X_train, Y_train, X_val, Y_val, epochs, model_folder, model_name):

    # Setting clustering parameters
    cluster_weights = tfmot.clustering.keras.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
    clustering_params = {
        'number_of_clusters': 2,
        'cluster_centroids_init': CentroidInitialization.KMEANS_PLUS_PLUS
    }
    clustered_model = cluster_weights(model, **clustering_params)

    # Compiling the model
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
    clustered_model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    # Fine Tuning
    batch_size = 16
    clustered_model.fit(
        X_train, Y_train,
        batch_size=batch_size, epochs=epochs,
        validation_data=(X_val, Y_val)
    )

    # Saving the keras model
    stripped_cluster_model = tfmot.clustering.keras.strip_clustering(clustered_model)
    export_dir = '{}/{}_clustered'.format(model_folder, model_name)
    stripped_cluster_model.save(export_dir)

    # Converting to TFLite model
    converter = tf.lite.TFLiteConverter.from_keras_model(stripped_cluster_model)
    tflite_model = converter.convert()

    return tflite_model


def quantization_aware_training(model, X_train, Y_train, X_val, Y_val, epochs, model_folder, model_name):

    # Setting QAT parameters
    annotated_model = tf.keras.models.clone_model(
        model,
        clone_function=utils.apply_quantization_to_non_bn,
    )
    quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    # Compiling the model
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
    quant_aware_model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Fine Tuning the model
    batch_size = 16
    quant_aware_model.fit(
        X_train, Y_train,
        batch_size=batch_size, epochs=epochs,
        validation_data=(X_val, Y_val)
    )

    # Export keras model
    export_dir = '{}/{}_qat'.format(model_folder, model_name)
    quant_aware_model.save(export_dir)

    # Convert to TFLite model
    converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_data_gen():
        for x in X_val:
            # yield [x]
            yield [np.reshape(x, (1, X_train.shape[1], X_train.shape[2], 1))]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    return tflite_model


def post_training_quantization(model, X_val):

    # Convert to TFLite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_data_gen():
        for x in X_val:
            # yield [x]
            yield [np.reshape(x, (1, X_val.shape[1], X_val.shape[2], 1))]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    return tflite_model


def sparsity_preserving_qat(model, X_train, Y_train, X_val, Y_val, qat, epochs, model_folder, model_name):

    # Setting prunning options
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, begin_step=0, frequency=100)
    }
    pruned_model = prune_low_magnitude(model, **pruning_params)

    # Compling purned model
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
    pruned_model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    # Fine tunning pruned model
    batch_size = 16
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep()
    ]
    pruned_model.fit(
        X_train, Y_train,
        batch_size=batch_size, epochs=epochs,
        validation_data=(X_val, Y_val),
        callbacks=callbacks
    )
    stripped_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

    if qat:
        # Setting sparsity aware quantization parameter
        annotated_model = tf.keras.models.clone_model(
            stripped_pruned_model,
            clone_function=utils.apply_quantization_to_non_bn,
        )
        pqat_model = tfmot.quantization.keras.quantize_apply(
            annotated_model,
            tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme()
        )

        # Compling pqat model
        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        pqat_model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )

        # Fine tuning pqat model
        batch_size = 16
        pqat_model.fit(
            X_train, Y_train,
            batch_size=batch_size, epochs=epochs,
            validation_data=(X_val, Y_val)
        )

        # Exporting keras pqat model
        export_dir = '{}/{}_sparse_qat'.format(model_folder, model_name)
        pqat_model.save(export_dir)
        converter = tf.lite.TFLiteConverter.from_keras_model(pqat_model)
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model(stripped_pruned_model)

    # Converting to TFLite model
    converter.optimizations = [tf.lite.Optimize.DEFAULT, tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
    def representative_data_gen():
        for x in X_val:
            yield [x]
            # yield [np.reshape(x, (1, X_train.shape[1], X_train.shape[2], 1))]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_model = converter.convert()

    return tflite_model


def cluster_preserving_qat(model, X_train, Y_train, X_val, Y_val, qat, epochs, model_folder, model_name):

    cluster_weights = tfmot.clustering.keras.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
    clustering_params = {
        'number_of_clusters': 2,
        'cluster_centroids_init': CentroidInitialization.KMEANS_PLUS_PLUS,
    }
    clustered_model = cluster_weights(model, **clustering_params)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
    clustered_model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    batch_size = 16
    clustered_model.fit(
        X_train, Y_train,
        batch_size=batch_size, epochs=epochs,
        validation_data=(X_val, Y_val)
    )

    stripped_clustered_model = tfmot.clustering.keras.strip_clustering(clustered_model)

    if qat:
        annotated_model = tf.keras.models.clone_model(
            stripped_clustered_model,
            clone_function=utils.apply_quantization_to_non_bn,
        )
        cqat_model = tfmot.quantization.keras.quantize_apply(
            annotated_model,
            tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme()
        )

        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        cqat_model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )

        batch_size = 16
        cqat_model.fit(
            X_train, Y_train,
            batch_size=batch_size, epochs=epochs,
            validation_data=(X_val, Y_val)
        )

        export_dir = '{}/{}_clustered_qat'.format(model_folder, model_name)
        cqat_model.save(export_dir)
        converter = tf.lite.TFLiteConverter.from_keras_model(cqat_model)
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model(stripped_clustered_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        for x in X_val:
            yield [x]
            # yield [np.reshape(x, (1, X_train.shape[1], X_train.shape[2], 1))]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_model = converter.convert()

    return tflite_model


def sparsity_preserving_clustering(model, X_train, Y_train, X_val, Y_val, epochs, model_folder, model_name):

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, begin_step=0, frequency=100)
    }
    pruned_model = prune_low_magnitude(model, **pruning_params)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
    pruned_model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    batch_size = 16
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep()
    ]
    pruned_model.fit(
        X_train, Y_train,
        batch_size=batch_size, epochs=epochs,
        validation_data=(X_val, Y_val),
        callbacks=callbacks
    )

    stripped_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

    cluster_weights = cluster.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
    clustering_params = {
        'number_of_clusters': 8,
        'cluster_centroids_init': CentroidInitialization.KMEANS_PLUS_PLUS,
        'preserve_sparsity': True
    }
    sparsity_clustered_model = cluster_weights(stripped_pruned_model, **clustering_params)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
    sparsity_clustered_model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    batch_size = 16
    sparsity_clustered_model.fit(
        X_train, Y_train,
        batch_size=batch_size, epochs=epochs,
        validation_data=(X_val, Y_val)
    )

    stripped_sparsity_clustered_model = tfmot.clustering.keras.strip_clustering(sparsity_clustered_model)

    export_dir = '{}/{}_sparse_clustered'.format(model_folder, model_name)
    stripped_sparsity_clustered_model.save(export_dir)

    converter = tf.lite.TFLiteConverter.from_keras_model(stripped_sparsity_clustered_model)
    converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
    tflite_model = converter.convert()

    return tflite_model


def cluster_and_sparsity_preserving_qat(model, X_train, Y_train, X_val, Y_val, qat, epochs, model_folder, model_name):

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, begin_step=0, frequency=100)
    }
    pruned_model = prune_low_magnitude(model, **pruning_params)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
    pruned_model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    batch_size = 16
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep()
    ]
    pruned_model.fit(
        X_train, Y_train,
        batch_size=batch_size, epochs=epochs,
        validation_data=(X_val, Y_val),
        callbacks=callbacks
    )

    stripped_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

    cluster_weights = cluster.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
    clustering_params = {
        'number_of_clusters': 8,
        'cluster_centroids_init': CentroidInitialization.KMEANS_PLUS_PLUS,
        'preserve_sparsity': True
    }
    sparsity_clustered_model = cluster_weights(stripped_pruned_model, **clustering_params)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
    sparsity_clustered_model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    batch_size = 16
    sparsity_clustered_model.fit(
        X_train, Y_train,
        batch_size=batch_size, epochs=epochs,
        validation_data=(X_val, Y_val)
    )

    stripped_sparse_clustered_model = tfmot.clustering.keras.strip_clustering(sparsity_clustered_model)

    if qat:
        annotated_model = tf.keras.models.clone_model(
            stripped_sparse_clustered_model,
            clone_function=utils.apply_quantization_to_non_bn,
        )
        pcqat_model = tfmot.quantization.keras.quantize_apply(
            annotated_model,
            tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme(preserve_sparsity=True)
        )

        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        pcqat_model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )

        batch_size = 16
        pcqat_model.fit(
            X_train, Y_train,
            batch_size=batch_size, epochs=epochs,
            validation_data=(X_val, Y_val)
        )

        export_dir = '{}/{}_sparse_clustered_qat'.format(model_folder, model_name)
        pcqat_model.save(export_dir)
        converter = tf.lite.TFLiteConverter.from_keras_model(pcqat_model)
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model(stripped_sparse_clustered_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT, tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
    def representative_data_gen():
        for x in X_val:
            yield [x]
            # yield [np.reshape(x, (1, X_train.shape[1], X_train.shape[2], 1))]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_model = converter.convert()

    return tflite_model
