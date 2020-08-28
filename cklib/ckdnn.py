from __future__ import print_function

import keras
import os
import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def reset_keras():
    sess = keras.backend.get_session()
    sess.close()
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    keras.backend.set_session(tf.Session(config = cfg))
    
def get_session():
    gpu_fraction = 0.4
    visible_device_list = '0,1,2'
    
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction = gpu_fraction,
        visible_device_list = visible_device_list
    )
    
    return tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

class DNN:
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_layers,
        gpu_option = False,
        learning_rate = 1e-5,
        epochs = 20,
        batch_size = 1000,
        memory_fraction = 0.3,
        dropout = 0,
        verbose = False,
        random_state = None
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.memory_fraction = memory_fraction
        self.dropout = dropout
        self.verbose = verbose
        
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = self.memory_fraction)
        self.sess_config = tf.ConfigProto(gpu_options = self.gpu_options)
        
        self.sess = tf.Session(config = self.sess_config)
        keras.backend.set_session(self.sess)
        
        with self.sess.as_default():
            self.model = keras.models.Sequential()

            # Input layer
            self.model.add(
                keras.layers.Dense(
                    units = hidden_layers[0],
                    kernel_initializer = keras.initializers.glorot_uniform(seed = self.random_state),
                    input_shape = (num_features,),
                    activation = keras.activations.relu
                )
            )

            self.model.add(keras.layers.BatchNormalization())
            self.model.add(keras.layers.Dropout(rate = self.dropout, seed = self.random_state))

            # Hidden layers
            for i in range(1, len(hidden_layers)):
                self.model.add(
                    keras.layers.Dense(
                        units = self.hidden_layers[i],
                        kernel_initializer = keras.initializers.glorot_uniform(seed = self.random_state),
                        activation = keras.activations.relu
                    )
                )

                self.model.add(keras.layers.BatchNormalization())
                self.model.add(keras.layers.Dropout(rate = self.dropout, seed = self.random_state))

            # Output layer
            self.model.add(
                keras.layers.Dense(
                    units = num_classes,
                    kernel_initializer = keras.initializers.glorot_uniform(seed = self.random_state),
                    activation = keras.activations.softmax
                )
            )

            if gpu_option:
                self.model = keras.utils.multi_gpu_model(self.model, gpus = 4)
            
            self.model.compile(
                loss = "categorical_crossentropy",
                optimizer = keras.optimizers.Adam(self.learning_rate),
                metrics = ["accuracy"]
            )

            if self.verbose:
                self.model.summary()
            
        
    def fit(self, X, y):
        with self.sess.as_default():
            encoded_y = np.eye(self.num_classes)[y.reshape((-1))]

            if not os.path.isdir("./tmp"):
                os.makedirs(os.path.join("./tmp"))

            self.model.fit(X, encoded_y, epochs = self.epochs, batch_size = self.batch_size,
                callbacks = [
                    keras.callbacks.ModelCheckpoint(
                        "./tmp/keras_model_checkpoint",
                        monitor = "loss",
                        mode = "auto",
                        save_best_only = True
                    ),
                    keras.callbacks.EarlyStopping(monitor = "loss", mode = "auto", baseline = 200, patience = 5, verbose = self.verbose)
                ]
            )

            self.model = keras.models.load_model("./tmp/keras_model_checkpoint")
        
    def predict(self, X):
        with self.sess.as_default():
            return self.model.predict_classes(X)
    
    def predict_proba(self, X):
        with self.sess.as_default():
            return self.model.predict_proba(X)
    
    def close(self):
        self.sess.close()