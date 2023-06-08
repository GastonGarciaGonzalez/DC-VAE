# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 07:03:05 2022

@author: gastong@fing.edu.uy
"""



import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv1D, BatchNormalization, Lambda, Cropping1D, Reshape, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from prts import ts_precision, ts_recall
import pickle
import gc


@keras.utils.register_keras_serializable()
class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self, name=None, k=1, **kwargs):
        super(Sampling, self).__init__(name=name)
        self.k = k
        super(Sampling, self).__init__(**kwargs)
        
    def get_config(self):
        config = super(Sampling, self).get_config()
        config['k'] = self.k
        return config #dict(list(config.items()))
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        seq = K.shape(z_mean)[1]
        dim = K.shape(z_mean)[2]
        epsilon = K.random_normal(shape=(batch, seq, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class DCVAE:
    
    def __init__(self,
                 T=32,
                 M=12,
                 cnn_units = [32, 16, 1],
                 dil_rate = [1,8,16],
                 kernel=2,
                 strs=1,
                 batch_size=32,
                 J=1,
                 epochs=100,
                 learning_rate=1e-3,
                 lr_decay=True,
                 decay_rate=0.96,
                 decay_step=1000,
                 name = '',
                 summary=False,
                 ):
        
        
        # network parameters
        input_shape = (T, M)
        self.M = M
        self.T = T
        self.J = J
        self.batch_size = batch_size
        self.epochs = epochs  
        self.name = name
        

        # model = encoder + decoder
        
        # Build encoder model
	    # =============================================================================
        # Input
        inputs = Input(shape=input_shape, name='input')
        h_enc_cnn = inputs
        
        # Hidden layers (1D Dilated Convolution)
        for i in range(len(cnn_units)):
            h_enc_cnn = Conv1D(cnn_units[i], kernel, activation='selu', use_bias=False,
                           strides=strs, padding="causal",
                           dilation_rate=dil_rate[i], name='dcnn_enc_%d'%(i))(h_enc_cnn)
            
        # Lastest    
        z_mean = Conv1D(J, 1, activation=None, use_bias=False,
                           strides=strs, padding="causal", name='z_mean')(h_enc_cnn)
        z_log_var = Conv1D(J, 1, activation=None, use_bias=False,
                           strides=strs, padding="causal", name='z_log_var')(h_enc_cnn)

        z_mean = Cropping1D(cropping=(self.T-1, 0))(z_mean)
        z_log_var = Cropping1D(cropping=(self.T-1, 0))(z_log_var)       

        # Reparameterization trick 
        # Output
        z = Sampling(name='z')((z_mean, z_log_var))
        # Instantiate encoder model
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        if summary:
            self.encoder.summary() 
        # =============================================================================

        # Build decoder model
        # =============================================================================
        # Input
        latent = Input(shape=(1, J), name='z_sampling_z')

        # Hidden layers (1D Dilated Convolution)
        latent_reshape = Reshape((J, ), input_shape=(1, J))(latent)
        repeat_z = RepeatVector(T)(latent_reshape)
        h_dec_cnn = repeat_z
        
        # Hidden layers (1D Dilated Convolution)
        for i in range(len(cnn_units)):
            h_dec_cnn = Conv1D(cnn_units[i], kernel, activation='selu', use_bias=False,
                           strides=strs, padding="causal", 
                           dilation_rate=dil_rate[i], name='dcnn_dec_%d'%i)(h_dec_cnn)
            
        # Lastest/Output
        x__mean = Conv1D(M, 1, activation=None, use_bias=False,
                                  padding="causal",
                                  name='x__mean_output')(h_dec_cnn)
        x_log_var = Conv1D(M, 1, activation=None, use_bias=False,
                                  padding="causal", 
                                  name='x_log_var_output')(h_dec_cnn)

        # Instantiate decoder model
        self.decoder = Model(latent, [x__mean, x_log_var], name='decoder')
        if summary:
            self.decoder.summary()
        # =============================================================================

        # Instantiate DC-VAE model
        # =============================================================================
        [x__mean, x_log_var] = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, [x__mean, x_log_var], name='vae')
        
        # Loss
        # Reconstruction term
        MSE = -0.5*K.mean(K.square((inputs - x__mean)/K.exp(x_log_var)),axis=-1) #Mean in M
        sigma_trace = -K.mean(x_log_var, axis=(-1)) #Mean in M
        log_likelihood = MSE+sigma_trace
        reconstruction_loss = K.mean(-log_likelihood) #Mean in the batch and T   
       
        # Priori hypothesis term
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.mean(kl_loss, axis=-1) #Mean in J
        kl_loss *= -0.5
        kl_loss = tf.reduce_mean(kl_loss) #Mean in the batch and T
        
        # Total
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        
        # Learning rate
        if lr_decay: 
            lr = optimizers.schedules.ExponentialDecay(learning_rate,
                                                    decay_steps=decay_step,
                                                    decay_rate=decay_rate,
                                                    staircase=True,
                                                    )
        else:
            lr = learning_rate

        # Optimaizer
        opt = optimizers.Adam(learning_rate=lr)

        # Metrics
        self.vae.add_metric(reconstruction_loss, name='reconst')
        self.vae.add_metric(kl_loss, name='kl')


        self.vae.compile(optimizer=opt)


    def fit(self, df_X=None, val_percent=0.1, seed=42):
        # Data
        X = df_X.values
        N = X.shape[0]

        # Samples: [N-T+1, T, M]
        X = np.array([X[i: i + self.T] for i in range(0, N - self.T+1)])

        # Random disorden of the samples
        rand_idx = np.random.permutation(X.shape[0])
        X = X[rand_idx]               
        
        # Callbacks
        early_stopping_cb = keras.callbacks.EarlyStopping(min_delta=1e-2,
                                                      patience=10,                                            
                                                      verbose=1,
                                                      mode='min')
        model_checkpoint_cb= keras.callbacks.ModelCheckpoint(
            filepath=self.name+'_best_model.h5',
            verbose=1,
            mode='min',
            save_best_only=True)
        
          
        # Model train
        self.history_ = self.vae.fit(X,
                     batch_size=self.batch_size,
                     epochs=self.epochs,
                     validation_split = val_percent,
                     callbacks=[early_stopping_cb,
                                model_checkpoint_cb]
                     )  
        
        # Save models
        self.encoder.save(self.name+'_encoder.h5')
        self.decoder.save(self.name+'_decoder.h5')
        self.vae.save(self.name+'_complete.h5')

        return self

         
    def predict(self,
                df_X=None, 
                load_model=False,
                large_result=True,
                load_alpha=True,
                alpha_set_up=[],
                alpha_set_down=[]):
        
        # Trained model
        if load_model:
            self.vae = keras.models.load_model(self.name+'_best_model.h5',
                                                    custom_objects={'sampling': Sampling},
                                                    compile = False)
            self.encoder = keras.models.load_model(self.name+'_encoder.h5',
                                                    custom_objects={'sampling': Sampling},
                                                    compile = False)
        
        # Inference model. Auxiliary model so that in the inference 
        # the prediction is only the last value of the sequence
        inp = Input(shape=(self.T, self.M))
        x = self.vae(inp) # apply trained model on the input
        out = Lambda(lambda y: [y[0][:,-1,:], y[1][:,-1,:]])(x)
        inference_model = Model(inp, out)
        
        
        # Data preprocess
        X = df_X.values
        N = df_X.shape[0]
        
        # Samples: [N-T+1, T, M]
        X = np.array([X[i: i + self.T] for i in range(0, N - self.T+1)])

        # Predictions
        prediction = inference_model.predict(X)
        mean_predict = prediction[0]
        sig_predict = np.sqrt(np.exp(prediction[1]))
        
        # Data evaluate (The first T-1 data are discarded)
        df_X_eval = df_X[self.T-1:]

        df_mean = pd.DataFrame(mean_predict, columns=df_X.columns, index=df_X.iloc[self.T-1:].index)
        df_sig = pd.DataFrame(sig_predict, columns=df_X.columns, index=df_X.iloc[self.T-1:].index)

        # Thresholds
        if len(alpha_set_up) == self.M:
            alpha_up = np.array(alpha_set_up)
        elif load_alpha:
            with open(self.name + '_alpha_up.pkl', 'rb') as f:
                alpha_up = pickle.load(f)
                f.close()
        else:
            alpha_up = self.alpha_up
            
        if len(alpha_set_down) == self.M:
            alpha_down = np.array(alpha_set_down)
        elif load_alpha:
            with open(self.name + '_alpha_down.pkl', 'rb') as f:
                alpha_down = pickle.load(f)
                f.close()
        else:
            alpha_down = self.alpha_down

        thdown = df_mean - alpha_down*df_sig
        thup = df_mean + alpha_up*df_sig
        
        # Evaluation
        df_anom_result = (df_X_eval < thdown) | (df_X_eval > thup)
        
        if large_result:
            df_score = -0.5*((df_X_eval - df_mean)**2)/(df_sig**2) - np.log(df_sig**2)
            latent_space = self.encoder.predict(X)[2]
            latent_space = latent_space[:,-1,:]
            df_latent_space = pd.DataFrame(latent_space, columns=np.arange(latent_space.shape[-1]), index=df_X.iloc[self.T-1:].index)
            return df_anom_result, df_score, df_mean, df_sig, df_latent_space
        else: 
            return df_anom_result



    def evaluate(self, load_model=False, df_X=None, seed=42):
        # Data
        X = df_X.values
        N = df_X.shape[0]
        
        # Samples: [N-T+1, T, M]
        X = np.array([X[i: i + self.T] for i in range(0, N - self.T+1)])

        # Trained model
        if load_model:
            self.vae = keras.models.load_model(self.name+'_complete.h5',
                                                    custom_objects={'sampling': Sampling},
                                                    compile = True)

        # Model evaluate
        value_elbo, reconstruction, kl = self.vae.evaluate(X,
                     batch_size=self.batch_size,
                     )  
        
        return value_elbo, reconstruction, kl
