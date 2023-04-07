import numpy as np
import tensorflow as tf
import cvxpy as cvx
from tensorflow import keras 
from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

from keras.layers.core import Dense, Dropout, Activation
import keras.callbacks as callbacks
from keras.utils import np_utils

import joblib


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

def auc_score(y_true,y_pred):
    return roc_auc_score(y_true, y_pred)

def sensitivity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp+fn)
    return sensitivity

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    return specificity

def sp_index(y_true, y_pred):
    spec = specificity_score(y_true, y_pred)
    sens = sensitivity_score(y_true, y_pred)
    return np.sqrt(spec*sens)*np.mean([spec, sens])

def acc_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def get_confusion_matrix(y_true, y_pred, class_labels=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, normalize='true').ravel() # only in binary case
    return (tn, fp, fn, tp)


class MLPClassificationModel:
    def __init__(self, n_hidden_neurons=2, verbose=2):
        self.n_hidden_neurons = n_hidden_neurons
        self.model = None
        self.trn_history = None
        self.trained = False
        self.verbose = verbose
    def __str__(self):
        m_str = 'Class MLPModel\n'
        if self.trained:
            m_str += 'Model is fitted, '
        else:
            m_str += 'Model is not fitted, '
        m_str += 'instance created with %i hidden neurons'%(self.n_hidden_neurons) 
        return m_str
    def model_loss(self, loss_alg='cat_crossent'):
        if loss_alg == 'cat_crossent':
            loss = keras.losses.CategoricalCrossentropy(from_logits=False,
                                                        label_smoothing=0.0,
                                                        axis=-1,
                                                        reduction="auto",
                                                        name=loss_alg,)
        elif loss_alg== 'mse':
            loss = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        elif loss_alg=='bi_crossent':
            loss =  keras.losses.BinaryCrossentropy(from_logits=False,
            label_smoothing=0.0,axis=-1,
            reduction="auto",
            name="loss_alg")

        return loss
        
    def model_optimizer(self, optimizer='adam', learning_rate = 0.001):
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate,
                                        beta_1=0.9,beta_2=0.999,
                                        epsilon=1e-07,amsgrad=False,
                                        name="Adam",)
        return opt
    
    def create_model(self, data, target, random_state=0, learning_rate=0.01):
        #tf.random.set_seed(random_state)

        model = tf.keras.Sequential()
        
        # add a input to isolate the input of NN model
        model.add(tf.keras.Input(shape=(data.shape[1],)))
        # add a non-linear single neuron layer
        hidden_layer = layers.Dense(units=self.n_hidden_neurons,
                                    activation='tanh',
                                    kernel_initializer=initializers.RandomNormal(stddev=0.01),
                                    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                                    bias_regularizer=regularizers.L2(1e-4),
                                    bias_initializer=initializers.Zeros()
                                   )
        model.add(hidden_layer)
        # add a non-linear output layer with max sparse target shape
        output_layer = layers.Dense(units=target.shape[1],
                                    activation='tanh',
                                    kernel_initializer=initializers.RandomNormal(stddev=0.01),
                                    bias_initializer=initializers.Zeros()
                                   )
        model.add(output_layer)
        # creating a optimization function using steepest gradient
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                                  decay_steps=100,
                                                                  decay_rate=0.9)
        
        # model optimizer
        optimizer = self.model_optimizer(optimizer='adam', learning_rate = learning_rate)
        # model loss
        loss = self.model_loss(loss_alg='mse')
       
        cat_acc_metric = keras.metrics.CategoricalAccuracy(name="cat_acc", dtype=None)
        acc_metric = keras.metrics.Accuracy(name="accuracy",dtype=None)
        mse_metric = keras.metrics.MeanSquaredError(name="mse", dtype=None)
        rmse_metric = keras.metrics.RootMeanSquaredError(name="rmse", dtype=None)

        model.compile(loss=loss, 
                      optimizer=optimizer,
                      metrics=[cat_acc_metric,
                               acc_metric,
                               mse_metric,
                               rmse_metric])
        return model
    def fit(self, X, Y,
            trn_id=None, 
            val_id=None, 
            epochs=50,
            batch_size=4,
            patience = 100,
            learning_rate=0.01, random_state=0):
        
        X_copy = X.copy()
        Y_copy = Y.copy()
        
        model = self.create_model(X_copy,Y_copy, random_state=random_state, learning_rate=learning_rate)
        
        # early stopping to avoid overtraining
        earlyStopping = callbacks.EarlyStopping(monitor='val_loss', 
                                                patience=patience,verbose=self.verbose, 
                                                mode='auto')
    
        trn_desc = model.fit(X_copy[trn_id,:], Y_copy[trn_id],
                             epochs=epochs,
                             batch_size=batch_size,
                             callbacks=[earlyStopping], 
                             verbose=self.verbose,
                             validation_data=(X_copy[val_id,:],
                                              Y_copy[val_id]),
                            )
        self.model = model
        self.trn_history = trn_desc
        self.trained = True
    def predict(self, data):
        return self.model.predict(data)
    def save(self, file_path):
        with open(file_path,'wb') as file_handler:
            joblib.dump([self.n_hidden_neurons, self.model,
                        self.trn_history, self.trained], file_handler)
    def load(self, file_path):
        with open(file_path,'rb') as file_handler:
            [self.n_hidden_neurons, self.model, self.trn_history, self.trained]= joblib.load(file_handler)
    def model_with_no_output_layer(self):
        buffer_model = tf.keras.Sequential()    
        # add a input to isolate the input of NN model
        buffer_model.add(tf.keras.Input(shape=(self.model.layers[0].get_weights()[0].shape[0],)))
        # add a non-linear single neuron layer
        hidden_layer = layers.Dense(units=self.model.layers[0].get_weights()[1].shape[0],
                                    activation='tanh')
        buffer_model.add(hidden_layer)    
        output_layer = layers.Dense(units=1,activation='tanh')
    
        for idx, layer in enumerate(buffer_model.layers):
            layer.set_weights(self.model.layers[idx].get_weights())
        return buffer_model
    def predict_one_layer_before_output(self, data):
        buffer_model = self.model_with_no_output_layer()
        return buffer_model.predict(data)

from sklearn.svm import SVC

class SVMClassificationModel:
    def __init__(self, kernel="linear", regularization=0.5, verbose=2):
        self.model = None
        self.trn_history = None
        self.trained = False
        self.verbose = verbose
        self.kernel = kernel
        self.regularization = regularization

    def __str__(self):
        m_str = 'Class SVMClassificationModel\n'
        if self.trained:
            m_str += 'Model is fitted, \n'
        else:
            m_str += 'Model is not fitted, \n'
        m_str += "Model created with %s of regularization and kernel %s"%(self.regularization, self.kernel)
        return m_str
    
    def create_model(self, class_weight=None, random_state=0):
        return SVC(C=self.regularization,
                   kernel=self.kernel, 
                   random_state=random_state,
                   verbose=self.verbose)
    
    def fit(self, X, Y,
            trn_id=None, 
            val_id=None, 
            random_state=0):
        model = self.create_model(random_state=random_state)
        if trn_id is None:
            model.fit(X,Y)
        else:
            model.fit(X[trn_id], Y[trn_id])
        self.model = model
        self.trained = True
        
    def predict(self, data):
        return self.model.predict(data)
    
    def save(self, file_path):
        with open(file_path,'wb') as file_handler:
            joblib.dump([self.regularization, self.kernel, self.model, self.trn_history, self.trained], file_handler)

    def load(self, file_path):
        with open(file_path,'rb') as file_handler:
            [self.regularization, self.kernel, self.model, self.trn_history, self.trained]= joblib.load(file_handler)


class KernelClassifier:
    def __init__(self, verbose = 2, kernel='polynomial', loss='logistic', kernel_arg=1, loss_arg=1):
        self.trained = False
        self.verbose = verbose
        self.kernel = kernel
        self.loss = loss
        self.params = None
        self.kernel_arg = kernel_arg
        self.loss_arg = loss_arg
        self.convert_output = False
        self.ravel_output = False

    def __str__(self):
        m_str = 'Class KernelClassifier\n'
        if self.trained:
            m_str += 'Model is fitted, \n'
        else:
            m_str += 'Model is not fitted, \n'
        m_str += "Model created with kernel " + self.kernel + " and " + self.loss + " as loss function"
        return m_str

    def transform(self, X):
        if self.kernel == 'polynomial':
            #k(x,x) = (x'x + c)^d
            #for d = 2 and c = self.kernel_arg
            #k(x,x) = <xn^2,.. ,x1^2, sqrt(2)*xn*xn-1, ..., sqrt(2)*xn*xn1, sqrt(2)*xn-1*xn-2, ..., sqrt(2)*xn-1*x1..., sqrt(2)*x2*x1, ..., sqrt(2c)*xn, ..., sqrt(2c)*x1, c>
            Z = [np.ones([X.shape[0],])*(self.kernel_arg**2)] #c

            for i in range(X.shape[1]):
                Z = np.vstack([Z, np.power(X[:,i],2)]) # xn^2
                Z = np.vstack([Z, X[:,i] * np.sqrt(2*self.kernel_arg)]) #sqrt(2c)*xn

            for i in range(1,X.shape[1]):
                for j in range(0,i):
                    Z = np.vstack([Z, np.sqrt(2) * np.multiply(X[:,i], X[:,j])]) #sqrt(2)*xn*xn1

            return Z.T

        raise NotImplementedError("transforming by kernel " + self.kernel)

    def fit(self, X, Y, trn_id=None, val_id=None, random_state=0):
        if trn_id is not None:
            X = X[trn_id]
            Y = Y[trn_id]

        if min(Y) == 0: # regressions operating with classes -1 and 1 not 0 and 1
            Y=Y*2-1
            self.convert_output = True

        Z = self.transform(X)
        w = cvx.Variable([Z.shape[1], 1])

        if len(Y.shape) == 1: # input format (n_samples,) required in format (n_samples,1)
            Y = np.reshape(Y, (Y.shape[0], 1))
            self.ravel_output = True


        #https://www.cvxpy.org/tutorial/functions/index.html
        #classification loss functions
        if self.loss == 'logistic':
            obj = cvx.Minimize(cvx.sum(cvx.logistic(-cvx.multiply(Y,Z@w)))) # cvx.logistic(x) = log(1 + e^x)
        elif self.loss == 'soft_margin':
            obj = cvx.Minimize(cvx.sum(cvx.pos(1-cvx.multiply(Y,Z@w)))) # cvx.pos(x) = max{0,x}
        elif self.loss == 'quadratic_soft_margin':
            obj = cvx.Minimize(cvx.sum(cvx.power(cvx.pos(1-cvx.multiply(Y,Z@w)), 2)))
        #regression loss functions
        elif self.loss == 'squared_loss':
            obj = cvx.Minimize(cvx.sum(cvx.power(Y-Z@w, 2)))
        elif self.loss == 'e-insensitive':
            obj = cvx.Minimize(cvx.sum(cvx.pos(cvx.abs(Y-Z@w) - self.loss_arg)))
        elif self.loss == 'huber':
            obj = cvx.Minimize(cvx.sum(cvx.huber(Y-Z@w, self.loss_arg)))
        else:
            raise NotImplementedError("fitting for " + self.loss + " as loss function")

        prob = cvx.Problem(obj).solve()
        self.params = w.value
        self.trained = True
        
    def predict(self, X):
        if not self.trained:
            raise UnboundLocalError("it is not possible to predict the data without training")

        if self.kernel == 'polynomial':
            Z = self.transform(X)
            YL = np.sign(Z@self.params)
            if self.convert_output:
                YL = (YL+1)/2

            if self.ravel_output:
                return YL.ravel()
            return YL

        raise NotImplementedError("kernel predicting by  " + self.kernel)
    
    def save(self, file_path):
        with open(file_path,'wb') as file_handler:
            joblib.dump([self.trained, self.verbose, self.kernel, self.loss, self.params, self.kernel_arg, self.loss_arg, self.convert_output, self.ravel_output], file_handler)

    def load(self, file_path):
        with open(file_path,'rb') as file_handler:
            [self.trained, self.verbose, self.kernel, self.loss, self.params, self.kernel_arg, self.loss_arg, self.convert_output, self.ravel_output]= joblib.load(file_handler)



