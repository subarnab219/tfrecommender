
# coding: utf-8

# In[146]:


import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.contrib.learn import *

import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

import os


# In[148]:


class tfrecommender(object):
    def __init__(self):
        pass
    
    def datasplit(self):
        ind = np.random.rand(len(self.user)) < (1-self.valid_size)
        np.random.shuffle(self.user)
        np.random.shuffle(self.item)
        np.random.shuffle(self.feedback)
        user_train, user_val = self.user[ind], self.user[~ind]
        item_train, item_val = self.item[ind], self.item[~ind]
        feedback_train, feedback_val = self.feedback[ind], self.feedback[~ind]
        return user_train, user_val, item_train, item_val, feedback_train, feedback_val
    
    def tensorflowgraph(self, max_user_id,max_item_id, embedding_size, reg_param, learning_rate):
        # create tensorflow graph
        g = tf.Graph()
        with g.as_default():
            # placeholders
            users = tf.placeholder(shape=[None], dtype=tf.int64)
            items = tf.placeholder(shape=[None], dtype=tf.int64)
            ratings = tf.placeholder(shape=[None], dtype=tf.float32)

            # variables
            with tf.variable_scope("embedding"):
                user_weight = tf.get_variable("user_w"
                                              , shape=[max_user_id + 1, embedding_size]
                                              , dtype=tf.float32
                                              , initializer=layers.xavier_initializer())

                item_weight = tf.get_variable("item_w"
                                               , shape=[max_item_id + 1, embedding_size]
                                               , dtype=tf.float32
                                               , initializer=layers.xavier_initializer())
            # prediction
            with tf.name_scope("inference"):
                user_embedding = tf.nn.embedding_lookup(user_weight, users)
                item_embedding = tf.nn.embedding_lookup(item_weight, items)
                pred = tf.reduce_sum(tf.multiply(user_embedding, item_embedding), 1) 

            # loss 
            with tf.name_scope("loss"):
                reg_loss = tf.contrib.layers.apply_regularization(layers.l2_regularizer(scale=reg_param),
                                                       weights_list=[user_weight, item_weight])
                loss = tf.nn.l2_loss(pred - ratings) + reg_loss
                train_ops = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
                rmse = tf.sqrt(tf.reduce_mean(tf.pow(pred - ratings, 2)))
            pass
        return g
    
    def fit(self, user, item, feedback, valid_size = None, embedding_size = None, reg_param = None, learning_rate = None):   
        self.valid_size = valid_size
        self.user = user
        self.feedback = feedback
        self.item = item

        max_user_id = user.max() 
        max_item_id = item.max() 
        
        user_train, user_val, item_train, item_val, feedback_train, feedback_val = self.datasplit()
#         g = self.tensorflowgraph(max_user_id,max_item_id, embedding_size, reg_param, learning_rate)
        # create tensorflow graph
        
        g = tf.Graph()
        with g.as_default():
            # placeholders
            users = tf.placeholder(shape=[None], dtype=tf.int64)
            items = tf.placeholder(shape=[None], dtype=tf.int64)
            ratings = tf.placeholder(shape=[None], dtype=tf.float32)

            # variables
            with tf.variable_scope("embedding"):
                user_weight = tf.get_variable("user_w"
                                              , shape=[max_user_id + 1, embedding_size]
                                              , dtype=tf.float32
                                              , initializer=layers.xavier_initializer())

                item_weight = tf.get_variable("item_w"
                                               , shape=[max_item_id + 1, embedding_size]
                                               , dtype=tf.float32
                                               , initializer=layers.xavier_initializer())
            # prediction
            with tf.name_scope("inference"):
                user_embedding = tf.nn.embedding_lookup(user_weight, users)
                item_embedding = tf.nn.embedding_lookup(item_weight, items)
                pred = tf.reduce_sum(tf.multiply(user_embedding, item_embedding), 1) 

            # loss 
            with tf.name_scope("loss"):
                reg_loss = tf.contrib.layers.apply_regularization(layers.l2_regularizer(scale=reg_param),
                                                       weights_list=[user_weight, item_weight])
                loss = tf.nn.l2_loss(pred - ratings) + reg_loss
                train_ops = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
                rmse = tf.sqrt(tf.reduce_mean(tf.pow(pred - ratings, 2)))
            pass
    
    
    
        # Training 
        epochs = 1000 # number of iterations 
        losses_train = []
        losses_val = []

        with tf.Session(graph=g) as sess:
            # initializer
            sess.run(tf.global_variables_initializer())


            train_input_dict = {  users: user_train
                                , items: item_train
                                , ratings: feedback_train}
            val_input_dict =  {  users: user_val
                                , items: item_val
                                , ratings: feedback_val}
            
            def check_overfit(validation_loss):
                n = len(validation_loss)
                if n < 5:
                    return False
                count = 0 
                for i in range(n-4, n):
                    if validation_loss[i] < validation_loss[i-1]:
                        count += 1
                    if count >=2:
                        return False
                return True
            
            for i in range(epochs):
                # run the training operation
                sess.run([train_ops], feed_dict=train_input_dict)

                # show intermediate results 
                if i % 5 == 0:
                    loss_train = sess.run(loss, feed_dict=train_input_dict)
                    loss_val = sess.run(loss, feed_dict=val_input_dict)
                    losses_train.append(loss_train)
                    losses_val.append(loss_val)


                    # check early stopping 
                    if(check_overfit(losses_val)):
                        print('overfit !')
                        break

                    print("iteration : {0} train loss: {1:.3f} , valid loss {2:.3f}".format(i,loss_train, loss_val))

#             # calculate RMSE on the test dataset
#             print('RMSE on test dataset : {0:.4f}'.format(sess.run(rmse, feed_dict=test_input_dict)))

            plt.plot(losses_train, label='train')
            plt.plot(losses_val, label='validation')
            #plt.ylim(0, 50000)
            plt.legend(loc='best')
            plt.title('Loss');
            
    def predict(self, user_features, item_features):
        """
        Predict recommendation scores for the given users and items.
        :param user_features: scipy.sparse matrix
        A matrix of user features of shape [n_users, n_user_features].
        :param item_features: scipy.sparse matrix
        A matrix of item features of shape [n_items, n_item_features].
        :return: np.ndarray
        The predictions in an ndarray of shape [n_users, n_items]
        """
        feed_dict = self._create_feed_dict(interactions_matrix=None,
                                           user_features_matrix=user_features,
                                           item_features_matrix=item_features)

        predictions = self.tf_prediction.eval(session=get_session(), feed_dict=feed_dict)

        return predictions
