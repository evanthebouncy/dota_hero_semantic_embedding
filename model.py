# BATTLE SHIP

import tensorflow as tf
import numpy as np
from data import *

def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  if name:
    return tf.Variable(initial, name)
  else:
    return tf.Variable(initial)

def bias_variable(shape, name=None):
  initial = tf.constant(0.1, shape=shape)
  if name:
    return tf.Variable(initial, name)
  else:
    return tf.Variable(initial)

class EmbedNet:

  def gen_feed_dict(self, heros, teams):
    ret = dict()
    ret[self.heros] = heros
    ret[self.teams] = teams
    return ret

  # load the model and give back a session
  def load_model(self, saved_loc):
    sess = self.sess
    self.saver.restore(sess, saved_loc)
    print("Model restored.")

  # make the model
  def __init__(self, sess):
    self.name = 'embednet'
    with tf.variable_scope(self.name) as scope:
      # set up placeholders
      self.heros = tf.placeholder(tf.float32, [N_BATCH, L], name="heros")
      self.teams = tf.placeholder(tf.float32, [N_BATCH, L , 2], name="teams")

      # embed the input
      embed = tf.layers.dense(inputs=self.heros, units=20)
      self.embed = embed

      embed_dim = int(embed.get_shape()[1])
    
      # do the prediction on top of the embedding
      W_preds = [weight_variable([embed_dim, 2]) for _ in range(L)]
      b_preds = [bias_variable([2]) for _ in range(L)]
      e2 = 1e-10

      self.query_preds = [tf.nn.softmax(tf.matmul(embed, W_preds[i]) + b_preds[i])+e2 for i in range(L)]

      # doing some reshape of the input tensor
      teams_trans = tf.transpose(self.teams, perm=[1,0,2])
      print teams_trans.get_shape()
      teams_split = tf.reshape(teams_trans, [L, N_BATCH, 2])
      teams_split = tf.unstack(teams_split)

      self.query_pred_costs = []
      for idx in range(L):
        blah = -tf.reduce_sum(teams_split[idx] * tf.log(self.query_preds[idx]))
        self.query_pred_costs.append(blah)
        
      self.cost_query_pred = sum(self.query_pred_costs)

      # ------------------------------------------------------------------------ training steps
      optimizer = tf.train.AdamOptimizer(0.01)


      pred_gvs = optimizer.compute_gradients(self.cost_query_pred)
      capped_pred_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in pred_gvs]
      #train_pred = optimizer.minimize(cost_pred, var_list = VAR_pred)
      self.train_query_pred = optimizer.apply_gradients(capped_pred_gvs)

      # train_query_pred = optimizer.minimize(cost_query_pred, var_list = VAR_pred)
      # Before starting, initialize the variables.  We will 'run' this first.
      self.init = tf.global_variables_initializer()
      self.saver = tf.train.Saver()
      self.sess = sess

  def initialize(self):
    self.sess.run(self.init)

  # save the model
  def save(self):
    model_loc = "./models/" + self.name+".ckpt"
    sess = self.sess
    save_path = self.saver.save(sess, model_loc)
    print("Model saved in file: %s" % save_path)

  # train on a particular data batch
  def train(self, data_batch):
    sess = self.sess

    heros, teams = data_batch
    feed_dic = self.gen_feed_dict(heros, teams)

    cost_query_pred_pre = sess.run([self.cost_query_pred], feed_dict=feed_dic)[0]
    sess.run([self.train_query_pred], feed_dict=feed_dic)
    cost_query_pred_post = sess.run([self.cost_query_pred], feed_dict=feed_dic)[0]
    print "train query pred ", cost_query_pred_pre, " ",\
      cost_query_pred_post, " ", True if cost_query_pred_post < cost_query_pred_pre else False

  # =========== HELPERS =============

  def get_embedding(self, all_heros):
    ret = dict()
    ret[self.heros] = all_heros
    return self.sess.run([self.embed], feed_dict=ret)
    
