#!/usr/bin/env python
# encoding: utf-8

"""
@version: V1.0.0
@author: gold_worker
@contact: dingxin_hz@163.com
@time: 2018-12-18 下午 2:57
"""

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

g1 = tf.Graph()
with g1.as_default():
    #在计算图g1中定义变量v,并设置初始值为0：
    v = tf.get_variable("v", shape=[1], initializer=tf.zeros_initializer)

g2 = tf.Graph()
with g2.as_default():
    # 在计算图g2中定义变量v,并设置初始值为1：
    v = tf.get_variable("v",shape=[1], initializer=tf.ones_initializer)

#在计算图g1中读取变量"v"的取值。
with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("",reuse=True):
        #在计算图g1中吗，变量v的取值应该为0
        print(sess.run(tf.get_variable("v")))

#在计算图g2中读取变量"v"的取值。
with tf.Session(graph=g2) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("",reuse=True):
        #在计算图g2中吗，变量v的取值应该为0
        print(sess.run(tf.get_variable("v")))



