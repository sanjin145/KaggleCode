#!/usr/bin/env python
# encoding: utf-8

"""
第一个TensorFlow测试类
@version: V1.0.0
@author: gold_worker
@contact: dingxin_hz@163.com
@time: 2018-12-18 下午 2:37
"""

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant([1.0,2.0],name="a")
b = tf.constant([2.0,3.0],name="b")

sess = tf.Session()
result = sess.run(a+b)
print(type(result))#ndarray

print(tf.get_default_graph())
print(a.graph is tf.get_default_graph())


