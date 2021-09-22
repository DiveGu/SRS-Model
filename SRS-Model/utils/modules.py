import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow


def self_attention(q,k,v,d):
    scores=tf.matmul(q,tf.transpose(x,perm=[1,0]))/(d**0.5) # [N,d] [d,N] -> [N,N]
    scores=tf.nn.softmax(scores,axis=-1) # [N,N]
    return tf.matmul(scores,v) # [N,N] [N,d] -> [N,d]


