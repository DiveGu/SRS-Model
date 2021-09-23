import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow


def self_attention(Q,K,V,mask,future_mask_flag=True):
    """
    Q:[N,seq_len,d]
    K:[N,seq_len,d]
    V:[N,seq_len,d]
    mask:[N,seq_len,seq_len] item id的mask，即pad id对应的mask为0
    """
    #d=K.get_shape().as_list()[-1]
    d=tf.cast(K.shape(-1),tf.float32)
    outputs=tf.matmul(Q,tf.transpose(K,perm=[0,2,1]))/(d**0.5) # [N,seq_len,d] [N,d,seq_len] -> [N,seq_len,seq_len]
    # mask[i,j]的s[i,j]为很小的负数 经过softmax s[i,j]=0 
    paddings=tf.ones_like(outputs)*(-2**32+1) # [N,seq_len,seq_len]
    outputs=tf.where(tf.equal(mask,0),paddings,outputs) # [N,seq_len,seq_len]

    # mask j>i的位置的scores
    if(future_mask_flag):
        diag_vals = tf.ones_like(outputs)  # [N,seq_len,seq_len]
        mask = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # [N,seq_len,seq_len]
        # masks为下三角tensor
        paddings = tf.ones_like(mask) * (-2 ** 32 + 1) # [N,seq_len,seq_len]
        outputs = tf.where(tf.equal(mask, 0), paddings, outputs)  # [N,seq_len,seq_len]


    outputs=tf.nn.softmax(outputs,axis=-1) # [N,seq_len,seq_len]
    outputs=tf.matmul(outputs,V) # [N,seq_len,seq_len] [N,seq_len,d] -> [N,seq_len,d]

    return outputs

