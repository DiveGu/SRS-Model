import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from utils.modules import self_attention

class SASRec():
    def __init__(self, args,data_config):
        self.model_type='SASRec'

        self.n_items=data_config['n_items']
        self.max_len=args.max_len
        self.block_num=args.block_num

        self.emb_dim=args.embed_size
        self.lr=args.lr

        self.batch_size=args.batch_size
        self.regs=eval(args.regs)

        # 定义输入placeholder
        self.hist=tf.placeholder(tf.int32,shape=[None,None],name='hist') # [N,max_len]
        self.pos_items=tf.placeholder(tf.int32,shape=[None,None],name='pos_items') # [N,1]
        self.neg_items=tf.placeholder(tf.int32,shape=[None,None],name='neg_items') # [N,1] or [N,4]

        # 初始化模型参数
        self.weights=self._init_weights()

        # 构造模型
        self._forward()

    # 初始化参数
    def _init_weights(self):
        all_weights=dict()
        initializer=tensorflow.contrib.layers.xavier_initializer()
        all_weights['item_embedding']=tf.Variable(initializer([self.n_items+1,self.emb_dim]),name='item_embedding')
        print('using xavier initialization')        

        # W_q W_k W_v
        for i in range(self.block_num):
            all_weights['W_q_{}'.format(i)]=tf.Variable(initializer([self.emb_dim,self.emb_dim]),name='W_q_{}'.format(i))
            all_weights['W_k_{}'.format(i)]=tf.Variable(initializer([self.emb_dim,self.emb_dim]),name='W_k_{}'.format(i))
            all_weights['W_v_{}'.format(i)]=tf.Variable(initializer([self.emb_dim,self.emb_dim]),name='W_v_{}'.format(i))

        return all_weights

    # 构造模型
    def _forward(self):
        # 1 得到序列的表示
        his_embeddings=tf.nn.embedding_lookup(self.weights['item_embedding'],self.hist) # [N,max_len,k]
        # pad iid=n_items mask为pad的mask
        mask=tf.cast(tf.equal(self.hist,self.n_items),tf.float32) # [N,max_len]
        mask=tf.expand_dims(mask,axis=2) # [N,max_len,1]
        mask=tf.matmul(mask,tf.transpose(mask,perm=[0,2,1])) # [N,max_len,1] [N,1,max_len]-> [N,max_len,max_len]
        his_represention=self._bulid_his_represention(his_embeddings,mask)
        # 2 得到pos和neg的target表示
        target_pos_embeddings=tf.nn.embedding_lookup(self.weights['item_embedding'],self.pos_items) # [N,1,k]
        target_neg_embeddings=tf.nn.embedding_lookup(self.weights['item_embedding'],self.neg_items) # [N,4,k]
        # 3 得到预测评分
        pos_preidct_scores=tf.nn.sigmoid(self._get_predict_score(his_represention,target_pos_embeddings)) # [N,1,k]
        neg_preidct_scores=tf.nn.sigmoid(self._get_predict_score(his_represention,target_neg_embeddings)) # [N,4,k]
        self.batch_ratings=pos_preidct_scores
        # 4 构造损失函数
        neg_num=tf.dtypes.cast(tf.shape(neg_preidct_scores)[1], tf.int32)
        cf_loss_list=[-tf.math.log(pos_preidct_scores),-tf.math.log(1-neg_preidct_scores)]
        cf_loss=tf.reduce_mean(tf.concat(cf_loss_list,axis=1))
        self.loss=cf_loss
        # 5 优化
        self.opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


    # 根据history items嵌入得到最终序列表示
    def _bulid_his_represention(self,his_embeddings,mask):
        outputs=his_embeddings # [N,max_len,k]
        # SA-FFN为一个block
        for i in range(self.block_num):
            # step 1-1：投影矩阵得 Q K V
            W_q=self.weights['W_q_{}'.format(i)]
            W_k=self.weights['W_k_{}'.format(i)]
            W_v=self.weights['W_v_{}'.format(i)]
            Q=tf.matmul(outputs,W_q) # [N,max_len,k]
            K=tf.matmul(outputs,W_k) # [N,max_len,k]
            V=tf.matmul(outputs,W_v) # [N,max_len,k]
            # step 1-2：进行SA
            outputs=self_attention(Q,K,V,mask) # [N,max,k]
            # step 2:FFN （注：每一层的FFN都相同）
            outputs=self._ffn(outputs) # [N,max_len,k]

        his_final_presention=outputs[:,-1,:] # [N,1,k]

        return his_final_presention

    # Self Attention之后的FFN
    def _ffn(self,inputs):
        outputs=inputs
        outputs=tf.layers.dense(outputs,
                                self.emb_dim,
                                use_bias=True,
                                activation='relu',
                                name="FFN_1",
                                reuse=tf.AUTO_REUSE,
                                )
        outputs=tf.layers.dense(outputs,
                                self.emb_dim,
                                use_bias=True,
                                activation=None,
                                name="FFN_2",
                                reuse=tf.AUTO_REUSE,
                                )
        return outputs

    # 根据hist表示和target表示预测评分
    def _get_predict_score(self,hist_e,target_e):
        # [N,1,k],[N,4,k]
        predict_logit=tf.multiply(hist_e,target_e) # [N,4,k]
        predict_logit=tf.reduce_sum(predict_logit,axis=2,keepdims=False) # [N,4,k] -> [N,4]

        return predict_logit

    # 训练
    def train(self,sess,feed_dict):
        return sess.run([self.opt,self.loss],feed_dict=feed_dict)

    # 预测
    def predict(self,sess,feed_dict):
        return sess.run(self.batch_ratings,feed_dict=feed_dict)
