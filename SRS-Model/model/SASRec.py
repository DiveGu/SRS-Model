import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class SASRec():
    def __init__(self, args,data_config):
        self.model_type='SASRec'

        self.n_items=data_config['n_items']
        self.maxlen=args.max_len

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

        return all_weights

    # 构造模型
    def _forward(self):
        # 1 得到序列的表示
        his_embeddings=tf.nn.embedding_lookup(self.weights['item_embedding'],self.hist) # [N,max_len,k]
        his_represention=self._bulid_his_represention(his_embeddings)
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
        return

    # 根据history items嵌入得到最终序列表示
    def _bulid_his_represention(self,his_embeddings):
        final_represention=tf.reduce_mean(his_embeddings,axis=1,keepdims=True) # [N,max_len,k] -> [N,1,k]
        return final_represention

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
