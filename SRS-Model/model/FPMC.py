"""
FPMC
"""

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class FPMC():
    def __init__(self, args,data_config):
        self.model_type='FPMC'

        self.n_items=data_config['n_items']
        self.n_users=data_config['n_users']
        self.max_len=args.max_len
        self.gru_layers=eval(args.gru_layers)

        self.emb_dim=args.embed_size

        self.lr=args.lr

        self.batch_size=args.batch_size
        self.regs=eval(args.regs)
        self.pairwise_loss=True

        # 定义输入placeholder
        self.users=tf.placeholder(tf.int32,shape=[None,None],name='users') # [N,1]
        self.hist=tf.placeholder(tf.int32,shape=[None,None],name='hist') # [N,max_len] max_len为MC中的阶数
        self.pos_items=tf.placeholder(tf.int32,shape=[None,None],name='pos_items') # [N,1]
        self.neg_items=tf.placeholder(tf.int32,shape=[None,None],name='neg_items') # [N,1] 
        self.drop_rate=tf.placeholder(tf.float32,name='dropout_rate') # 丢弃的比例

        # 初始化模型参数
        self.weights=self._init_weights()

        # 构造模型
        self._forward()

    # 初始化参数
    def _init_weights(self):
        all_weights=dict()
        initializer=tensorflow.contrib.layers.xavier_initializer()
        # 分别是 UI HI IU IH R=<UI,IU>+<HI,IH>
        all_weights['user_embedding']=tf.Variable(initializer([self.n_users,self.emb_dim]),name='user_embedding')
        all_weights['history_hi_embedding']=tf.Variable(initializer([self.n_items+1,self.emb_dim]),name='history_hi_embedding')
        all_weights['target_iu_embedding']=tf.Variable(initializer([self.n_items+1,self.emb_dim]),name='target_iu_embedding')
        all_weights['target_ih_embedding']=tf.Variable(initializer([self.n_items+1,self.emb_dim]),name='target_ih_embedding')
        
        print('using xavier initialization')        
        # 位置嵌入
        all_weights['position_embedding']=tf.Variable(initializer([self.max_len,self.emb_dim]),name='position_embedding')
       

        return all_weights

    # 构造模型
    def _forward(self):
        # 1 得到四种嵌入
        u_embeddings_ui=tf.nn.embedding_lookup(self.weights['user_embedding'],self.users) # [N,1,k]
        his_embeddings_hi=self._get_mask_emb(self.weights['history_hi_embedding'],self.hist,self.n_items) # [N,max_len,k]
        
        pos_embeddings_iu=self._get_mask_emb(self.weights['target_iu_embedding'],self.pos_items,self.n_items) # [N,1,k]
        pos_embeddings_ih=self._get_mask_emb(self.weights['target_ih_embedding'],self.pos_items,self.n_items) # [N,1,k]

        neg_embeddings_iu=self._get_mask_emb(self.weights['target_iu_embedding'],self.neg_items,self.n_items) # [N,1,k]
        neg_embeddings_ih=self._get_mask_emb(self.weights['target_ih_embedding'],self.neg_items,self.n_items) # [N,1,k]

        print(u_embeddings_ui.shape)
        print(his_embeddings_hi.shape)

        # 2 计算对pos neg的预测评分
        pos_preidct_scores=self._get_predict_score(u_embeddings_ui,his_embeddings_hi,pos_embeddings_iu,pos_embeddings_ih)
        neg_preidct_scores=self._get_predict_score(u_embeddings_ui,his_embeddings_hi,neg_embeddings_iu,neg_embeddings_ih)
        self.batch_ratings=pos_preidct_scores

        # 4 构造损失函数
        if(self.pairwise_loss):
            cf_loss=tf.log(tf.nn.sigmoid(pos_preidct_scores-neg_preidct_scores)+1e-24)
            cf_loss=-(tf.reduce_mean(cf_loss)) # [N,1] -> 1
        else:
            cf_loss_list=[-tf.math.log(pos_preidct_scores+1e-24),-tf.math.log(1-neg_preidct_scores+1e-24)]
            cf_loss=tf.reduce_mean(tf.concat(cf_loss_list,axis=1))

        reg_loss=tf.nn.l2_loss(u_embeddings_ui)+tf.nn.l2_loss(his_embeddings_hi)+\
        tf.nn.l2_loss(pos_embeddings_iu)+tf.nn.l2_loss(pos_embeddings_ih)+\
        tf.nn.l2_loss(neg_embeddings_iu)+tf.nn.l2_loss(neg_embeddings_ih)

        reg_loss=reg_loss*self.regs[0]

        self.loss=cf_loss+reg_loss
        # 5 优化
        self.opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # 获取ids的嵌入 其中等于pad_id的位置返回0
    def _get_mask_emb(self,embeddings,ids,pad_id):
        outputs=tf.nn.embedding_lookup(embeddings,ids) # [N,max_len,k]
        mask=tf.cast(tf.not_equal(ids,pad_id),tf.float32) # [N,max_len]
        mask=tf.expand_dims(mask,axis=-1) # [N,max_len,1]
        return tf.multiply(outputs,mask) # [N,max_len,k]

    
    # 根据hist表示和target表示预测评分
    def _get_predict_score(self,ui,hi,iu,ih):
        # [N,1,k],[N,len,k] [N,1,k] [N,1,k]
        #print(hist_e.get_shape())
        logit_ui=tf.multiply(ui,iu) # [N,1,k]
        logit_ui=tf.reduce_sum(logit_ui,axis=2,keepdims=False) # [N,1,k] -> [N,1]

        logit_hi=tf.multiply(hi,ih) # [N,len,k]
        logit_hi=tf.reduce_sum(logit_hi,axis=2,keepdims=False) # [N,len,k] -> [N,len]
        logit_hi=tf.reduce_sum(logit_hi,axis=1,keepdims=True) # [N,len] -> [N,1]

        return logit_ui+logit_hi # [N,1]

    # 训练
    def train(self,sess,feed_dict):
        return sess.run([self.opt,self.loss],feed_dict=feed_dict)

    # 预测
    def predict(self,sess,feed_dict):
        return sess.run(self.batch_ratings,feed_dict=feed_dict)


