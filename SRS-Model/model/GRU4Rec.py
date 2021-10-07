"""
https://github.com/slientGe/Sequential_Recommendation_Tensorflow/blob/master/models/GRU4Rec/model_GRU4rec.py
"""

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class GRU4Rec():
    def __init__(self, args,data_config):
        self.model_type='GRU4Rec'

        self.n_items=data_config['n_items']
        self.max_len=args.max_len
        self.gru_layers=eval(args.gru_layers)

        self.emb_dim=args.embed_size

        self.lr=args.lr

        self.batch_size=args.batch_size
        self.regs=eval(args.regs)

        # 定义输入placeholder
        self.hist=tf.placeholder(tf.int32,shape=[None,None],name='hist') # [N,max_len]
        self.pos_items=tf.placeholder(tf.int32,shape=[None,None],name='pos_items') # [N,1]
        self.neg_items=tf.placeholder(tf.int32,shape=[None,None],name='neg_items') # [N,1] or [N,4]
        self.drop_rate=tf.placeholder(tf.float32,name='dropout_rate') # 丢弃的比例

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
        # 位置嵌入
        all_weights['position_embedding']=tf.Variable(initializer([self.max_len,self.emb_dim]),name='position_embedding')
        
        # target item embedding
        all_weights['target_item_embedding']=tf.Variable(initializer([self.n_items+1,self.emb_dim]),name='target_item_embedding')

        return all_weights

    # 构造模型
    def _forward(self):
        # 1 得到序列的表示
        his_embeddings=tf.nn.embedding_lookup(self.weights['item_embedding'],self.hist) # [N,max_len,k]
        #his_embeddings=his_embeddings+self.weights['position_embedding'] # [N,max_len,k]+[max_len,k]
        # pad iid=n_items mask为pad的mask 也就是 pad对应为0 其他item对应为1
        mask=tf.cast(tf.not_equal(self.hist,self.n_items),tf.float32) # [N,max_len]
        mask=tf.expand_dims(mask,axis=2) # [N,max_len,1]
        his_embeddings=tf.multiply(his_embeddings,mask) # [N,max_len,k] 把pad位置的嵌入置为0
        his_represention=self._bulid_his_represention(his_embeddings) # [N,1,k]

        # 2 得到pos和neg的target表示
        target_pos_embeddings=tf.nn.embedding_lookup(self.weights['target_item_embedding'],self.pos_items) # [N,1,k]
        target_neg_embeddings=tf.nn.embedding_lookup(self.weights['target_item_embedding'],self.neg_items) # [N,4,k]
        # 3 得到预测评分
        pos_preidct_scores=tf.nn.sigmoid(self._get_predict_score(his_represention,target_pos_embeddings)) # [N,1]
        neg_preidct_scores=tf.nn.sigmoid(self._get_predict_score(his_represention,target_neg_embeddings)) # [N,4]
        self.batch_ratings=pos_preidct_scores
        # 4 构造损失函数
        #neg_num=tf.dtypes.cast(tf.shape(neg_preidct_scores)[1], tf.int32)
        cf_loss_list=[-tf.math.log(pos_preidct_scores+1e-24),-tf.math.log(1-neg_preidct_scores+1e-24)]
        cf_loss=tf.reduce_mean(tf.concat(cf_loss_list,axis=1))
        reg_loss=tf.nn.l2_loss(his_embeddings)+tf.nn.l2_loss(target_pos_embeddings)+tf.nn.l2_loss(target_neg_embeddings)
        reg_loss=reg_loss*self.regs[0]

        self.loss=cf_loss+reg_loss
        # 5 优化
        self.opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


    # 根据history items嵌入得到最终序列表示
    def _bulid_his_represention(self,his_embeddings):
        cells=[]
        # step 1：构造每一层的GRU Cell
        for h_dim in self.gru_layers:
            cell = tensorflow.contrib.rnn.GRUCell(h_dim, activation=tf.nn.tanh) # [input_dim,h_dim]
            #cell = rnn.DropoutWrapper(cell, output_keep_prob=in_KP)
            cells.append(cell)

        self.cell = tensorflow.contrib.rnn.MultiRNNCell(cells) # [N,max_len,h_dim]
        zero_state = self.cell.zero_state(tf.shape(his_embeddings)[0], dtype=tf.float32) # [N]

        outputs, state = tf.nn.dynamic_rnn(self.cell,his_embeddings,initial_state=zero_state) # [N,max_len,k]
        his_final_presention = outputs[:,-1:,:] # [N,1,h_dim] 注意：如果使用outputs[:,-1,:] 维度会变成 [N,h_dim]

        # step 2：最后一层的GRU隐藏层+MLP
        his_final_presention=tf.layers.dense(his_final_presention,
                                                self.emb_dim,
                                                use_bias=True,
                                                activation="relu",
                                                )
        # [N,1,h_dim] -> [N,1,k]

        return his_final_presention

    
    # 根据hist表示和target表示预测评分
    def _get_predict_score(self,hist_e,target_e):
        # [N,1,k],[N,4,k]
        #print(hist_e.get_shape())
        predict_logit=tf.multiply(hist_e,target_e) # [N,4,k]
        predict_logit=tf.reduce_sum(predict_logit,axis=2,keepdims=False) # [N,4,k] -> [N,4]

        return predict_logit

    # 训练
    def train(self,sess,feed_dict):
        return sess.run([self.opt,self.loss],feed_dict=feed_dict)

    # 预测
    def predict(self,sess,feed_dict):
        return sess.run(self.batch_ratings,feed_dict=feed_dict)

