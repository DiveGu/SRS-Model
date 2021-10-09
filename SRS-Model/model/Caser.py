"""
Caser
"""

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Caser():
    def __init__(self, args,data_config):
        self.model_type='Caser'

        self.n_items=data_config['n_items']
        self.n_users=data_config['n_users']
        self.max_len=args.max_len
        self.h_filter_sizes=eval(args.h_filter_size)
        self.h_filter_num=args.h_filter_num
        self.v_filter_num=args.v_filter_num

        self.emb_dim=args.embed_size

        self.lr=args.lr

        self.batch_size=args.batch_size
        self.regs=eval(args.regs)

        # 定义输入placeholder
        self.users=tf.placeholder(tf.int32,shape=[None,None],name='users') # [N,1]
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
        all_weights['user_embedding']=tf.Variable(initializer([self.n_users,self.emb_dim]),name='user_embedding')
        print('using xavier initialization')        
        # 位置嵌入
        all_weights['position_embedding']=tf.Variable(initializer([self.max_len,self.emb_dim]),name='position_embedding')
        
        # h filter
        for h_size in self.h_filter_sizes:
            # [h_size,k,1,num]
            all_weights['h_filter_w_size_{}'.format(h_size)]=tf.Variable(
                tf.truncated_normal([h_size,self.emb_dim,1,self.h_filter_num],stddev=0.1),
                name='h_filter_w_size_{}'.format(h_size))
            all_weights['h_filter_b_size_{}'.format(h_size)]=tf.Variable(
                tf.constant(0.1, shape=[self.h_filter_num]),
                name='h_filter_b_size_{}'.format(h_size))

        # v filter
        # [max_len,1,1,num]
        all_weights['v_filter_w']=tf.Variable(
            tf.truncated_normal([self.max_len,1,1,self.v_filter_num],stddev=0.1),
            name='v_filter_w')
        all_weights['v_filter_b']=tf.Variable(
            tf.constant(0.1, shape=[self.v_filter_num]),
            name='v_filter_b')
        

        # target item embedding
        all_weights['target_item_embedding']=tf.Variable(initializer([self.n_items+1,2*self.emb_dim]),name='target_item_embedding')
        # target item bias
        all_weights['target_item_bias']=tf.Variable(initializer([self.n_items+1]),name='target_item_bias')

        return all_weights

    # 构造模型
    def _forward(self):
        # 1 得到序列的表示
        user_embeddings=tf.nn.embedding_lookup(self.weights['user_embedding'],self.users) # [N,1,k]
        user_embeddings=tf.squeeze(user_embeddings,axis=1) # [N,1,k] -> [N,k]
        his_embeddings=tf.nn.embedding_lookup(self.weights['item_embedding'],self.hist) # [N,max_len,k]
        # pad iid=n_items mask为pad的mask 也就是 pad对应为0 其他item对应为1
        mask=tf.cast(tf.not_equal(self.hist,self.n_items),tf.float32) # [N,max_len]
        mask=tf.expand_dims(mask,axis=2) # [N,max_len,1]
        his_embeddings=tf.multiply(his_embeddings,mask) # [N,max_len,k] 把pad位置的嵌入置为0
        his_embeddings=tf.expand_dims(his_embeddings,axis=3) # [N,max_len,k] -> [N,max_len,k,1]

        his_represention=self._bulid_his_represention(his_embeddings,user_embeddings) # [N,1,k]

        # 2 得到pos和neg的target表示
        target_pos_embeddings=tf.nn.embedding_lookup(self.weights['target_item_embedding'],self.pos_items) # [N,1,k]
        target_neg_embeddings=tf.nn.embedding_lookup(self.weights['target_item_embedding'],self.neg_items) # [N,4,k]
        target_pos_bias=tf.nn.embedding_lookup(self.weights['target_item_bias'],self.pos_items) # [N,1]
        target_neg_bias=tf.nn.embedding_lookup(self.weights['target_item_bias'],self.neg_items) # [N,neg_num]
        
        # 3 得到预测评分
        pos_preidct_scores=tf.nn.sigmoid(self._get_predict_score(his_represention,target_pos_embeddings,target_pos_bias)) # [N,1]
        neg_preidct_scores=tf.nn.sigmoid(self._get_predict_score(his_represention,target_neg_embeddings,target_neg_bias)) # [N,neg_num]
        self.batch_ratings=pos_preidct_scores
        
        # 4 构造损失函数
        #neg_num=tf.dtypes.cast(tf.shape(neg_preidct_scores)[1], tf.int32)
        cf_loss_list=[-tf.math.log(pos_preidct_scores+1e-24),-tf.math.log(1-neg_preidct_scores+1e-24)]
        cf_loss=tf.reduce_mean(tf.concat(cf_loss_list,axis=1))
        reg_loss=tf.nn.l2_loss(his_embeddings)+tf.nn.l2_loss(target_pos_embeddings)+tf.nn.l2_loss(target_neg_embeddings)+\
            tf.nn.l2_loss(target_pos_bias)+tf.nn.l2_loss(target_neg_bias)

        reg_loss=reg_loss*self.regs[0]

        self.loss=cf_loss+reg_loss
        # 5 优化
        self.opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


    # 根据history items嵌入得到最终序列表示
    def _bulid_his_represention(self,his_embeddings,user_embeddings):
        """
        his_embeddings:[N,max_len,k,1]
        user_embeddings:[N,k]
        """
        h_feat=[]
        # 【step 1】: h filter 
        for h_size in self.h_filter_sizes:
            # 1-1：卷积[N,max_len,k,1] -> [N,max_len-size+1,1,1]
            conv1=tf.nn.conv2d(input=his_embeddings,
                               filter=self.weights['h_filter_w_size_{}'.format(h_size)], # [size,k,1,num]
                               strides=[1,1,1,1],
                               padding='VALID',)
            # 1-2：+bias [N,max_len-size+1,1,num]
            conv1=tf.nn.bias_add(conv1,self.weights['h_filter_b_size_{}'.format(h_size)])
            # 1-3：最大池化 [N,1,1,num]
            pool1 = tf.nn.max_pool(value=conv1, 
                                   ksize=[1, self.max_len-h_size+1, 1, 1],  
                                   strides=[1, 1, 1, 1], # 不用管这个维度
                                   padding='VALID',) # [N,1,1,num]

            print('h size:{},conv1:{},pool1:{}'.format(h_size,conv1.get_shape(),pool1.get_shape()))

            h_feat.append(tf.squeeze(pool1,axis=[1,2])) # 新增 [N,num]

        # 【step 2】: v filter 
        # 2-1：卷积 [N,max_len,k,1] -> [N,1,k,num]
        conv2=tf.nn.conv2d(input=his_embeddings,
                           filter=self.weights['v_filter_w'], # [max_len,1,1,num]
                           strides=[1,1,1,1],
                           padding='VALID',)
        # 2-2：+bias
        conv2=tf.nn.bias_add(conv2,self.weights['v_filter_b'])

        print(conv2.get_shape())

        # 【step 3】: concat[seq,uid]+MLP
        h_feat=tf.concat(h_feat,axis=1) # [N,num* ||h_size||]
        v_feat=tf.reshape(conv2,[-1,self.emb_dim*self.v_filter_num]) # [N,1,k,num]->[N,k*num]

        # -> [N,k]
        cnn_feat=tf.layers.dense(tf.concat([h_feat,v_feat],axis=1),
                                 self.emb_dim,
                                 use_bias=True,
                                 activation='relu',)

        his_final_presention=tf.concat([cnn_feat,user_embeddings],axis=1) # [N,2k]
        his_final_presention=tf.expand_dims(his_final_presention,axis=1) # [N,2k] -> [N,1,2k]

        return his_final_presention

    
    # 根据hist表示和target表示预测评分
    def _get_predict_score(self,hist_e,target_e,target_b):
        # [N,1,2k],[N,neg_num,2k],[N,neg_num] 
        #print(hist_e.get_shape())
        predict_logit=tf.multiply(hist_e,target_e) # [N,neg_num,2k]
        predict_logit=tf.reduce_sum(predict_logit,axis=2,keepdims=False) # [N,neg_num,2k] -> [N,neg_num]

        return predict_logit+target_b

    # 训练
    def train(self,sess,feed_dict):
        return sess.run([self.opt,self.loss],feed_dict=feed_dict)

    # 预测
    def predict(self,sess,feed_dict):
        return sess.run(self.batch_ratings,feed_dict=feed_dict)


