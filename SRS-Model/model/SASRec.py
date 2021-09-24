import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from utils.modules import self_attention,multi_head_self_attention

class SASRec():
    def __init__(self, args,data_config):
        self.model_type='SASRec'

        self.n_items=data_config['n_items']
        self.max_len=args.max_len
        self.block_num=args.block_num
        self.head_num=args.head_num

        self.emb_dim=args.embed_size
        self.head_dim=self.emb_dim//self.head_num

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
        # W_q_list_i 第i层block的W_Q列表 元素长度为head_num
        for i in range(self.block_num):
            all_weights['W_q_list_{}'.format(i)]=[tf.Variable(initializer([self.emb_dim,self.head_dim]),name='W_q_{}_{}'.format(i,k)) for k in range(self.head_num)]
            all_weights['W_k_list_{}'.format(i)]=[tf.Variable(initializer([self.emb_dim,self.head_dim]),name='W_q_{}_{}'.format(i,k)) for k in range(self.head_num)]
            all_weights['W_v_list_{}'.format(i)]=[tf.Variable(initializer([self.emb_dim,self.head_dim]),name='W_q_{}_{}'.format(i,k)) for k in range(self.head_num)]
        # W_o 多头concat之后线性转化矩阵
        all_weights['W_sub_2_out']=tf.Variable(initializer([self.head_num*self.head_dim,self.emb_dim]),name='W_sub_2_out')
        return all_weights

    # 构造模型
    def _forward(self):
        # 1 得到序列的表示
        his_embeddings=tf.nn.embedding_lookup(self.weights['item_embedding'],self.hist) # [N,max_len,k]
        his_embeddings=his_embeddings+self.weights['position_embedding'] # [N,max_len,k]+[max_len,k]
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
            outputs=self._SA_FFN_block(outputs,mask,i)

        his_final_presention=outputs[:,-1:,:] # [N,1,k] 注意：如果使用outputs[:,-1,:] 维度会变成 [N,k]

        return his_final_presention

    
    # 第i个 SA+FFN block
    def _SA_FFN_block(self,inputs,mask,i):
        # step 1：SA层 
        # step 1-1：投影矩阵得 Q K V
        W_q_list=self.weights['W_q_list_{}'.format(i)]
        W_k_list=self.weights['W_k_list_{}'.format(i)]
        W_v_list=self.weights['W_v_list_{}'.format(i)]
        # Output=Input+Dropout(F(LN(Input)))
        inputs_lnorm=tensorflow.contrib.layers.layer_norm(inputs,center=True, scale=True)
        Q_list=[tf.matmul(inputs_lnorm,W_q_list[k]) for k in range(self.head_num)] # [N,max_len,k]
        K_list=[tf.matmul(inputs_lnorm,W_k_list[k]) for k in range(self.head_num)] # [N,max_len,k]
        V_list=[tf.matmul(inputs_lnorm,W_v_list[k]) for k in range(self.head_num)] # [N,max_len,k]

        outputs=tf.nn.dropout(multi_head_self_attention(Q_list,K_list,V_list,mask),rate=self.drop_rate) # [N,max_len,k]
        # 多头的话 需要线性转化成最终的输出
        if(self.head_num>1):
            outputs=tf.matmul(outputs,self.weights['W_sub_2_out']) # [N,max_len,head_num*head_dim] [head_num*head_dim,k] -> [N,max_len,k]
        outputs=inputs+outputs

        # step 2：FFN层 
        # Output=Input+Dropout(F(LN(Input)))
        inputs_lnorm=tensorflow.contrib.layers.layer_norm(outputs,center=True, scale=True)
        outputs=outputs+tf.nn.dropout(self._ffn(inputs_lnorm),rate=self.drop_rate) # [N,max_len,k]

        return outputs

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
