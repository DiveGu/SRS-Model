"""
- DGCF
- 2021/7/28
"""
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class PDCF():
    def __init__(self,data_config,pretrain_data,args):
        self.model_type='PDCF'
        self.pretrain_data=pretrain_data

        self.n_users=data_config['n_users']
        self.n_items=data_config['n_items']
        # 通过data_config传进来邻接矩阵
        # 需要提取将邻接矩阵A转化成coo [row,col,data] h即为row t即为col
        self.norm_adj = data_config['norm_adj'] # 邻接矩阵
        self.all_h_list = data_config['all_h_list'] # 所有Edge的头节点 list
        self.all_t_list = data_config['all_t_list'] # 所有Edge的尾节点 list
        self.A_in_shape = self.norm_adj.tocoo().shape # 邻接矩阵的形状

        self.n_fold=100
        self.n_iteration=args.n_iteration

        self.layer_num=args.layer_num

        self.verbose=args.verbose

        self.emb_dim=args.embed_size
        self.lr=args.lr

        # factor数量
        self.factor_num=args.factor_num
        self.factor_dim=self.emb_dim//self.factor_num
        self.emb_dim=self.factor_num*self.factor_dim # 保证emb_dim/factor_num==factor_dim

        self.batch_size=args.batch_size
        self.regs=eval(args.regs)

        # 定义输入placeholder
        self.users=tf.placeholder(tf.int32,shape=[None,],name='users')
        self.pos_items=tf.placeholder(tf.int32,shape=[None,],name='pos_items')
        self.neg_items=tf.placeholder(tf.int32,shape=[None,],name='neg_items')

        # 初始化模型参数
        self.weights=self._init_weights()

        self._forward()

        # 模型参数量
        self._static_params()

    # 初始化模型参数
    def _init_weights(self):
        all_weights=dict()

        initializer=tensorflow.contrib.layers.xavier_initializer()
        if(self.pretrain_data==None):
            all_weights['user_embedding']=tf.Variable(initializer([self.n_users,self.emb_dim]),name='user_embedding')
            all_weights['item_embedding']=tf.Variable(initializer([self.n_items,self.emb_dim]),name='item_embedding')
            print('using xavier initialization')        
        else:
            all_weights['user_embedding']=tf.Variable(initial_value=self.pretrain_data['user_embed'],trainable=True,
                                                      name='user_embedding',dtype=tf.float32)
            all_weights['item_embedding']=tf.Variable(initial_value=self.pretrain_data['item_embed'],trainable=True,
                                                      name='item_embedding',dtype=tf.float32)
            print('using pretrained user/item embeddings')
        return all_weights

    # 构造模型
    def _forward(self):
        # 1 获取最终嵌入
        # 1-1 获取base_factor嵌入
        node_base_embeddings,self.output_factors_distribution = self._factor_dynamic_routing()
        # 1-2 获取high_factor嵌入
        node_high_embeddings=self._get_high_factor_emb(node_base_embeddings)
        self.user_final_embeddings,self.item_final_embeddings=self._get_high_factor_emb(node_base_embeddings)
        # 1-3 拼接成最终嵌入
        self.user_final_embeddings,self.item_final_embeddings=tf.split(
            tf.concat([node_base_embeddings,node_high_embeddings],axis=1),
            [self.n_users,self.n_items],axis=0)

        # 2 查嵌入表获得u i表示
        u_e=tf.nn.embedding_lookup(self.user_final_embeddings,self.users) 
        pos_i_e=tf.nn.embedding_lookup(self.item_final_embeddings,self.pos_items)
        neg_i_e=tf.nn.embedding_lookup(self.item_final_embeddings,self.neg_items)
        # 3 预测评分
        self.batch_predictions=tf.reduce_sum(tf.multiply(u_e,pos_i_e),axis=1) # [N,1]
        # 4 构造cf损失函数
        self.mf_loss,self.reg_loss=self._creat_cf_loss(u_e,pos_i_e,neg_i_e)
        # 4 构造factor独立性损失函数
        if(self.regs[1]<1e-5):
            self.factor_loss=tf.constant(0.0)
        else:
            self.factor_loss=self._creat_factor_loss_cos(tf.split(self.user_final_embeddings,self.factor_num,axis=1),self.users)+\
                                    self._creat_factor_loss_cos(tf.split(self.item_final_embeddings,self.factor_num,axis=1),self.pos_items)+\
                                    self._creat_factor_loss_cos(tf.split(self.item_final_embeddings,self.factor_num,axis=1),self.neg_items)

        self.loss=self.mf_loss+self.reg_loss+self.factor_loss

        self.opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

 
    # 模型主体架构 在每个factor上按照score进行动态路由
    def _factor_dynamic_routing(self):
        # 拼接u i 获取所有节点的整个嵌入表
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0) # [U+I,k]
        all_embeddings = [ego_embeddings]

        # 每层卷积时的factor_num 路由机制的迭代次数
        factor_num_layer_list=[self.factor_num,self.factor_num,self.factor_num,self.factor_num]
        iter_num_layer_list=[self.n_iteration,self.n_iteration,self.n_iteration,self.n_iteration] 
        # 最终每个Edge在各个factor上的分布
        output_factors_distribution = []
        # 初始在各个factor上的分布 [4,Edge]
        A_values = tf.ones(shape=[self.factor_num, len(self.all_h_list)])

        # 【step 1】：做L层卷积 每层卷积迭代t次
        for i in range(self.layer_num):
            factor_num_cur=factor_num_layer_list[i]
            iter_num_cur=iter_num_layer_list[i]
            # 把整个嵌入划分成factor_num个子嵌入表 每个factor上的维度是factor_dim
            ego_factor_emb_list=tf.split(all_embeddings[-1],factor_num_cur,axis=1)
            # 每层卷积需要一个存放H_t的变量
            ego_factor_emb_list_t=[x for x in ego_factor_emb_list]

            # 【step 2】：动态路由 迭代iter_num次
            for t in range(iter_num_cur):
                # 每次动态路由需要存放A_values需要更新的值
                A_factors_update=[]
                # 通过上一步（初始的）A_values 计算出在每个factor上的A_score A_score_Laplace
                A_factors, D_col_factors, D_row_factors=self._convert_A_values_to_A_factors_tensor(factor_num_cur, A_values, pick=False)

                for k in range(factor_num_cur):
                    # 【step 2-1】：根据 H_0 和 S_t 计算 H_t
                    # D_col_factors[i] * A_factors[i] * D_col_factors[i] 为拉普拉斯矩阵
                    # ego_factor_emb_list相当于u_0，ego_factor_emb_list_t相当于u_t
                    ego_factor_emb_list_t[k]=tf.sparse.sparse_dense_matmul(D_col_factors[k],ego_factor_emb_list[k])
                    ego_factor_emb_list_t[k]=tf.sparse.sparse_dense_matmul(A_factors[k],ego_factor_emb_list_t[k])
                    ego_factor_emb_list_t[k]=tf.sparse.sparse_dense_matmul(D_col_factors[k],ego_factor_emb_list_t[k])
                    
                    # 【step 2-2】：迭代更新score 根据 H_t H_0 和 S_t 计算 S_t+1
                    # 取所有Edge的头节点
                    head_node_embeddings=tf.nn.embedding_lookup(ego_factor_emb_list_t[k],self.all_h_list) # [Edge,factor_dim]
                    # 取所有Edge的尾节点
                    tail_node_embeddings=tf.nn.embedding_lookup(ego_factor_emb_list[k],self.all_t_list) # [Edge,factor_dim]
                    # 将add score限制在(-1,1)之间
                    head_node_embeddings = tf.math.l2_normalize(head_node_embeddings, axis=1)
                    tail_node_embeddings = tf.math.l2_normalize(tail_node_embeddings, axis=1)
                    
                    ## 更新score
                    #A_factors[k]=A_factors[k]+tf.reduce_sum(tf.multiply(head_node_embeddings,tf.nn.tanh(tail_node_embeddings)),keepdims=False) # [Edge]+[Edge]
                    A_factors_update.append(tf.reduce_sum(tf.multiply(head_node_embeddings,tf.nn.tanh(tail_node_embeddings)),axis=1,keepdims=False)) # [Edge]

                # 【step 2-3】：完成【step 2-1】【step 2-2】
                # 因为A_factor_score在上述的更新中是按照factor一个一个更新的（A_factor_score的一行）
                A_factors_update_values=tf.stack(A_factors_update,axis=0) # [Eage] -> [factor_num,Edge]
                A_values+=A_factors_update_values

                # 存储最终的factor_score分布
                if(t==iter_num_cur-1):
                    output_factors_distribution.append(A_factors)

            # 【step 2-4】：完成一层的卷积操作
            all_embeddings.append(tf.concat(ego_factor_emb_list_t,axis=1))

        # 把各层得到的嵌入相加再拆出user item emb
        all_embeddings = tf.stack(all_embeddings, 1) # -> [U+I,layer_num+1,emb_dim]
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False) # -> [U+I,emb_dim]


        return all_embeddings,output_factors_distribution

    # 传入所有factor的A_values 计算每个factor下的A_score以及A_score的拉普拉斯矩阵
    def _convert_A_values_to_A_factors_tensor(self, f_num, A_factor_values, pick=True):
        """
        f_num:factor数量
        A_factor_values:A值 [f_num,Edge]
        pick:True,缩小最弱的factor score；False,不对A_value进行修改
        --------------
        return:
        A_factors: factor的softmax之后的A_values(sp tensor) list
        D_col_factors:factor的softmax之后的对每一行求和的sum 拉普拉斯算子(sp tensor) list
        D_row_factors:factor的softmax之后的对每一列求和的sum 拉普拉斯算子(sp tensor) list
        """
        A_factors = []
        D_col_factors = []
        D_row_factors = []
        # 邻接矩阵（所有Edge）的头尾节点
        A_indices = np.mat([self.all_h_list, self.all_t_list]).transpose() # [2,Edge] -> [Edge,2]
        # D_indices是[U+I,U+I]矩阵中对角线的idx
        D_indices = np.mat([list(range(self.n_users+self.n_items)), list(range(self.n_users+self.n_items))]).transpose() # [2,U+I] -> [U+I,2]

        # 【step 1】：对于A按照factor维度进行softmax
        if pick:
            A_factor_scores = tf.nn.softmax(A_factor_values, 0)
            min_A = tf.reduce_min(A_factor_scores, 0) # 对于每个边 找到所有factor中最小的score [f_num,Edge]
            index = A_factor_scores > (min_A + 0.0000001)
            index = tf.cast(index, tf.float32)*(self.pick_level-1.0) + 1.0  # adjust the weight of the minimum factor to 1/self.pick_level

            A_factor_scores = A_factor_scores * index
            A_factor_scores = A_factor_scores / tf.reduce_sum(A_factor_scores, 0)
        else:
            A_factor_scores = tf.nn.softmax(A_factor_values, 0)
        
        # 【step 2】：对于每个factor将list的score转化为tf.sparse形式
        for i in range(0, f_num):
            # 【2-1】:提取factor i 对应的score行 并转化为稀疏tensor
            A_i_scores = A_factor_scores[i] # 1维 [Edge]
            # [Edge,2] [Edge] () -> tensor [U+I,U+I]
            A_i_tensor = tf.SparseTensor(A_indices, A_i_scores, self.A_in_shape)

            # 【2-2】:计算每个factor下的拉普拉斯矩阵
            # 计算当前factor i的score的度（准确的说，是sum A value）
            # 分别是：求每一行的和 求每一列的和
            D_i_col_scores = 1/tf.math.sqrt(tf.sparse_reduce_sum(A_i_tensor, axis=1,keepdims=False)) # [U+I,]
            D_i_row_scores = 1/tf.math.sqrt(tf.sparse_reduce_sum(A_i_tensor, axis=0,keepdims=False)) # [,U+I]
            
            # 【2-3】:重新修正两个Laplace 稀疏tensor 的形状
            # 形状由 [U+I] -> [U+I,U+I]，变成一个只有对角线非0的矩阵
            D_i_col_tensor = tf.SparseTensor(D_indices, D_i_col_scores, self.A_in_shape) # [U+I,U+I]
            D_i_row_tensor = tf.SparseTensor(D_indices, D_i_row_scores, self.A_in_shape) # [U+I,U+I]

            A_factors.append(A_i_tensor)
            D_col_factors.append(D_i_col_tensor)
            D_row_factors.append(D_i_row_tensor)

        # return a (n_factors)-length list of laplacian matrix
        return A_factors, D_col_factors, D_row_factors

    # 根据base_factor_emb获得high_factor_emb
    def _get_high_factor_emb(self,H_base):
        H_high=tf.layers.dense(H_base,
                                self.emb_dim,
                                use_bias=True,
                                kernel_regularizer=tf.keras.regularizers.l2(self.regs[2]),
                                bias_regularizer=tf.keras.regularizers.l2(self.regs[2]),
                                activation='relu',
                                name="high_factor_embedding_layer",
                                reuse=tf.AUTO_REUSE,
                                ) # [U+I,emb_dim]
        #H_final=tf.concat([H_base,H_high],axis=1) # [U+I,2*emb_dim]

        #user_final_embeddings,item_final_embeddings=tf.split(H_final,[self.n_users,self.n_items],axis=0)
        return H_final

    # 将邻接矩阵分成fold块
    def _split_A_hat(self, X):
        """
        X:一个稀疏矩阵
        return 分成fold块[每一块矩阵是原矩阵的几行]的稀疏矩阵list
        """
        A_fold_hat = []

        # 按整行来对于矩阵分块 将N行的矩阵分成fold块
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    # 将X转化成稀疏矩阵
    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        X:sp的矩阵
        """
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose() # [2,N] -> [N,2]
        # 创建稀疏矩阵、indices是index[row col]、data是值
        # 注：加上shape 因为可能有全0行 全0列 所以coo形式必须加shape
        return tf.SparseTensor(indices, coo.data, coo.shape)

    # 构造cf损失函数
    def _creat_cf_loss(self,u_e,pos_i_e,neg_i_e):

        pos_scores=tf.reduce_sum(tf.multiply(u_e,pos_i_e),axis=1) # [N,1,K] [N,1,K] -> [N,1,K] -> [N,1]
        neg_scores=tf.reduce_sum(tf.multiply(u_e,neg_i_e),axis=1) # [N,1,K] [N,1,K] -> [N,1,K] -> [N,1]

        regular=tf.nn.l2_loss(u_e)+tf.nn.l2_loss(pos_i_e)+tf.nn.l2_loss(neg_i_e) # 1

        diff=tf.log(tf.nn.sigmoid(pos_scores-neg_scores)) # [N,1]

        mf_loss=-(tf.reduce_mean(diff)) # [N,1] -> 1
        reg_loss=self.regs[0]*regular

        return mf_loss,reg_loss

    # 构造独立性损失函数[cos距离]
    def _creat_factor_loss_cos(self,factor_emb_list,ids):
        emb_list=[]
        for emb in factor_emb_list:
            emb_list.append(tf.nn.embedding_lookup(emb,ids))
       
        def get_cos(X_list):
            sum_square=tf.square(tf.add_n(X_list)) # [N,k]
            square_sum=[]
            for X in X_list:
                square_sum.append(tf.square(X)) # [N,k]
            square_sum=tf.square(tf.add_n(square_sum))

            return 0.5*tf.sqrt(tf.reduce_sum(tf.square(sum_square-square_sum)))

        # 计算cos距离
        factor_loss=get_cos(emb_list)
        factor_loss=self.regs[1]*factor_loss

        return factor_loss

    # 统计参数量
    def _static_params(self):
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

    # train
    def train(self,sess,feed_dict):
        return sess.run([self.opt,self.loss,self.mf_loss,self.factor_loss],feed_dict)

    # predict
    def predict(self,sess,feed_dict):
        batch_predictions=sess.run(self.batch_predictions,feed_dict)
        return batch_predictions

    # 
