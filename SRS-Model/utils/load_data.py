'''
加载处理好的数据
生成batch样本
'''
import json
from time import time
import numpy as np
import pandas as pd
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import defaultdict

# 数据集准备
class Data_Sequence():

    def __init__(self, path,batch_size,max_len,
                 train_neg_num,test_neg_num):
        """
        train_neg_num：训练时每个正样本的负采样数量
        test_neg_num：测试pos item时对比的neg items数量
        """
        self.path = path
        self.batch_size = batch_size

        self.train_file = path + '/train.csv'
        self.test_file = path + '/test.csv'
        
        self.max_len=max_len
        self.train_neg_num=train_neg_num
        self.test_neg_num=test_neg_num

    # 负采样 从[0,n_items-1]采样出不在pos_list中的
    def gen_neg(self,pos_list):
        neg_id=pos_list[0]
        while(neg_id in set(pos_list)):
            neg_id=random.randint(0,self.n_items-1)
        return neg_id

    # 负采样 从[0,n_items-1]采样出neg_num个不在pos_list中的
    def gen_neg_list(self,pos_list,neg_num):
        neg_list=[]
        for i in range(neg_num):
            neg_id=self.gen_neg(pos_list+neg_list)
            neg_list.append(neg_id)

        return neg_list


    # 加载数据集
    def load_dataset(self):
        tmp_path='{}dataset-{}-{}-{}.npz'.format(self.path,self.max_len,self.train_neg_num,self.test_neg_num)
        try:
            dataset=np.load(tmp_path)
            def get_ar_list(dataset,name):
                return [dataset[name+'_hist'],dataset[name+'_pos_id'],dataset[name+'_neg_id'],dataset[name+'_uid']]

            self.train=get_ar_list(dataset,'train')
            self.val=get_ar_list(dataset,'val')
            self.test=get_ar_list(dataset,'test')
            self.n_items=int(dataset['n_items'][0])
            self.n_users=int(dataset['n_users'][0])

            # train+val
            for i in range(len(self.train)):
                self.train[i]=np.concatenate((self.train[i],self.val[i]),axis=0)

        except:
            self.train,self.val,self.test,self.n_items,self.n_users=self._create_dataset()
            np.savez(tmp_path, 
                     train_hist=self.train[0],
                     train_pos_id=self.train[1],
                     train_neg_id=self.train[2],
                     train_uid=self.train[3],

                     val_hist=self.val[0],
                     val_pos_id=self.val[1],
                     val_neg_id=self.val[2],
                     val_uid=self.train[3],

                     test_hist=self.test[0],
                     test_pos_id=self.test[1],
                     test_neg_id=self.test[2],
                     test_uid=self.train[3],

                     n_items=np.array([self.n_items]),
                     n_users=np.array([self.n_users]),
                     )
            print('create the dataset in path: ', tmp_path)

    # 创造数据集
    def _create_dataset(self):
        t0=time()
        train_df=pd.read_csv(self.train_file)
        test_df=pd.read_csv(self.test_file)
        n_items=max(train_df['item'].max(),test_df['item'].max())+1
        n_users=max(train_df['user'].max(),test_df['user'].max())+1
        self.n_items=n_items
        self.n_users=n_users
        # 获取test pos id
        test_user_dict=dict(zip(test_df['user'],test_df['item']))

        train_data,val_data,test_data=defaultdict(list),defaultdict(list),defaultdict(list)
        for user_id,df in train_df[['user','item','timestamp']].groupby('user'):
            df=df.sort_values(['timestamp'],ascending=True).reset_index(drop=True) # 按照时间升序排列
            ## just test
            #if(user_id==0):
            #    df.to_csv(self.path+'uid_0.csv')
            #    print(df['item'].tolist())

            pos_list=df['item'].tolist()

            """
            1 pos_list：时间升序排列的item序列 长度为len
            2 pos_list[i]为正样本 pos_list[0:i]为hist序列
            3 所以pos_list能生成 len-1 条样本
            4 test还能生成 1 条样本
            5 需要的负样本数量为 (len-1)*train_neg_num + test_neg_num
            """
            cur_user_neg_list=[self.gen_neg_list(pos_list+[test_user_dict[user_id]],self.train_neg_num) for _ in range(len(pos_list)-1)]
            cur_user_neg_list.append(self.gen_neg_list(pos_list+[test_user_dict[user_id]],self.test_neg_num))
            # i为正样本在序列中的idx
            for i in range(1,len(pos_list)):
                if(i==len(pos_list)-1):
                    val_data['user_id'].append(user_id)
                    val_data['hist'].append(pos_list[:i])
                    val_data['pos_id'].append(pos_list[i])
                    val_data['neg_id'].append(cur_user_neg_list[i-1])
                else:
                    train_data['user_id'].append(user_id)
                    train_data['hist'].append(pos_list[:i])
                    train_data['pos_id'].append(pos_list[i])
                    train_data['neg_id'].append(cur_user_neg_list[i-1])
            # test data
            test_data['user_id'].append(user_id)
            test_data['hist'].append(pos_list)
            test_data['pos_id'].append(test_user_dict[user_id])
            test_data['neg_id'].append(cur_user_neg_list[len(pos_list)-1])


        # 使用n_items作为pad id，按照maxlen进行pad
        train = [pad_sequences(train_data['hist'], maxlen=self.max_len,value=n_items), 
                 np.array(train_data['pos_id']).reshape((-1,1)),
                 np.array(train_data['neg_id']).reshape((-1,self.train_neg_num)),
                 np.array(train_data['user_id']).reshape((-1,1)),]

        val = [pad_sequences(val_data['hist'], maxlen=self.max_len,value=n_items), 
               np.array(val_data['pos_id']).reshape((-1,1)),
               np.array(val_data['neg_id']).reshape((-1,self.train_neg_num)),
               np.array(val_data['user_id']).reshape((-1,1)),]

        test = [pad_sequences(test_data['hist'], maxlen=self.max_len,value=n_items),
                np.array(test_data['pos_id']).reshape((-1,1)),
                np.array(test_data['neg_id']).reshape((-1,self.test_neg_num)),
                np.array(test_data['user_id']).reshape((-1,1)),]

        t1=time()
        print('creat dataset cost:[{:.1f}s]'.format(t1-t0))
        return train,val,test,n_items,n_users

    # 生成训练batch
    def generate_train_batch(self,idx):
        #batch_num=self.train_df//self.batch_size
        if(idx==0):
            # 1 负采样
            #self.df_copy=self.train_df[['user','item']]
            #self.df_copy['neg_item']=self._sample()
            # 2 打乱数据
            state = np.random.get_state()
            np.random.shuffle(self.train[0])
            for ar in self.train[1:]:
                np.random.set_state(state)
                np.random.shuffle(ar)
            x=1

        # 3 生成batch数据
        start=idx*self.batch_size
        end=(idx+1)*self.batch_size
        end=end if end<self.train[0].shape[0] else self.train[0].shape[0]
        batch_data={
            'user_id':self.train[3][start:end],
            'hist':self.train[0][start:end],
            'pos_id':self.train[1][start:end],
            'neg_id':self.train[2][start:end],
        }

        #return self.df_copy[start:end].values
        return batch_data 

    # 生成batch的feed_dict字典
    def generate_train_feed_dict(self,model,batch_data,drop_rate):
        feed_dict={
            model.users:batch_data['user_id'],
            model.hist:batch_data['hist'],
            model.pos_items:batch_data['pos_id'],
            model.neg_items:batch_data['neg_id'],
            model.drop_rate:drop_rate,
        }
        
        return feed_dict

    # 生成test的feed_dict字典
    def generate_test_feed_dict(self,model):
        feed_dict={
            model.users:self.test[3],
            model.hist:self.test[0],
            model.pos_items:np.concatenate((self.test[1],self.test[2]),axis=1),
            model.drop_rate:0.,
        }
        
        return feed_dict

    # 为PopRec生成预测list
    def generate_pop_feed(self):
        i_list=np.concatenate((self.test[1],self.test[2]),axis=1)    
        return list(i_list)


    # 统计训练集、验证集、测试集的数据量
    def print_data_info(self):
        print(self.train[3].shape)
        print('train size:{}'.format(self.train[0].shape))
        print('val size:{}'.format(self.val[0].shape))
        print('test size:{}'.format(self.test[0].shape))

#x=[1]
#x=np.array(x)
#print(type(x[0]))