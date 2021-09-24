"""
- 2021/9/22

"""
import os
import sys
import random
import numpy as np
import pandas as pd
import tensorflow
import tensorflow.compat.v1 as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from time import time,strftime,localtime # 要用time()就不能import time了
from utils.parser import parse_args
from utils.helper import *
from utils.metrics import evaluate_performance

from utils.load_data import Data_Sequence

from model.SASRec import SASRec

SEED=2021
tf.set_random_seed(SEED)

def main():
    # =================1：构造数据集===================
    t0=time()
    args = parse_args()
    data_path='{}experiment_data/{}/{}_{}/'.format(args.data_path,args.dataset,args.prepro,args.test_method)
    data_generator=Data_Sequence(data_path,args.batch_size)
    data_generator.load_dataset()
    data_generator.print_data_info()
    t1=time()
    print('load dataset cost [{:.1f}s]'.format(t1-t0))
    # =================2：构造模型=====================
    data_config=dict()
    data_config['n_items']=data_generator.n_items
    model=SASRec(args,data_config)

    # =================3：训练模型=====================
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    loss_log,hit_log,ndcg_log=[],[],[]

    for epoch in range(args.epochs):
        t0=time()
        loss=0.
        batch_num=args.epochs//args.batch_size if args.epochs%args.batch_size==0 else args.epochs//args.batch_size+1
        for batch_idx in range(batch_num):
            batch_feed_data=data_generator.generate_train_batch(batch_idx)
            batch_feed_dict=data_generator.generate_train_feed_dict(model,batch_feed_data,args.drop_rate)
            _,batch_loss=model.train(sess,batch_feed_dict)
            loss+=batch_loss

        t1=time()
        loss=loss/batch_num
        loss_log.append(loss)
        show_loss_step=1
        show_val_step=5
        
        if((epoch+1)%show_loss_step==0):
            print('epoch:{}[{:.1f}s],loss:{:.5f}'.format(epoch,t1-t0,loss))
        if((epoch+1)%show_val_step==0):
            test_feed_dict=data_generator.generate_test_feed_dict(model)
            predict_score=model.predict(sess,test_feed_dict)
            predict_score=np.array(predict_score)
            hit,ndcg=evaluate_performance(predict_score,10)
            print('epoch:{},hit:{:.5f},ndcg:{:.5f}'.format(epoch,hit,ndcg))

if __name__=='__main__':
    main()
