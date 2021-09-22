"""
- 运行
- 2021/5/9
"""
import os
import sys
import numpy as np
import tensorflow.compat.v1 as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from utils.helper import *
from utils.parser import parse_args
from time import time,strftime,localtime # 要用time()就不能import time了
from utils.batch_test import test

from model.BPRMF import BPRMF
from model.NeuMF import NeuMF
from model.DisenMF import DisenMF
from model.LightGCN import LightGCN
from model.NAIS import NAIS
from model.DGCF import DGCF
from model.GNUD import GNUD
from model.PDCF import PDCF

from utils.load_data import Data
from utils.load_data_history import Data_History
from utils.load_data_graph import Data_Graph

SEED=2021
tf.set_random_seed(SEED)

# 加载预训练的user/item 嵌入
def load_pretrain_data(args):
    pre_model='bprmf'

    pretrain_path="{}{}/{}_{}/{}/pretrain_tensor/{}.npz".format(args.proj_path,args.dataset,args.prepro,args.test_method,'BPRMF',pre_model)
    #pretrain_data=np.load(pretrain_path)
    try:
        pretrain_data=np.load(pretrain_path)
        print(f'load the pretrained {pre_model} model params')
    except Exception:
        pretrain_data=None
    return pretrain_data

def main():
    t0=time()
    args = parse_args()
    data_path='{}experiment_data/{}/{}_{}/'.format(args.data_path,args.dataset,args.prepro,args.test_method)
    # 加载数据类 生成batch_data
    if(args.model_type in ['bprmf','neumf','DisenMF']):
        data_generator=Data(data_path,args.batch_size)
        data_config=dict()
        data_config['n_users']=data_generator.n_users
        data_config['n_items']=data_generator.n_items
    elif(args.model_type in ['LightGCN']):
        data_generator=Data_Graph(data_path,args.batch_size)
        data_config=dict()
        data_config['n_users']=data_generator.n_users
        data_config['n_items']=data_generator.n_items
        adj_matrix, norm_adj_matrix, mean_adj_matrix=data_generator.get_adj_matrix()
        data_config['norm_adj']=norm_adj_matrix
    elif(args.model_type in ['DGCF','GNUD','PDCF']):
        data_generator=Data_Graph(data_path,args.batch_size)
        data_config=dict()
        data_config['n_users']=data_generator.n_users
        data_config['n_items']=data_generator.n_items
        adj_matrix,_,_=data_generator.get_adj_matrix()
        data_config['norm_adj']=adj_matrix
        all_h_list,all_t_list=get_head_tail_list(adj_matrix)
        data_config['all_h_list']=all_h_list
        data_config['all_t_list']=all_t_list
        

    elif(args.model_type in ['NAIS']):
        data_generator=Data_History(data_path,args.batch_size)
        data_config=dict()
        data_config['n_users']=data_generator.n_users
        data_config['n_items']=data_generator.n_items

    # 构造pretrain_data
    # 加载预训练模型参数1：预训练的嵌入
    if args.pretrain in [1]:
        pretrain_data=load_pretrain_data(args)
    else:
        pretrain_data=None

    # 构造模型
    if(args.model_type=='bprmf'):
        model=BPRMF(data_config,pretrain_data,args)
    elif(args.model_type=='neumf'):
        model=NeuMF(data_config,pretrain_data,args)
    elif(args.model_type=='DisenMF'):
        model=DisenMF(data_config,pretrain_data,args)
    elif(args.model_type=='LightGCN'):
        model=LightGCN(data_config,pretrain_data,args)
    elif(args.model_type=='NAIS'):
        model=NAIS(data_config,pretrain_data,args)
    elif(args.model_type=='DGCF'):
        model=DGCF(data_config,pretrain_data,args)
    elif(args.model_type=='GNUD'):
        model=GNUD(data_config,pretrain_data,args)
    elif(args.model_type=='PDCF'):
        model=PDCF(data_config,pretrain_data,args)

    """
    **********************************************
    初始化sess
    """
    saver = tf.train.Saver()
    # 当前的模型参数要不要都保存
    if(args.save_model_flag==1):
        model_parameters_path="{}{}/{}_{}/{}/model_parameters/".format(args.proj_path,args.dataset,args.prepro,args.test_method,args.model_type)
        save_saver = tf.train.Saver(max_to_keep=1)
        ensureDir(model_parameters_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    """
    ********************************************** 
    使用保存的模型参数继续训练
    """
    # 加载预训练模型参数2：tf保存的整个模型参数
    if args.pretrain==2:
        model_parameters_path="{}{}/{}_{}/{}/model_parameters/".format(args.proj_path,args.dataset,args.prepro,args.test_method,args.model_type)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_parameters_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            #sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path) # 从tf model path中读取所有变量
            print('load the pretrained model parameters from: ', model_parameters_path)

            # *********************************************************
            # 获取保存的模型在test上的表现
            if args.pretrain_report == 1:
                users_to_test = list(data_generator.test_user_dict.keys())
                ret = test(sess, model,data_generator, users_to_test, drop_flag=False, batch_test_flag=False)
                cur_best_pre_0 = ret['recall'][args.best_k_idx]

                pretrain_recall_str="pretrained model: recall:[{}] ndcg:[{}] hit:[{}] precision:[{}] auc:[{}]".\
                        format(
                                convert_list_2_str(ret['recall'],5),convert_list_2_str(ret['ndcg'],5),
                                convert_list_2_str(ret['hit_ratio'],5),convert_list_2_str(ret['precision'],5),
                                str(ret['auc'])[:5],
                            )
                print(pretrain_recall_str)
        else:
            #sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('can not find saved model parameters.')
    else:
        cur_best_pre_0 = 0.
        print('without pretraining.')

    """
    ********************************************** 
    训练
    """
    loss_log,pre_log,rec_log,ndcg_log,hit_log,auc_log=[],[],[],[],[],[]
    stopping_step=0
    should_stop=False

    # 训练epoch次数 遍历每个epoch
    for epoch in range(args.epoch):
        t1=time()
        loss,mf_loss,reg_loss=0.,0.,0.
        n_batch=data_generator.n_train//args.batch_size+1
        for idx in range(n_batch):
            # 获取batch数据
            batch_data=data_generator.generate_train_cf_batch(idx)
            # 构造feed_fict
            feed_dict=data_generator.generate_train_feed_dict(model,batch_data)
            # run
            _,batch_loss,batch_mf_loss,batch_reg_loss=model.train(sess,feed_dict)
            loss+=batch_loss
            mf_loss+=batch_mf_loss
            reg_loss+=batch_reg_loss

        loss/=n_batch
        mf_loss/=n_batch
        reg_loss/=n_batch
        #print(mf_loss)
        #print(reg_loss)

        #loss_log.append(loss)

        if(np.isnan(loss)):
            print('ERROR:loss is nan')
            #sys.exit()

        # 每隔show_step的epoch 进行test计算评价指标
        show_step=20
        if(epoch+1)%show_step!=0:
            # 每隔verbose的epoch 输出当前epoch的loss信息
            if(args.verbose>0 and epoch%args.verbose==0):
                print_str='Epoch {} [{:.1f}s]: train loss==[{:.5f}={:.5f}+{:.5f}]'\
                    .format(epoch,time()-t1,loss,mf_loss,reg_loss)
                print(print_str)

            continue

        """
        **********************************************
        测试 计算评价指标并记录 （每隔show_step步）
        """
        t2=time()
        users_to_test = list(data_generator.test_user_dict.keys())
        ret = test(sess, model,data_generator, users_to_test, drop_flag=False, batch_test_flag=False)

        # 记录固定epoch时的test评价指标
        print('TODO:test')
        t3=time()

        loss_log.append(loss)
        pre_log.append(ret['precision'])
        rec_log.append(ret['recall'])
        ndcg_log.append(ret['ndcg'])
        hit_log.append(ret['hit_ratio'])
        auc_log.append(ret['auc'])
        
        test_recall_str="Epoch:{} [{:.1f}s],loss=[{:.4f}={:.4f}+{:.4f}] recall:[{}] ndcg:[{}] hit:[{}] auc:[{}]".\
                        format(
                                epoch+1,t3-t2,loss,mf_loss,reg_loss,
                                convert_list_2_str(rec_log[-1],5),convert_list_2_str(ndcg_log[-1],5),
                                convert_list_2_str(hit_log[-1],5),str(auc_log[-1])[:5],
                            )
        print(test_recall_str)

        """
        ********************************************** 
        并且如果评价指标上升的话 保存模型参数（save_model_flag==1）
        """
        if(ret['recall'][args.best_k_idx]>cur_best_pre_0):
            cur_best_pre_0=ret['recall'][args.best_k_idx]
            if(args.save_model_flag==1):
                save_saver.save(sess, model_parameters_path, global_step=epoch)
                print('save the model parameters in path: ', model_parameters_path)
            if(args.save_model_tensor_flag==1):
                temp_save_path = "{}{}/{}_{}/{}/pretrain_tensor/{}.npz".format(args.proj_path,args.dataset,args.prepro,args.test_method,args.model_type,args.model_type)
                ensureDir(temp_save_path)
                model.save_tensor(sess,temp_save_path)
                #try:
                #    tmp_path="{}{}/{}_{}/{}/pretrain_tensor/".format(args.proj_path,args.dataset,args.prepro,args.test_method,args.model_type)
                #    ensureDir(tmp_path)
                #    model.save_tensor(tmp_path+args.model_type+'.npz')
                #except:
                #    print('failed:save the model tensor ')

    """
    **********************************************
    保存模型的表现
    """
    recs = np.array(rec_log)
    pres = np.array(pre_log)
    ndcgs = np.array(ndcg_log)
    hit = np.array(hit_log)
    #auc=np.array(auc_loger)
    
    # 选出recall@k最佳的idx 
    best_rec_k = max(recs[:,args.best_k_idx])
    idx = list(recs[:, args.best_k_idx]).index(best_rec_k)
    best_epoch=show_step*(idx+1)

    final_info="Best Epoch:{} [{:.1f}s] recall:[{}] ndcg:[{}] hit:[{}] precison:[{}]".\
                format(
                       best_epoch,time()-t0,
                       convert_list_2_str(rec_log[idx],7),
                       convert_list_2_str(ndcg_log[idx],7),
                       convert_list_2_str(hit_log[idx],7),
                       convert_list_2_str(pre_log[idx],7),
                       )
    print(final_info)
    
    cur_time=strftime("%Y-%m-%d %H:%M:%S", localtime()) 
    
    save_path="{}/{}/{}_{}/{}/".format(args.proj_path,args.dataset,args.prepro,args.test_method,args.model_type)

    def get_args_str():
        str_list=[]
        str_list.append('model_type:{}'.format(args.model_type))
        str_list.append('model_des:{}'.format(args.model_des))
        str_list.append('embed_size:{}'.format(args.embed_size))
        str_list.append('regs:{}'.format(args.regs))
        str_list.append('lr:{}'.format(args.lr))
        str_list.append('epoch:{}'.format(args.epoch))
        str_list.append('batch_size:{}'.format(args.batch_size))
        str_list.append('pretrain:{}'.format(args.pretrain))
        str_list.append('Ks:{}'.format(args.Ks))
        str_list.append('best_k_idx:{}'.format(args.best_k_idx))

        return str_list

    params_list=get_args_str()

    # 保存实验的参数设置和最终表现 
    ensureDir(save_path+'result.txt')
    f = open(save_path+'result.txt', 'a') # 追加
    f.write("**********************************************\n")
    f.write(cur_time+"\n")
    f.write("**********************************************\n")
    f.write(" ".join(params_list)+"\n")
    f.write(final_info+"\n")
    f.close()

    # 保存记录
    ensureDir(save_path+'log.txt')
    f = open(save_path+'log.txt', 'a') # 追加

    f.write("**********************************************\n")
    f.write(cur_time+"\n")
    f.write("**********************************************\n")
    f.write(str(loss_log)+"\n")
    f.write(str([x.tolist() for x in rec_log])+"\n")
    f.write(str([x.tolist() for x in ndcg_log])+"\n")
    f.write(str([x.tolist() for x in hit_log])+"\n")
    f.write(str([x.tolist() for x in pre_log])+"\n")
    f.write(str(auc_log)+"\n")
    f.close()

    print('save to {} ok!'.format(save_path+'result.txt'))


if __name__=='__main__':
    main()