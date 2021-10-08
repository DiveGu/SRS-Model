"""
参数设置
"""
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run RS-Model.")
    # 路径参数
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='F:/data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='F:/data/experiment_output/',
                        help='Project path.')
    # 数据集 数据处理参数
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset from {lastfm,ml-1m,gowalla, yelp2018, amazon-book}')

    parser.add_argument('--prepro',nargs='?',default='5-core',
                        help='Choose data preprocess from {orgin,x-filter,x-core}')

    parser.add_argument('--test_method',nargs='?',default='tloo',
                        help='Choose a way to get test dataset from {fo,ufo, loo, tloo, tfo}')

    parser.add_argument('--max_len',type=int,default=1,
                        help='user behavior max length')

    parser.add_argument('--train_neg_num',type=int,default=1,
                        help='the neg num when train pos num=1')

    parser.add_argument('--test_neg_num',type=int,default=100,
                        help='the neg num when evaluate test performence')

    # 模型参数
    parser.add_argument('--model_type',nargs='?',default='FPMC',
                        help='Choose a model from {SASRec,GRU4Rec,FPMC}.')
    parser.add_argument('--model_des',nargs='?',default='train_test',
                        help='record something')

    # SASRec参数
    parser.add_argument('--block_num',type=int,default=2,
                        help='the block num')
    parser.add_argument('--head_num',type=int,default=1,
                        help='the head num')

    # GRU4Rec参数
    parser.add_argument('--gru_layers', nargs='?', default='[20,20]',
                        help='gru_layers.')

    parser.add_argument('--embed_size',type=int,default=50,
                        help='CF embedding size')
    parser.add_argument('--regs', nargs='?', default='[0,1e-5,1e-6]',
                        help='Regularization.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='CF batch size.')
    parser.add_argument('--drop_rate', type=float, default=0.5,
                        help='Dropout rate.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Epoch number.')

    parser.add_argument('--verbose', type=int, default=10,
                        help='Display every verbose epoch.')

    # 模型是否需要读取【预训练参数】或者【保存的所有模型参数】
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2:Pretrain with stored models.')
    
    parser.add_argument('--pretrain_report', type=int, default=1,
                        help='show saved pretrained model preformance or not .. recall')
    # 是否需要保存模型的参数
    parser.add_argument('--save_model_flag', type=int, default=0,
                        help='save model parameters or not.')
    # 是否需要保存模型部分参数（嵌入、注意力分布等）
    parser.add_argument('--save_model_tensor_flag', type=int, default=0,
                        help='save some tensors in model or not.')

    # 评价指标K
    parser.add_argument('--Ks', nargs='?', default='[1,5,10,20,50]',
                        help='top K.')
    parser.add_argument('--best_k_idx', type=int, default=2,
                        help='best hit k idx in Ks')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='test rs part or all.')


    return parser.parse_args()

