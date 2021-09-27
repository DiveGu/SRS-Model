import numpy as np
import pandas as pd
from utils.parser import parse_args
from utils.helper import *
from utils.metrics import evaluate_performance
from utils.load_data import Data_Sequence

class PopRec:
    def __init__(self,args,data_config):
        self.train_df=data_config['train_df']
        self._create_pop_score()
       
    # 计算每个item的流行度
    def _create_pop_score(self):
        self.pop_scores=dict(self.train_df['item'].value_counts())

    def predict(self,item_list):
        ret=[]
        for iids in item_list:
            ret.append([self.pop_scores.get(iid,0) for iid in iids])
        return ret


def main():
    args = parse_args()
    data_path='{}experiment_data/{}/{}_{}/'.format(args.data_path,args.dataset,args.prepro,args.test_method)
    data_config=dict()
    data_config['train_df']=pd.read_csv(data_path+'train.csv')
    data_generator=Data_Sequence(data_path,args.batch_size,args.max_len,
                                 args.train_neg_num,args.test_neg_num)
    data_generator.load_dataset()
    # ====================
    model=PopRec(args,data_config)
    predict_score=model.predict(data_generator.generate_pop_feed())
    predict_score=np.array(predict_score)
    hit,ndcg=evaluate_performance(predict_score,10)
    print('hit:{:.5f},ndcg:{:.5f}'.format(hit,ndcg))

main()