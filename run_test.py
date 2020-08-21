import os
import random

import torch
import numpy as np
import pandas as pd
import torch.optim as optim

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, 
                             accuracy_score)


from sacred import Experiment
from sacred.observers import MongoObserver

from looper import Looper
from tab_net import TabNet, LinearM

from data_processor import ( PandasCatLoader, 
                             get_feature_sizes, 
                             LabelEnc)


"""
Main script to run tests.

Test results are stored with sacred (MongoDb) and available with Omniboard or Sacred board

1. Train linear model
2. Train NN

Two NN are avaliable 
- MLP + embeddings and residual connections
- linear (simple 3l MLP)


"""

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

TRAIN_PATH = '../../../data_main/fraud/TRAIN.csv'
VAL_PATH = '../../../data_main/fraud/VAL.csv'

params = dict()
params['TARGET'] = 'isFraud'
params['embed_scaler'] = 1/1.6   # division value, AutoGluone uses it as a multiplicative value
params['embed_exponent'] = 0.65  # embeded_scaler & embed_exponent are used to determin embedding size for categorical feature
params['min_embed_size'] = 2
params['net_type'] = 'embeddings'  # other values 'linear'
params["layer_norm"] = True
params["batch_norm"] = False
params["mid_features"] = 512  # Mid layer size in residual block, size before output
params["num_residual"] = 2    # Number of residual blocks each block has "mid_features" input and output
params["drop_out"] = 0
params['out_size'] = 2  # Num categories or 1 for regression 
params['cat_th'] = 50   # Maximum cardinality for categorical features
params['batch_size'] = 512
params['lr'] = 0.0004
params['tags'] = ['Tabular NET','Fraud']  # tags for sacred / omniboard logging
params['drop_na'] = 0.6  # Drop column if more than 60% of values are Na                 

EXP_NAME = ' '.join([*params['tags'],params['net_type'],'_Fraud'])

ex = Experiment(EXP_NAME)
ex.add_config(params)
ex.observers.append(MongoObserver.create(url="mongodb://mongo_user:pass@dockersacredomni_mongo_1/", db_name="sacred"))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    

# Median imputer
# TODO: use mean for real values and median for categorical

def na_imputer(data, prefix='_col_na_binary'):
    for col in data.columns:
        if not data[col].dtype.kind in 'biufc':
            continue 
        num_na = data[col].isna().sum()
        if num_na > 0:
            data[col].fillna(data[col].median(), inplace=True)
            print(f'Nan in {col} {num_na}, imputing ...')
            data[col+prefix] = data[col].isna().astype(int)
    return data


class RealNormalizer:
    def __init__(self):
        self.mapper = dict()
    
    def fit_transform(self, data, columns_to_normalize):
        for col in columns_to_normalize:
            mean, std = data[col].mean(), data[col].std()
            self.mapper[col] = (mean, std)
            if std == 0: continue
            data[col] = (data[col] - mean)/std
        
        return data
    
    def transform(self, data):
        for col in self.mapper:
            mean, std = self.mapper[col]
            if std == 0: continue
            data[col] = (data[col] - mean)/std
        
        return data

               
def data_processing(TRAIN_PATH, VAL_PATH, params):
    train = pd.read_csv(TRAIN_PATH)#.sample(10000)
    val = pd.read_csv(VAL_PATH)#.sample(3000)
    
    columns_na = train.loc[0,train.isna().sum()>params['drop_na']*(train.shape[0])].index.to_list()
    if len(columns_na)>0:
        train = train.drop(columns_na, axis=1)
        val = val.drop(columns_na, axis=1)
    
    train = na_imputer(train)
    val = na_imputer(val)
    

    if params['net_type'] == 'linear': 
        params = get_feature_sizes(train.drop(params['TARGET'], axis=1), 
                                   params,
                                   th=1)
    else:
        params = get_feature_sizes(train.drop(params['TARGET'], axis=1), 
                                   params,
                                   th=params['cat_th'])
    cat_columns = list(params['cat_embed_sizes'].keys())
    if len(cat_columns)>0:
        label_enc = LabelEnc()
        train = label_enc.fit_transform(train, cat_columns)
        val = label_enc.transform(val)

    if len(params['high_var_objects'])>0:
        train = train.drop(params['high_var_objects'], axis=1)
        val = val.drop(params['high_var_objects'], axis=1)
   

    normalizer = RealNormalizer()
    train = normalizer.fit_transform(train, params['real_features'])
    val = normalizer.transform(val)
    
    return train, val, params


def make_dataset(train_frame, val_frame, params, shuffle=True, batch_size=512):
    loaders = []
    train_frame.reset_index(drop=True, inplace=True)
    val_frame.reset_index(drop=True, inplace=True)
    for cur_data in [train_frame, val_frame]:
        source = PandasCatLoader(cur_data, params)
        loader = torch.utils.data.DataLoader(
            source, batch_size=batch_size, num_workers=12, shuffle=shuffle
        )
        loaders.append(loader)

    return loaders


@ex.capture
def add_metrics(_run, key, value, order):
    ex.log_scalar(key, float(value), order)
    
    
@ex.main
def main(params):
    set_seed(44)
    print('Traning net type: ',params['net_type'])
    train, val, params = data_processing(TRAIN_PATH, VAL_PATH, params)
    train_set, val_set =  make_dataset(train, val, params, shuffle=True, batch_size=params['batch_size'])
        
    # train logistic regression for baseline score
    print(f'Training linear model')
    clf = LogisticRegression(random_state=0, max_iter = 300)
    clf.fit(train.drop(params['TARGET'], axis=1), train[params['TARGET']])
    preds = clf.predict_proba(val.drop(params['TARGET'], axis=1))
    roc_auc = roc_auc_score(val[params['TARGET']], preds[:,1])
    add_metrics(key='Linear model', value=roc_auc, order=1)
    
    if params['net_type'] == 'linear': model = LinearM(params)
    if params['net_type'] == 'embeddings': model = TabNet(params)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # TODO: move problem type to params
    
    looper = Looper(device, 'cls', add_metrics)
    
    model.cuda()
    torch.save(model.state_dict(),EXP_NAME+'.pth')
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-06)
    loss_fn = torch.nn.CrossEntropyLoss()

    val_funcs = [roc_auc_score, accuracy_score]
    print(f'Training neural {params["net_type"]}')
    looper.train(model, 
            10,
            train_set, 
            val_set,
            optimizer,
            loss_fn,
            val_funcs)
    
    torch.save(model.state_dict(),EXP_NAME+'.pth')
    return float(max(looper.history()['roc_auc_score']))
    

if __name__ == '__main__':
    run = ex._create_run()
    run(run.config)
