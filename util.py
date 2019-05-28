# coding: utf-8
import numpy as np
import pandas as pd
from collections import OrderedDict

def sparse_tuple_from(sequences, dtype=np.int32):
    
    indices = []
    values = []
    for n, seq in enumerate(sequences):
         indices.extend(zip([n]*len(seq), range(len(seq))))
         values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

def transferlabeltoInt(Y):
    seq_list = [[] for i in range(len(Y))]
    for i,seq in enumerate(Y):
        for c in seq:
            if(c=='+'): 
                seq_list[i].append('10')
            elif(c=='-'): 
                seq_list[i].append('11')
            elif(c=='*'): 
                seq_list[i].append('12')
            elif(c=='='): 
                seq_list[i].append('13')
            elif(c=='('): 
                seq_list[i].append('14')
            elif(c==')'): 
                seq_list[i].append('15')
            else: 
                seq_list[i].append(c)
    y=np.asarray(seq_list)
    
    train_targets = sparse_tuple_from(y)
    return train_targets

def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    real_label='0123456789+-*=()'
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor,real_label))
    return result
    
def decode_a_seq(indexes, spars_tensor,real_label):
    decoded = []
    for m in indexes:
        stri = real_label[spars_tensor[1][m]]
        decoded.append(stri)
    return decoded
    
def record_training_data(training_accuracy, validation_accuracy, loss):
    df1 = pd.DataFrame({'key': list(training_accuracy.keys()),'col_training_accuracy': list(training_accuracy.values())})
    df2 = pd.DataFrame({'key': list(validation_accuracy.keys()),'col_validation_accuracy': list(validation_accuracy.values())})
    df3 = pd.DataFrame({'key': list(loss.keys()),'col_loss': list(loss.values())})
    result1 = pd.merge(df3, df2, how='left',on='key')
    result2 = pd.merge(result1, df1, how='left',on='key')
    result2.to_csv('test_result/result.csv')
    
def merge_record_data():
    re = pd.read_csv('test_result/result.csv')
    training_accuracy_before =re.loc[:,['key','col_training_accuracy']]
    validation_accuracy_before =re.loc[:,['key','col_validation_accuracy']]
    loss_before =re.loc[:,['key','col_loss']]
    training_accuracy_before.dropna(axis=0, how='any', inplace=True)
    validation_accuracy_before.dropna(axis=0, how='any', inplace=True)
    loss_before.dropna(axis=0, how='any', inplace=True)
    training_accuracy = OrderedDict(training_accuracy_before.values)
    validation_accuracy = OrderedDict(validation_accuracy_before.values)
    loss_re = OrderedDict(loss_before.values)
    return (training_accuracy, validation_accuracy, loss_re)