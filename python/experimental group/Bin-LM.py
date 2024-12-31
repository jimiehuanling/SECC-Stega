import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
import utils
import train_model_bin as ls
import attack_type
import random

def random_grouping(chars):
    # 检查字符数量是否为65
    if len(chars) != 65:
        raise ValueError("字符数量必须为65")
    # 随机打乱字符列表
    random.shuffle(chars)
    # 分组
    groups = [chars[i:i + 2] for i in range(0, 62, 2)]  
    groups.append(chars[62:])  
    return groups

def main():
    # 超参数
    hidden_size = 512
    num_layers = 2
    batch_size = 128
    seq_len = 100
    epochs = 2
    lr = 0.01
    file_path = 'shakespeare.txt'
    LTcodes_path = 'newbits.txt'
    inital_text = 'We are'
    
    # 读取数据集
    text, vocab = ls.get_vocabulary(file_path)
    voc_list = []
    for i in vocab:
        voc_list.append(i)
    
    groups = random_grouping(voc_list)

    
    #================模型处理阶段===============
    # 建立模型
    model = ls.lstm_model(vocab, hidden_size, num_layers)  
    # 加载模型 
    model.load_state_dict(torch.load("lstm_model.net"))  
    #===========================================
    
    
    #================信息嵌入阶段================
    # 根据LT编码结果生成文本
    
    new_text = ls.sample(model, groups, length=60, top_k=65, sentence=inital_text) 
    print('='*100) 
    print('the generated text is:', new_text)  
    print('the length of the text:', len(new_text))
    #===========================================

    #================文本攻击阶段================
    real_text = new_text[len(inital_text):]
    print('the real text:', real_text)
    print('the length of the real text:', len(real_text))
    # 篡改攻击
    a1 = attack_type.str_change(real_text)
    print('the text after change_attack:', a1)
    print(len(a1))
    # 删除攻击
    a2 = attack_type.str_delete(real_text)
    print('the text after delete_attack:', a2)
    print(len(a2))
    # 换位攻击
    a3 = attack_type.str_swap(real_text)
    print('the text after swap_attack:', a3)
    print(len(a3))
    #============================================
    
    #================信息提取阶段=================
    print('='*100)
    print('篡改攻击')
    # 篡改攻击
    # 提取秘密信息矩阵
    length1 = len(a1)
    extract_matrix1 = ls.extract_from_text(groups, a1)

    # 进行偶校验
    checked_matrix1, index_matrix1 = utils.validate_parity_codes(extract_matrix1)
    # print('extract_matrix:', extract_matrix)
    print('length of extract_matrix:',len(extract_matrix1))
    # print('useful_matrix:', checked_matrix)
    print('row of checked matrix:', len(checked_matrix1))   
    print('length of index matrix:', len(index_matrix1))
    LT_matrix = utils.loadandparity('newbits.txt')
    print('LTmatrix:',LT_matrix)
    print('chmateix:',checked_matrix1)
    
    modified_bitstream1 = '/data/guoyuzhe/rnn-test/LTextract/bin-changed.txt' 
    index_matrix_name1 = '/data/guoyuzhe/rnn-test/LTextract/bin-changed-index.txt'
    checked_str1 = ''
    for str in checked_matrix1:
        checked_str1 = checked_str1 + str
    utils.save_bit_strings_to_file(checked_matrix1, modified_bitstream1)
    # utils.save_string_to_file(checked_str, modified_bitstream)
    utils.save_numeric_array_to_file(index_matrix1, index_matrix_name1)
    print(f"比特字符串已保存到 {modified_bitstream1}")
    print(f"数字数组已保存到 {index_matrix_name1}")
   
    print('='*100)
    print('删除攻击')
    # 删除攻击
    # 提取秘密信息矩阵
    length2 = len(a2)
    extract_matrix2 = ls.extract_from_text(groups, a2)
    # 进行偶校验
    checked_matrix2, index_matrix2 = utils.validate_parity_codes(extract_matrix2)
    # print('extract_matrix:', extract_matrix)
    print('length of extract_matrix:',len(extract_matrix2))
    # print('useful_matrix:', checked_matrix)
    print('row of checked matrix:', len(checked_matrix2))   
    print('length of index matrix:', len(index_matrix2))
    modified_bitstream2 = '/data/guoyuzhe/rnn-test/LTextract/bin-deleted.txt' 
    index_matrix_name2 = '/data/guoyuzhe/rnn-test/LTextract/bin-deleted-index.txt'
    checked_str2 = ''
    for str in checked_matrix2:
        checked_str2 = checked_str2 + str
    utils.save_bit_strings_to_file(checked_matrix2, modified_bitstream2)
    # utils.save_string_to_file(checked_str, modified_bitstream)
    utils.save_numeric_array_to_file(index_matrix2, index_matrix_name2)
    print(f"比特字符串已保存到 {modified_bitstream2}")
    print(f"数字数组已保存到 {index_matrix_name2}")

    print('='*100)
    print('换位攻击')
    # 提取秘密信息矩阵
    length3 = len(a3)
    extract_matrix3 = ls.extract_from_text(groups, a3)

    # 进行偶校验
    checked_matrix3, index_matrix3 = utils.validate_parity_codes(extract_matrix3)
    # print('extract_matrix:', extract_matrix)
    print('length of extract_matrix:',len(extract_matrix3))
    # print('useful_matrix:', checked_matrix)
    print('row of checked matrix:', len(checked_matrix3))   
    print('length of index matrix:', len(index_matrix3))
    LT_matrix = utils.loadandparity('newbits.txt')
    print('LTmatrix:',LT_matrix)
    print('chmateix:',checked_matrix3)
    
    modified_bitstream3 = '/data/guoyuzhe/rnn-test/LTextract/bin-swapped.txt' 
    index_matrix_name3 = '/data/guoyuzhe/rnn-test/LTextract/bin-swapped-index.txt'
    checked_str3 = ''
    for str in checked_matrix3:
        checked_str3 = checked_str3 + str
    utils.save_bit_strings_to_file(checked_matrix3, modified_bitstream3)
    # utils.save_string_to_file(checked_str, modified_bitstream)
    utils.save_numeric_array_to_file(index_matrix3, index_matrix_name3)
    print(f"比特字符串已保存到 {modified_bitstream3}")
    print(f"数字数组已保存到 {index_matrix_name3}")
    #=============================================

if __name__ == "__main__":
    main()

