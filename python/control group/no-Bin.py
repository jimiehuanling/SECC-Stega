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

def predict(model, groups, char, LTstr, top_k=None, hidden_size=None):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()  
    vocab_size = 65
    with torch.no_grad():
        char = np.array([char])  
        char = char.reshape(-1, 1) 
        char_encoding = model.onehot_encode(char)  
        char_encoding = char_encoding.toarray()
        char_encoding = char_encoding.reshape(1, 1, -1) 
        char_tensor = torch.tensor(char_encoding, dtype=torch.float32)  
        char_tensor = char_tensor.to(device)

        out, hidden_size = model(char_tensor, hidden_size) 

        probs = F.softmax(out, dim=1).squeeze()  

        # 选择概率top_K
        # indices为top_k的索引
        if top_k is None:  
            indices = np.arange(vocab_size)
        else:
            probs, indices = probs.topk(top_k)
            indices = indices.cpu().numpy()
        probs = probs.cpu().numpy()
        l_indices = []
        for i in indices:
            l_indices.append(i)
        
        
        string_index = LTstr
        int_index = utils.binary_to_decimal(string_index)
        # print('='*100)
        # print('LTstr:',LTstr)
        # print('int:', int_index)
        # char_index = np.random.choice(indices, p=probs/probs.sum())  
        group = groups[int_index]

        char_prob = []
        for v in group :
            
            i = model.char_int[v]
            try:
                index = l_indices.index(i)
                char_prob.append(probs[index])
            except ValueError:
                print(f"元素不在列表中")

        if char_prob[0]>=char_prob[1]:
            char = group[0]
        else:
            char = group[1]

        # print('='*100) 
        # print('group:',group)
        # print('pros:',char_prob)
        # print('char:',char)
        # char_index = indices[int_index]
        # char = model.int_char[char_index]
        # pro_index = np.where(indices == char_index)[0][0]
        # print(indices)
        # print(probs)
        # print(char_index)
        # print('the selected char is:',char)
        # print('the prob of the selected word is ',probs[pro_index])
        # print('the prob of the selected word is ',probs[int_index])
    return char, hidden_size


def sample(model, groups, data_matrix, length, top_k=None, sentence="We are", LTpath='newbits.txt'):
    hidden_size = None
    new_sentence = [char for char in sentence]
    for i in range(length):
        next_char, hidden_size = predict(model, groups, new_sentence[-1], data_matrix[i], top_k=top_k, hidden_size=hidden_size)
        new_sentence.append(next_char)
    return "".join(new_sentence)

def extract_from_text(groups, text):
    decode_matrix = []
    for t in text:
        for i, g in enumerate(groups):
            if t in g:
                LT_index = utils.decimal_to_binary(i)
                decode_matrix.append(LT_index)
    return decode_matrix

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
    
    data_matrix = ['1001',
                 '0001',
                 '1000',
                 '0010',
                 '0010',
                 '1010',
                 '0011',
                 '1001',
                 '0111',
                 '1010',
                 '1100',
                 '1011',
                 '0010',
                 '1011',
                 '1111',
                 '0100',
                 '0101',
                 '1111',
                 '1101',
                 '0011',
                 ]
    #================信息嵌入阶段================
    # 根据LT编码结果生成文本
    embed_data = ''
    for i in data_matrix:
        embed_data += i
    
    embed_matrix = [embed_data[i:i + 5] for i in range(0, len(embed_data), 5)]

    new_text = sample(model, groups, embed_matrix, length=len(embed_matrix), top_k=65, sentence=inital_text) 
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
    # 篡改攻击
    extract_matrix1 = extract_from_text(groups, a1)
    # 删除攻击
    extract_matrix2 = extract_from_text(groups, a2)
    # 删除攻击
    extract_matrix3 = extract_from_text(groups, a3)

    extract_data1 = ''
    extract_data2 = ''
    extract_data3 = ''

    for i in extract_matrix1:
        extract_data1 += i
    
    for i in extract_matrix2:
        extract_data2 += i
    
    for i in extract_matrix3:
        extract_data3 += i

    pad_char = '0'
    default_length = 80
    if len(extract_data2) < default_length:
        extract_data2 = extract_data2.ljust(default_length, pad_char)
    else:
        extract_data2 = extract_data2[:default_length]

    print('embed_data:',embed_data)
    print('extract_data1:', extract_data1)
    print('extract_data2:', extract_data2)
    print('extract_data3:', extract_data3)
    diff1 = utils.count_different_characters(embed_data, extract_data1)
    diff2 = utils.count_different_characters(embed_data, extract_data2)
    diff3 = utils.count_different_characters(embed_data, extract_data3)
    ber1 = diff1/len(embed_data)
    ber2 = diff2/len(embed_data)
    ber3 = diff3/len(embed_data)
    print('ber1:',ber1)
    print('ber2:',ber2)
    print('ber3:',ber3)
    #=============================================

if __name__ == "__main__":
    main()

