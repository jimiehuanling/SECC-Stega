import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
import utils
import train_model_flc as ls
import attack_type

def predict(model, char, LTstr, top_k=None, hidden_size=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        
        string_index = LTstr
        int_index = utils.binary_to_decimal(string_index)
        # print('='*100)
        # print('LTstr:',LTstr)
        # print('int:', int_index)
        # char_index = np.random.choice(indices, p=probs/probs.sum())  
        char_index = indices[int_index]
        char = model.int_char[char_index]
        # pro_index = np.where(indices == char_index)[0][0]
        # print(indices)
        # print(probs)
        # print(char_index)
        # print('the selected char is:',char)
        # print('the prob of the selected word is ',probs[pro_index])
        # print('the prob of the selected word is ',probs[int_index])
    return char, hidden_size


def sample(model, data_matrix, length, top_k=None, sentence="We are"):
    hidden_size = None
    new_sentence = [char for char in sentence]
    for i in range(length):
        next_char, hidden_size = predict(model, new_sentence[-1], data_matrix[i], top_k=top_k, hidden_size=hidden_size)
        new_sentence.append(next_char)
    return "".join(new_sentence)


def extract_from_text(model, length, text, top_k=None, sentence="We are", LTpath='newbits.txt'):
    hidden_size = None
    new_sentence = [char for char in sentence]
    # new_text = [char for char in text]
    LT_encode_matrix = utils.loadandparity(LTpath)
    LT_decode_matrix = []
    for i in range(length):
        next_char, binary_str, hidden_size = anti_predict(model, new_sentence[-1], text[i], top_k=top_k, hidden_size=hidden_size)
        new_sentence.append(next_char)
        LT_decode_matrix.append(binary_str)
    # print('LTencode:', LT_encode_matrix)
    # print('LTdecode:', LT_decode_matrix)
    return LT_decode_matrix

def anti_predict(model, char, text_str, top_k=None, hidden_size=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        if top_k is None:  
            indices = np.arange(vocab_size)
        else:
            probs, indices = probs.topk(top_k)
            indices = indices.cpu().numpy()
        probs = probs.cpu().numpy()
        
        # print('indices', indices)
        # print('model.char_int', model.char_int)
        char_index = model.char_int.get(text_str)
        int_index = utils.find_char_index(indices, char_index)
        binary_index = utils.decimal_to_binary(int_index)
        char = text_str
        # print(indices)
        # print(probs)
        # print(char_index)
        # print('the selected char is:', char)
        # print('the prob of the selected word is ',probs[char_index])
        
    return char, binary_index, hidden_size 

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
    
    #================信息嵌入阶段================
    # 根据LT编码结果生成文本
    new_text = sample(model, embed_matrix, length=len(embed_matrix), top_k=32, sentence=inital_text) 
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
    # 删除攻击
    a2 = attack_type.str_delete(real_text)
    print('the text after delete_attack:', a2)
    # 换位攻击
    a3 = attack_type.str_swap(real_text)
    print('the text after swap_attack:', a3)
    #============================================
    
    #================信息提取阶段=================
    print('='*100)
    print('篡改攻击')
    length1 = len(a1)
    extract_matrix1 = extract_from_text(model, length1, a1, top_k=32, sentence=inital_text)
    
   
    print('='*100)
    print('删除攻击')
    length2 = len(a2)
    extract_matrix2 = extract_from_text(model, length2, a2, top_k=32, sentence=inital_text)

    print('='*100)
    print('换位攻击')
    length3 = len(a3)
    extract_matrix3 = extract_from_text(model, length3, a3, top_k=32, sentence=inital_text)



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

