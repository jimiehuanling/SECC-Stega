import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
import utils
import random
import math
import heapq
import attack_type
import train_model_adg as ls
# 定义哈夫曼树节点
class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    # 定义比较运算符以便在优先队列中排序
    def __lt__(self, other):
        return self.freq < other.freq
    
# 构建哈夫曼树
def build_huffman_tree(char_freq):
    # 创建优先队列
    priority_queue = [Node(char, freq) for char, freq in char_freq.items()]
    heapq.heapify(priority_queue)
    
    # 构建哈夫曼树
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        
        heapq.heappush(priority_queue, merged)
    
    return priority_queue[0]

# 哈夫曼编码
def generate_huffman_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    
    if node.char is not None:
        codebook[node.char] = prefix
    else:
        generate_huffman_codes(node.left, prefix + "0", codebook)
        generate_huffman_codes(node.right, prefix + "1", codebook)
    
    return codebook

# 哈夫曼解码全部字符
def huffman_decode(bit_stream, huffman_tree):
    decoded_string = ""
    current = huffman_tree
    for bit in bit_stream:
        if bit == '0':
            current = current.left
        else:  # bit == '1'
            current = current.right
        
        if current.char is not None:
            decoded_string += current.char
            current = huffman_tree
    return decoded_string

# 哈夫曼解码一个字符
def huffman_decode_one_char(bit_stream, huffman_tree):
    current = huffman_tree
    for i, bit in enumerate(bit_stream):
        if bit == '0':
            current = current.left
        else:  # bit == '1'
            current = current.right
        
        if current.char is not None:
            # 返回字符及其在比特流中的长度
            return current.char, i + 1  
    # 如果未找到字符，返回None和0
    return None, 0  

# 哈夫曼解码一个字符对应的比特
def huffman_decode_one_bits(target_char, huffman_codes):
    chars = list(huffman_codes.keys())
    bits = list(huffman_codes.values())

    index = None
    if target_char in chars:
        index =chars.index(target_char)
    if index != None:
        return bits[index]
    else:
        return ''
def sample(model, data_matrix, sentence="We are"):
    hidden_size = None
    new_sentence = [char for char in sentence]
    # bitstream为整个秘密信息的字符串形式
    
    bitstream = ''
    for i in data_matrix:
        bitstream += i
    print('bitstream:',bitstream)

    index_char=''
    while(index_char!=None):
        
        charAndprob, hidden_size = predict(model, new_sentence[-1], hidden_size=hidden_size)
        chars_list = list(charAndprob.keys())
        probs_list = list(charAndprob.values())
        
        pmax = probs_list[0]
        # 计算标准值
        a = math.log2(pmax)
        v = math.floor(-a)
        u = 2**v
        pmean = 1/u
        # print('u:', u)
        # print('pmean:', pmean)

        # G为分组的总集合
        G = []
        for i in range(u-1):
            # g为每次的分组
            g = []
            # 将目前降序排列的概率中的最大值添加到g中
            g.append(chars_list[0])
            # 移除该元素
            chars_list.pop(0)
            probs_list.pop(0)
            p = charAndprob[g[0]]

            while p < pmean:
                # 从目前的概率列表中选择与e最接近的概率值并添加到g中
                e = pmean - p
                
                # 计算概率的绝对值
                dif_abs =[]
                for prob in probs_list:
                    dif = abs(prob - e)
                    dif_abs.append(dif)
                # print('dif_abs:',dif_abs)
                
                abs_min = min(dif_abs)
                if abs_min < 2*e:
                    abs_min_index = dif_abs.index(abs_min)
                    # print('abs_min_index:',abs_min_index)

                    g.append(chars_list[abs_min_index])
                    p += probs_list[abs_min_index]
                    chars_list = chars_list[:abs_min_index]+chars_list[abs_min_index+1:]
                    probs_list = probs_list[:abs_min_index]+probs_list[abs_min_index+1:]

                else:
                    break
                   
            prob_rest = 0
            for prob in probs_list:
                prob_rest += prob
            pmean = prob_rest/(u - i + 1)
            G.append(g)
        g = []   
        for i in chars_list:
            g.append(i)
        G.append(g)
        
        if len(G) == 1:
            g1 = []
            g2 = []
            g1 = G[0][0]
            g2 = G[0][1:]
            G = []
            G.append(g1)
            G.append(g2)

        # print('G:', G)
        # print('length of G:', len(G))

        # 当分组数大于等于2时才能用来嵌入秘密信息
        if len(G) >= 2:
            GAndlen = {}
            for i, group in enumerate(G) :
                GAndlen[f'{i}'] = 66 - len(group)
                GAndlen[f'{i}'] = len(group)
            
            #print(GAndlen)
            # 构建哈夫曼树
            huffman_tree = build_huffman_tree(GAndlen)

            index_char, length = huffman_decode_one_char(bitstream, huffman_tree)
            # print('index_char:',index_char)
            bitstream = bitstream[length:]
            if index_char!=None:
                index_int = int(index_char)
                decoded_char = random.choice(G[index_int])
                new_sentence.append(decoded_char)   
                # print('char:',decoded_char)

    return "".join(new_sentence)

def predict(model, char, hidden_size=None):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()  
    vocab_size = 65
    top_k = 65
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
        

        # probss为概率列表，charss为对应的字符列表
        probss =[]
        charss = []
        for i in probs:
            probss.append(i)
        
        for i in indices:
            j = model.int_char[i]
            charss.append(j)
        
        # 将字符和概率组装成字典
        combined = zip(charss, probss)
        charAndprob = dict(combined)
        
    return charAndprob, hidden_size
def extract_from_text(model, text, sentence="We are"):
    hidden_size = None
    new_text = list(text)
    new_sentence = list(sentence)
    bitstream = ''
    
    for token in new_text:
    # for ii in range(5):
        # token = text[ii]
        # print('='*100)
        charAndprob, hidden_size = predict(model, new_sentence[-1], hidden_size=hidden_size)
        chars_list = list(charAndprob.keys())
        probs_list = list(charAndprob.values())
        
        pmax = probs_list[0]
        # 计算标准值
        a = math.log2(pmax)
        v = math.floor(-a)
        u = 2**v
        pmean = 1/u
        # print('u:', u)
        # print('pmean:', pmean)

        # G为分组的总集合
        G = []
        for i in range(u-1):
            # g为每次的分组
            g = []
            # 将目前降序排列的概率中的最大值添加到g中
            g.append(chars_list[0])
            # 移除该元素
            chars_list.pop(0)
            probs_list.pop(0)
            p = charAndprob[g[0]]

            while p < pmean:
                # 从目前的概率列表中选择与e最接近的概率值并添加到g中
                e = pmean - p
                
                # 计算概率的绝对值
                dif_abs =[]
                for prob in probs_list:
                    dif = abs(prob - e)
                    dif_abs.append(dif)
                # print('dif_abs:',dif_abs)
                
                abs_min = min(dif_abs)
                if abs_min < 2*e:
                    abs_min_index = dif_abs.index(abs_min)
                    # print('abs_min_index:',abs_min_index)

                    g.append(chars_list[abs_min_index])
                    p += probs_list[abs_min_index]
                    chars_list = chars_list[:abs_min_index]+chars_list[abs_min_index+1:]
                    probs_list = probs_list[:abs_min_index]+probs_list[abs_min_index+1:]

                else:
                    break
                   
            prob_rest = 0
            for prob in probs_list:
                prob_rest += prob
            pmean = prob_rest/(u - i + 1)
            G.append(g)
        g = []   
        for i in chars_list:
            g.append(i)
        G.append(g)
        
        if len(G) == 1:
            g1 = []
            g2 = []
            g1 = G[0][0]
            g2 = G[0][1:]
            G = []
            G.append(g1)
            G.append(g2)

        # print('G:', G)
        # print('length of G:', len(G))

        # 当分组数大于等于2时才能用来嵌入秘密信息
        if len(G) >= 2:
            GAndlen = {}
            for i, group in enumerate(G) :
                GAndlen[f'{i}'] = len(group)
            # print(GAndlen)

            # 构建哈夫曼树
            huffman_tree = build_huffman_tree(GAndlen)
            # 哈夫曼编码
            huffman_codes = generate_huffman_codes(huffman_tree)
            # for char,code in huffman_codes.items():
            #     print(f"{char}: {code}")
        
            for index, group in enumerate(G):
                # print('token:',token)
                # print('group:',group)

                if token in group:
                    group_index = str(index)
                    # print('group_index:',group_index)
                    bits = huffman_decode_one_bits(group_index, huffman_codes)
                    bitstream += bits

                    # print('bits:',bits)
                    new_sentence.append(token)
        
    pad_char = '0'
    default_length = 80
    if len(bitstream) < default_length:
        bitstream = bitstream.ljust(default_length, pad_char)
    else:
        bitstream = bitstream[:default_length]
    return bitstream

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
    new_text = sample(model, data_matrix, sentence=inital_text) 
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
    print('the text after swap_attack:',a3)
    #============================================
    
    #================信息提取阶段=================
    print('='*100)
    print('篡改攻击')
    # 篡改攻击
    # 提取秘密信息矩阵
    extract_data1 = extract_from_text(model, a1, sentence=inital_text)
    
   
    print('='*100)
    print('删除攻击')
    # 删除攻击
    # 提取秘密信息矩阵
    extract_data2 = extract_from_text(model, a2, sentence=inital_text)

    print('='*100)
    print('换位攻击')
    # 删除攻击
    # 提取秘密信息矩阵
    extract_data3 = extract_from_text(model, a3, sentence=inital_text)

    embed_data = ''
    for i in data_matrix:
        embed_data += i
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

