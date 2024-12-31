import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
import math
import random
import heapq
import utils

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
    
class lstm_model(nn.Module):
    def __init__(self, vocab, hidden_size, num_layers, dropout=0.5):
        super(lstm_model, self).__init__()
        self.vocab = vocab  # 字符数据集
        # 索引，字符
        self.int_char = {i: char for i, char in enumerate(vocab)}
        self.char_int = {char: i for i, char in self.int_char.items()}
        # 对字符进行one-hot encoding
        self.encoder = OneHotEncoder().fit(vocab.reshape(-1, 1))

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # lstm层
        self.lstm = nn.LSTM(len(vocab), hidden_size, num_layers, batch_first=True, dropout=dropout)

        # 全连接层
        self.linear = nn.Linear(hidden_size, len(vocab))

    def forward(self, sequence, hs=None):
        out, hs = self.lstm(sequence, hs)  
        out = out.reshape(-1, self.hidden_size)  
        output = self.linear(out)  
        return output, hs

    def onehot_encode(self, data):
        return self.encoder.transform(data)

    def onehot_decode(self, data):
        return self.encoder.inverse_transform(data)

    def label_encode(self, data):
        return np.array([self.char_int[ch] for ch in data])

    def label_decode(self, data):
        return np.array([self.int_char[ch] for ch in data])


def get_batches(data, batch_size, seq_len):
    '''
    :param data: 源数据，输入格式(num_samples, num_features)
    :param batch_size: batch的大小
    :param seq_len: 序列的长度（精度）
    :return: （batch_size, seq_len, num_features）
    '''
    num_features = data.shape[1]
    # print(f'hang{data.shape[0]}')
    # print(f'lie{data.shape[1]}')
    num_chars = batch_size * seq_len  
    num_batches = int(np.floor(data.shape[0] / num_chars))  

    need_chars = num_batches * num_chars  

    targets = np.append(data[1:], data[0]).reshape(data.shape)
    inputs = data[:need_chars]
    targets = targets[:need_chars]
    targets = targets.reshape(batch_size, -1, num_features)
    inputs = inputs.reshape(batch_size, -1, num_features)
    

    for i in range(0, inputs.shape[1], seq_len):
        x = inputs[:, i: i+seq_len]
        y = targets[:, i: i+seq_len]
        yield x, y  


def train(model, data, batch_size, seq_len, epochs, lr, valid):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'the used device is {device}')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if valid is not None:
        data = model.onehot_encode(data.reshape(-1, 1))
        data = data.toarray()
        valid = model.onehot_encode(valid.reshape(-1, 1))
        valid = valid.toarray()
    else:
        data = model.onehot_encode(data.reshape(-1, 1))
        data = data.toarray()

    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        model.train()
        hs = None  
        train_ls = 0.0
        val_ls = 0.0
        for x, y in get_batches(data, batch_size, seq_len):
            optimizer.zero_grad()
            x = torch.tensor(x).float().to(device)
            out, hs = model(x, hs)
            hs = ([h.data for h in hs])
            y = y.reshape(-1, len(model.vocab))
            y = model.onehot_decode(y)
            y = model.label_encode(y.squeeze())
            y = torch.from_numpy(y).long().to(device)
            loss = criterion(out, y.squeeze())
            loss.backward()
            optimizer.step()
            train_ls += loss.item()

        if valid is not None:
            model.eval()
            hs = None
            with torch.no_grad():
                for x, y in get_batches(valid, batch_size, seq_len):
                    x = torch.tensor(x).float().to(device)  
                    out, hs = model(x, hs)

                    hs = ([h.data for h in hs])  

                    y = y.reshape(-1, len(model.vocab))  
                    y = model.onehot_decode(y)  
                    y = model.label_encode(y.squeeze())  
                    y = torch.from_numpy(y).long().to(device)

                    loss = criterion(out, y.squeeze())  
                    val_ls += loss.item()
                val_loss.append(np.mean(val_ls))
            train_loss.append(np.mean(train_ls))
        print(f'------------------Epochs{epoch} | {epoch}--------------------')
        print(f"Train_loss:{train_loss[-1]}")
        if val_loss:
            print(f'Val_Loss: {val_loss[-1]}')
        

    plt.plot(train_loss, label="Train_loss")
    plt.plot(val_loss, label="Val loss")
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.show()

    model_name = "lstm_model.net"
    # 保存训练后的模型
    with open(model_name, 'wb') as f:  
        torch.save(model.state_dict(), f)


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


def sample(model, sentence="We are", LTpath='newbits.txt'):
    hidden_size = None
    new_sentence = [char for char in sentence]
    LT_matrix = utils.loadandparity(LTpath)
    # bitstream为整个秘密信息的字符串形式
    bitstream = ''
    for i in LT_matrix:
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


def get_vocabulary(file_path):
    with open(file_path) as data:
        text = data.read()
        vocab = np.array(sorted(set(text)))
        return text, vocab

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
    default_length = 300
    if len(bitstream) < default_length:
        bitstream = bitstream.ljust(default_length, pad_char)
        bitstream = [bitstream[i:i + 5] for i in range(0, len(bitstream), 5)]
    else:
        bitstream = bitstream[:300]
        bitstream = [bitstream[i:i + 5] for i in range(0, len(bitstream), 5)]
    return bitstream



def main():
    # 超参数
    hidden_size = 512
    num_layers = 2
    batch_size = 128
    seq_len = 100
    epochs = 100
    lr = 0.01
    file_path = 'shakespeare.txt'
    LTcodes_path = 'newbits.txt'
    inital_text = 'We are'
    
    
    # 读取数据集
    text, vocab = get_vocabulary(file_path)
    vocab_size = len(vocab)
    # print(vocab_size)

    # 划分训练测试集
    val_len = int(np.floor(0.2 * len(text)))  
    trainset = np.array(list(text[:-val_len]))
    validset = np.array(list(text[-val_len:]))
    # print(trainset)

    #================模型处理阶段===============
    # 建立模型
    model = lstm_model(vocab, hidden_size, num_layers)  
    # 训练模型
    train(model, trainset, batch_size, seq_len, epochs, lr, validset)  
    #===========================================
    

if __name__ == "__main__":
    main()

