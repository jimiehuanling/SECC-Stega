import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
import utils

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


def sample(model, length, top_k=None, sentence="We are", LTpath='newbits.txt'):
    hidden_size = None
    new_sentence = [char for char in sentence]
    LT_matrix = utils.loadandparity(LTpath)
    for i in range(length):
        next_char, hidden_size = predict(model, new_sentence[-1], LT_matrix[i], top_k=top_k, hidden_size=hidden_size)
        new_sentence.append(next_char)
    return "".join(new_sentence)


def get_vocabulary(file_path):
    with open(file_path) as data:
        text = data.read()
        vocab = np.array(sorted(set(text)))
        return text, vocab

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

