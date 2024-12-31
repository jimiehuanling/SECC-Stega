clear,clc;
%%
% 设置参数
Im=imread('message.bmp');
Im=im2double(Im);
message_matrix = Im;

% 设置参数：包数、包长
% packet_num为包数
packet_num=size(message_matrix,1);
% packet_length为包长
packet_length=size(message_matrix,2);
%%
% 篡改攻击
% 读取编码数据及其映射关系
% data为LT编码结果
% fileName = 'modified_bitstream.txt';
fileName = 'flc-changed.txt';
fileID = fopen(fileName, 'r');
bitM = {};
data =[];
lineIndex = 1;
while ~feof(fileID)
    line = fgetl(fileID);
    a = line(1:end-1);
    bitM{lineIndex} = a;
    doubleArray = zeros(1, length(a));
    for i = 1:length(a)
        doubleArray(i) = str2double(a(i));
    end
    data = [data ; doubleArray];
    lineIndex = lineIndex + 1;
end
fclose(fileID);
% data2是LT编码数据对应的索引
data2 = load('index.txt');
pn = size(data,1);
all_p = 3 * packet_num;
data2 = data2';
B1 = reshape(data2, packet_num, all_p);
B11 =B1';
B111 = [];

resti = load('flc-changed-index.txt');
resti = resti' + 1 ;
% 按行遍历矩阵
for i = 1:size(B11, 1)
    % 访问第i行
    row = B11(i, :);
    isInList = ismember(i, resti);
    if isInList
        B111 = [B111;row];
    end
end
%%
% LT解码
turn=all_p;
kuan=packet_num+packet_length;
watermarking = data;
H = B111;
% 设置丢包率模拟真实通信信道
packet_loss=0;
rank_statistic=[];
receive_packet=0;
tag_decode=0;
%%
% 选择LT码解码
% 解码方式，1为BP，2为高斯
decode_tag = 2;
watermarking_decode=[];
H_decode=[];

for i=1:turn      
    if decode_tag == 1             
        %BP译码         
        %接收包加一              
        receive_packet = receive_packet + 1;                 
        %传递的参数有：参与异或的原始数据包的序号，异或后的结果，译码序号矩阵，译码结果，矩阵秩                
        [H_decode,watermarking_decode,tag_decode] = LT_decode_BP(H(i,:),watermarking(i,:),H_decode,watermarking_decode);               
        %计算现在的秩               
        rank_statistic = [rank_statistic find_rank(H_decode)];                      
    elseif decode_tag == 2               
        %高斯译码               
        %接受包加一           
        receive_packet = receive_packet + 1;     
        %进行高斯消元处理               
        %传递的参数有：参与异或的原始数据包的序号，异或后的结果，译码序号矩阵，译码结果，矩阵秩         
        [H_decode,watermarking_decode,tag_decode,rank_statistic] = LT_decode_Guassian(H(i,:),watermarking(i,:),H_decode,watermarking_decode,rank_statistic);            
    end
       
    % tag_decode代表整个译码是否成功，如果成功则跳出循环            
    % 展示           
    if tag_decode == 1                                      
        disp('decode success');
        disp('used packet num is');
        disp(receive_packet);                
        disp('send packet num is');               
        disp(i);                 
        break;            
    end   
end

%%
% 展示提取的秘密信息
for i=1:packet_num
    for j=1:packet_length
        I2(i,j)=watermarking_decode(i,j);        
    end
end
% subimage(I2);
% title('秘密信息（提取）');
ncxishu=NC(Im,I2);
BERating=BER(Im,I2);
%%
% 读取编码数据及其映射关系
% data为LT编码结果
% fileName = 'modified_bitstream.txt';
fileName = 'flc-deleted.txt';
fileID = fopen(fileName, 'r');
bitM = {};
data =[];
lineIndex = 1;
while ~feof(fileID)
    line = fgetl(fileID);
    a = line(1:end-1);
    bitM{lineIndex} = a;
    doubleArray = zeros(1, length(a));
    for i = 1:length(a)
        doubleArray(i) = str2double(a(i));
    end
    data = [data ; doubleArray];
    lineIndex = lineIndex + 1;
end
fclose(fileID);
% data2是LT编码数据对应的索引
data2 = load('index.txt');
pn = size(data,1);
all_p = 3 * packet_num;
data2 = data2';
B1 = reshape(data2, packet_num, all_p);
B11 =B1';
B111 = [];

resti = load('flc-deleted-index.txt');
resti = resti' + 1 ;
% 按行遍历矩阵
for i = 1:size(B11, 1)
    % 访问第i行
    row = B11(i, :);
    isInList = ismember(i, resti);
    if isInList
        B111 = [B111;row];
    end
end
%%
% LT解码
turn=all_p;
kuan=packet_num+packet_length;
watermarking = data;
H = B111;
% 设置丢包率模拟真实通信信道
packet_loss=0;
rank_statistic=[];
receive_packet=0;
tag_decode=0;
%%
% 选择LT码解码
% 解码方式，1为BP，2为高斯
decode_tag = 2;
watermarking_decode=[];
H_decode=[];

for i=1:turn      
    if decode_tag == 1             
        %BP译码         
        %接收包加一              
        receive_packet = receive_packet + 1;                 
        %传递的参数有：参与异或的原始数据包的序号，异或后的结果，译码序号矩阵，译码结果，矩阵秩                
        [H_decode,watermarking_decode,tag_decode] = LT_decode_BP(H(i,:),watermarking(i,:),H_decode,watermarking_decode);               
        %计算现在的秩               
        rank_statistic = [rank_statistic find_rank(H_decode)];                      
    elseif decode_tag == 2               
        %高斯译码               
        %接受包加一           
        receive_packet = receive_packet + 1;     
        %进行高斯消元处理               
        %传递的参数有：参与异或的原始数据包的序号，异或后的结果，译码序号矩阵，译码结果，矩阵秩         
        [H_decode,watermarking_decode,tag_decode,rank_statistic] = LT_decode_Guassian(H(i,:),watermarking(i,:),H_decode,watermarking_decode,rank_statistic);            
    end
       
    % tag_decode代表整个译码是否成功，如果成功则跳出循环            
    % 展示           
    if tag_decode == 1                                      
        disp('decode success');
        disp('used packet num is');
        disp(receive_packet);                
        disp('send packet num is');               
        disp(i);                 
        break;            
    end   
end

%%
% 展示提取的秘密信息
for i=1:packet_num
    for j=1:packet_length
        I2(i,j)=watermarking_decode(i,j);        
    end
end
% subimage(I2);
% title('秘密信息（提取）');
ncxishu=NC(Im,I2);
BERating1=BER(Im,I2);
%%
% 换位攻击
% 读取编码数据及其映射关系
% data为LT编码结果
% fileName = 'modified_bitstream.txt';
fileName = 'flc-swapped.txt';
fileID = fopen(fileName, 'r');
bitM = {};
data =[];
lineIndex = 1;
while ~feof(fileID)
    line = fgetl(fileID);
    a = line(1:end-1);
    bitM{lineIndex} = a;
    doubleArray = zeros(1, length(a));
    for i = 1:length(a)
        doubleArray(i) = str2double(a(i));
    end
    data = [data ; doubleArray];
    lineIndex = lineIndex + 1;
end
fclose(fileID);
% data2是LT编码数据对应的索引
data2 = load('index.txt');
pn = size(data,1);
all_p = 3 * packet_num;
data2 = data2';
B1 = reshape(data2, packet_num, all_p);
B11 =B1';
B111 = [];

resti = load('flc-swapped-index.txt');
resti = resti' + 1 ;
% 按行遍历矩阵
for i = 1:size(B11, 1)
    % 访问第i行
    row = B11(i, :);
    isInList = ismember(i, resti);
    if isInList
        B111 = [B111;row];
    end
end
%%
% LT解码
turn=all_p;
kuan=packet_num+packet_length;
watermarking = data;
H = B111;
% 设置丢包率模拟真实通信信道
packet_loss=0;
rank_statistic=[];
receive_packet=0;
tag_decode=0;
%%
% 选择LT码解码
% 解码方式，1为BP，2为高斯
decode_tag = 2;
watermarking_decode=[];
H_decode=[];

for i=1:turn      
    if decode_tag == 1             
        %BP译码         
        %接收包加一              
        receive_packet = receive_packet + 1;                 
        %传递的参数有：参与异或的原始数据包的序号，异或后的结果，译码序号矩阵，译码结果，矩阵秩                
        [H_decode,watermarking_decode,tag_decode] = LT_decode_BP(H(i,:),watermarking(i,:),H_decode,watermarking_decode);               
        %计算现在的秩               
        rank_statistic = [rank_statistic find_rank(H_decode)];                      
    elseif decode_tag == 2               
        %高斯译码               
        %接受包加一           
        receive_packet = receive_packet + 1;     
        %进行高斯消元处理               
        %传递的参数有：参与异或的原始数据包的序号，异或后的结果，译码序号矩阵，译码结果，矩阵秩         
        [H_decode,watermarking_decode,tag_decode,rank_statistic] = LT_decode_Guassian(H(i,:),watermarking(i,:),H_decode,watermarking_decode,rank_statistic);            
    end
       
    % tag_decode代表整个译码是否成功，如果成功则跳出循环            
    % 展示           
    if tag_decode == 1                                      
        disp('decode success');
        disp('used packet num is');
        disp(receive_packet);                
        disp('send packet num is');               
        disp(i);                 
        break;            
    end   
end

%%
% 展示提取的秘密信息
for i=1:packet_num
    for j=1:packet_length
        I2(i,j)=watermarking_decode(i,j);        
    end
end
% subimage(I2);
% title('秘密信息（提取）');
ncxishu=NC(Im,I2);
BERating2=BER(Im,I2);