clear,clc;
%%
% ���ò���
Im=imread('message.bmp');
Im=im2double(Im);
message_matrix = Im;

% ���ò���������������
% packet_numΪ����
packet_num=size(message_matrix,1);
% packet_lengthΪ����
packet_length=size(message_matrix,2);
%%
% �۸Ĺ���
% ��ȡ�������ݼ���ӳ���ϵ
% dataΪLT������
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
% data2��LT�������ݶ�Ӧ������
data2 = load('index.txt');
pn = size(data,1);
all_p = 3 * packet_num;
data2 = data2';
B1 = reshape(data2, packet_num, all_p);
B11 =B1';
B111 = [];

resti = load('flc-changed-index.txt');
resti = resti' + 1 ;
% ���б�������
for i = 1:size(B11, 1)
    % ���ʵ�i��
    row = B11(i, :);
    isInList = ismember(i, resti);
    if isInList
        B111 = [B111;row];
    end
end
%%
% LT����
turn=all_p;
kuan=packet_num+packet_length;
watermarking = data;
H = B111;
% ���ö�����ģ����ʵͨ���ŵ�
packet_loss=0;
rank_statistic=[];
receive_packet=0;
tag_decode=0;
%%
% ѡ��LT�����
% ���뷽ʽ��1ΪBP��2Ϊ��˹
decode_tag = 2;
watermarking_decode=[];
H_decode=[];

for i=1:turn      
    if decode_tag == 1             
        %BP����         
        %���հ���һ              
        receive_packet = receive_packet + 1;                 
        %���ݵĲ����У���������ԭʼ���ݰ�����ţ�����Ľ����������ž�����������������                
        [H_decode,watermarking_decode,tag_decode] = LT_decode_BP(H(i,:),watermarking(i,:),H_decode,watermarking_decode);               
        %�������ڵ���               
        rank_statistic = [rank_statistic find_rank(H_decode)];                      
    elseif decode_tag == 2               
        %��˹����               
        %���ܰ���һ           
        receive_packet = receive_packet + 1;     
        %���и�˹��Ԫ����               
        %���ݵĲ����У���������ԭʼ���ݰ�����ţ�����Ľ����������ž�����������������         
        [H_decode,watermarking_decode,tag_decode,rank_statistic] = LT_decode_Guassian(H(i,:),watermarking(i,:),H_decode,watermarking_decode,rank_statistic);            
    end
       
    % tag_decode�������������Ƿ�ɹ�������ɹ�������ѭ��            
    % չʾ           
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
% չʾ��ȡ��������Ϣ
for i=1:packet_num
    for j=1:packet_length
        I2(i,j)=watermarking_decode(i,j);        
    end
end
% subimage(I2);
% title('������Ϣ����ȡ��');
ncxishu=NC(Im,I2);
BERating=BER(Im,I2);
%%
% ��ȡ�������ݼ���ӳ���ϵ
% dataΪLT������
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
% data2��LT�������ݶ�Ӧ������
data2 = load('index.txt');
pn = size(data,1);
all_p = 3 * packet_num;
data2 = data2';
B1 = reshape(data2, packet_num, all_p);
B11 =B1';
B111 = [];

resti = load('flc-deleted-index.txt');
resti = resti' + 1 ;
% ���б�������
for i = 1:size(B11, 1)
    % ���ʵ�i��
    row = B11(i, :);
    isInList = ismember(i, resti);
    if isInList
        B111 = [B111;row];
    end
end
%%
% LT����
turn=all_p;
kuan=packet_num+packet_length;
watermarking = data;
H = B111;
% ���ö�����ģ����ʵͨ���ŵ�
packet_loss=0;
rank_statistic=[];
receive_packet=0;
tag_decode=0;
%%
% ѡ��LT�����
% ���뷽ʽ��1ΪBP��2Ϊ��˹
decode_tag = 2;
watermarking_decode=[];
H_decode=[];

for i=1:turn      
    if decode_tag == 1             
        %BP����         
        %���հ���һ              
        receive_packet = receive_packet + 1;                 
        %���ݵĲ����У���������ԭʼ���ݰ�����ţ�����Ľ����������ž�����������������                
        [H_decode,watermarking_decode,tag_decode] = LT_decode_BP(H(i,:),watermarking(i,:),H_decode,watermarking_decode);               
        %�������ڵ���               
        rank_statistic = [rank_statistic find_rank(H_decode)];                      
    elseif decode_tag == 2               
        %��˹����               
        %���ܰ���һ           
        receive_packet = receive_packet + 1;     
        %���и�˹��Ԫ����               
        %���ݵĲ����У���������ԭʼ���ݰ�����ţ�����Ľ����������ž�����������������         
        [H_decode,watermarking_decode,tag_decode,rank_statistic] = LT_decode_Guassian(H(i,:),watermarking(i,:),H_decode,watermarking_decode,rank_statistic);            
    end
       
    % tag_decode�������������Ƿ�ɹ�������ɹ�������ѭ��            
    % չʾ           
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
% չʾ��ȡ��������Ϣ
for i=1:packet_num
    for j=1:packet_length
        I2(i,j)=watermarking_decode(i,j);        
    end
end
% subimage(I2);
% title('������Ϣ����ȡ��');
ncxishu=NC(Im,I2);
BERating1=BER(Im,I2);
%%
% ��λ����
% ��ȡ�������ݼ���ӳ���ϵ
% dataΪLT������
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
% data2��LT�������ݶ�Ӧ������
data2 = load('index.txt');
pn = size(data,1);
all_p = 3 * packet_num;
data2 = data2';
B1 = reshape(data2, packet_num, all_p);
B11 =B1';
B111 = [];

resti = load('flc-swapped-index.txt');
resti = resti' + 1 ;
% ���б�������
for i = 1:size(B11, 1)
    % ���ʵ�i��
    row = B11(i, :);
    isInList = ismember(i, resti);
    if isInList
        B111 = [B111;row];
    end
end
%%
% LT����
turn=all_p;
kuan=packet_num+packet_length;
watermarking = data;
H = B111;
% ���ö�����ģ����ʵͨ���ŵ�
packet_loss=0;
rank_statistic=[];
receive_packet=0;
tag_decode=0;
%%
% ѡ��LT�����
% ���뷽ʽ��1ΪBP��2Ϊ��˹
decode_tag = 2;
watermarking_decode=[];
H_decode=[];

for i=1:turn      
    if decode_tag == 1             
        %BP����         
        %���հ���һ              
        receive_packet = receive_packet + 1;                 
        %���ݵĲ����У���������ԭʼ���ݰ�����ţ�����Ľ����������ž�����������������                
        [H_decode,watermarking_decode,tag_decode] = LT_decode_BP(H(i,:),watermarking(i,:),H_decode,watermarking_decode);               
        %�������ڵ���               
        rank_statistic = [rank_statistic find_rank(H_decode)];                      
    elseif decode_tag == 2               
        %��˹����               
        %���ܰ���һ           
        receive_packet = receive_packet + 1;     
        %���и�˹��Ԫ����               
        %���ݵĲ����У���������ԭʼ���ݰ�����ţ�����Ľ����������ž�����������������         
        [H_decode,watermarking_decode,tag_decode,rank_statistic] = LT_decode_Guassian(H(i,:),watermarking(i,:),H_decode,watermarking_decode,rank_statistic);            
    end
       
    % tag_decode�������������Ƿ�ɹ�������ɹ�������ѭ��            
    % չʾ           
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
% չʾ��ȡ��������Ϣ
for i=1:packet_num
    for j=1:packet_length
        I2(i,j)=watermarking_decode(i,j);        
    end
end
% subimage(I2);
% title('������Ϣ����ȡ��');
ncxishu=NC(Im,I2);
BERating2=BER(Im,I2);