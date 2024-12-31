clear,clc;
t1 = clock;
%%
% �������40���صĶ�����������Ϊ������Ϣ������ΪBMP��ʽ��ͼƬ
message_matrix = randi([0, 1], 20, 4);
imwrite(message_matrix, 'message.bmp');

% ����LT���������������Ϣ����������
% packet_numΪ����
packet_num = size(message_matrix,1);
% packet_lengthΪ����
packet_length = size(message_matrix,2);
% һ�㽫�������ݰ�����������Ϊ������������
packet_number = 3*packet_num;
% packet_int�Ǳ������ݼ�ӳ��
% A�Ǳ������ݣ�B��ӳ��
packet_int = [];  
A = [];
B = [];
%% 
% LT�������
% ³���²��ֲ�������һ�����ʷֲ��������������Ƶģ��Ͱ��������
distribution_matrix_prob = robust_solition(packet_num);  
distribution_sum = sum(distribution_matrix_prob);
% ���ɱ������ݰ�
for runtag = 1:packet_number
    % �õ����η��Ͷ���������³���²��ֲ���ѡ��һ���ȷֲ���ÿ����һ�����������Ҫѡ��һ����
    send_degree = randsrc(1,1,[1:size(distribution_matrix_prob,2);distribution_matrix_prob]);           
    % �����ʼ��
    % H��ʾ������һ�������ʱ����ʹ�õ����Ϊ1��δʹ�õ�Ϊ0
    % code_encodeΪһ�������������
    code_encode = zeros(1,packet_length);          
    H = zeros(1,packet_num);        
    % ѡ���ļ���ԭʼ���ݰ�Ϊ������������
    message_encode_pos = [];

    % ��ʼѡ��ԭʼ���ݰ�         
    % �����ѡ������ԭʼ���ݰ�          
    % send_degreeΪ���εĶ���        
    i = 1;      
    % ��i<=����send_degreeʱ����һֱ����ѭ����ֱ������������
            
    while i <= send_degree       
        %ѭ����������֤ѭ�����Լ��� 
        i = i + 1;   
        %��1��packet_num�������ѡ��һ����   
        temp_pos = randi(packet_num);
        if H(temp_pos) == 0              
            H(temp_pos) = 1;         
            %���õ��İ������ȫ����¼����  
            message_encode_pos = [message_encode_pos temp_pos]; 
        %��������֮ǰ��ѡ�����������ѡ��һ��
        else
            i = i - 1;    
        end          
    end
    
    % ����ѡ����ԭʼ���ݰ����������          
    % code_encode���������һ���������ݰ��Ľ����H����ӳ��           
    for i = 1:size(message_encode_pos,2)                    
        code_encode =  rem(code_encode+message_matrix(message_encode_pos(i),:),2);
    end
    
    inte=[];
    inte=[code_encode H];
    A=[A code_encode];
    B=[B H];
    packet_int=[packet_int;inte];  
end
%%
% ��������������Ϣ�Լ������ֱ����txt�ļ�
fileID1 = fopen('newbits.txt', 'w');
for i = 1:length(A)
    fprintf(fileID1, '%d\n', A(i));
end
fclose(fileID1);
fileID2 = fopen('index.txt', 'w');
for i = 1:length(B)
    fprintf(fileID2, '%d\n', B(i));
end
fclose(fileID2);
%%
t2 = clock;
t = etime(t2,t1);