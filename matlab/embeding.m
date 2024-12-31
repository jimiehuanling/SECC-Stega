clear,clc;
t1 = clock;
%%
% 随机生成40比特的二进制数据作为秘密信息并保存为BMP格式的图片
message_matrix = randi([0, 1], 20, 4);
imwrite(message_matrix, 'message.bmp');

% 设置LT编码参数：秘密信息包数、包长
% packet_num为包数
packet_num = size(message_matrix,1);
% packet_length为包长
packet_length = size(message_matrix,2);
% 一般将编码数据包的数量设置为包数的整数倍
packet_number = 3*packet_num;
% packet_int是编码数据及映射
% A是编码数据，B是映射
packet_int = [];  
A = [];
B = [];
%% 
% LT编码过程
% 鲁棒孤波分布，返回一个概率分布，度数是有限制的，和包数不相等
distribution_matrix_prob = robust_solition(packet_num);  
distribution_sum = sum(distribution_matrix_prob);
% 生成编码数据包
for runtag = 1:packet_number
    % 得到本次发送度数，按照鲁棒孤波分布，选择一个度分布，每生成一个编码包，就要选择一个度
    send_degree = randsrc(1,1,[1:size(distribution_matrix_prob,2);distribution_matrix_prob]);           
    % 编码初始化
    % H表示在生成一个编码包时，所使用的序号为1，未使用的为0
    % code_encode为一个编码包的数据
    code_encode = zeros(1,packet_length);          
    H = zeros(1,packet_num);        
    % 选择哪几个原始数据包为编码包的异或结果
    message_encode_pos = [];

    % 开始选择原始数据包         
    % 随机挑选度数个原始数据包          
    % send_degree为本次的度数        
    i = 1;      
    % 当i<=度数send_degree时，则一直保持循环，直到经过度数次
            
    while i <= send_degree       
        %循环条件，保证循环可以继续 
        i = i + 1;   
        %从1到packet_num包中随机选择一个包   
        temp_pos = randi(packet_num);
        if H(temp_pos) == 0              
            H(temp_pos) = 1;         
            %把用到的包的序号全部记录下来  
            message_encode_pos = [message_encode_pos temp_pos]; 
        %如果这个包之前被选择过，则重新选择一次
        else
            i = i - 1;    
        end          
    end
    
    % 对挑选出的原始数据包进行异或处理          
    % code_encode代表异或后的一个编码数据包的结果，H是其映射           
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
% 将编码后的秘密信息以及索引分别存入txt文件
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