clear;
clc;
%ѭ������
loop_num = 1;
%ͳ������
receive_packet_statistic = [];

%��·�������� 
%speed = 100000;
send_bit_matrix_orig_mat = zeros(1,50);
send_bit_matrix_orig_mat1 = zeros(1,50);

%�����㷨--2ΪGE��1ΪBP
decode_tag = 2;
 

%������
ser_matrix = 1e-4;


packet_loss_num = 0;
dynamic_statistic = cell(1,loop_num);



code_send = cell(1,2);
send_bit_total = 0;

%���Ͳ���
file_length = 100000;  %�����ܳ���
K_base = 100; %ԭʼ���ݵķְ���
%ԭʼ���ݵ��볤Ӧ�õ����±�����볤����Ϊ�±�������ԭʼ���ݰ����õ���

w = 3000;
%�������
for loop = 1:loop_num
    for p = ser_matrix 
        %������Ϣ���� 
        K = K_base; %���η����볤��LT���볤ȷ�������Զ���һЩ���ԣ�����ֻ�Ǽ�д��
        
        %����packet_num�Ͱ���packet_length
        packet_num = K;
        packet_length = file_length/K;

        packet_loss = compute_packet_loss( p,packet_length); %���ݰ�����������ȷ�������ʣ�Ҳ�����趨�̶�������
        send_packet =  LT_link_simulate(packet_num,packet_length,decode_tag,receive_packet_statistic,packet_loss); 
        send_redudancy = send_packet*packet_length/file_length;

    end
end

