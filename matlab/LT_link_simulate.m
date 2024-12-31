function [ send_packet ] = LT_link_simulate(packet_num,packet_length,decode_tag,receive_packet_statistic,packet_loss)
        
        %��Ϣ����,ʹ��α�����������һ��α�����������Ϊ��Ϣ���󣬼�ԭʼ����
        %��packet_num��ԭʼ���ݰ���ÿ��ԭʼ���ݰ��ĳ���Ϊpacket_length
        message_matrix = randi([0 1],packet_num,packet_length);

        %packet_num������packet_length����
        %��ʼ�����շ�
        rank_statistic = [0]; %������ͳ�������ȷ�ӳ����ǰ����ɹ������ݰ���
        H_decode = zeros(packet_num,packet_num);
        code_decode = zeros(packet_num,packet_length);

        receive_packet = 0;  %��ʾ���շ�����ɹ�ʱʵ�ʽ��յ������ݰ�
        send_packet = 0; %���ͷ����͵���������

        %³���²��ֲ�
        distribution_matrix_prob = robust_solition(packet_num);  

        success_tag = 0;
        while(~success_tag)
            %�õ����η��Ͷ���
            %size(a,1)��������������size(a,2)������������
            send_degree = randsrc(1,1,[1:size(distribution_matrix_prob,2);distribution_matrix_prob]);           
            
            %�����ʼ��
            H = zeros(1,packet_num);
            code_encode = zeros(1,packet_length);
            message_encode_pos = [];

            %�����ѡ������ԭʼ���ݰ�    
            i = 1;
            while i <= send_degree
                i = i + 1;
                
                temp_pos = randi(size(message_matrix,1));
                if H(temp_pos) == 0
                    H(temp_pos) = 1;
                    message_encode_pos = [message_encode_pos temp_pos];
                else
                    i = i-1;
                end
            end    

            %����ѡ����ԭʼ���ݰ�����������    
            for i = 1:size(message_encode_pos,2)            
                code_encode =  rem(code_encode+message_matrix(message_encode_pos(i),:),2);
            end  

            %����
            code_send{1,1} = code_encode; 
            code_send{1,2} = H;
            send_packet = send_packet + 1;


            %����
            

            %������·����
            packet_loss_tag = randsrc(1,1,[[0 1];[1 - packet_loss packet_loss]]);            
            if packet_loss_tag == 1
                rank_statistic = [rank_statistic rank_statistic(size(rank_statistic,2))]; 
                continue;
            end
            
            %decode_tagΪ1��BP��Ϊ2�Ǹ�˹
            if decode_tag == 1
                %������յ��İ���
                receive_packet = receive_packet + 1; 
                
                %BP����
                %code_send{1,2}Ϊ������ţ�{1,1}Ϊ�����
                [H_decode,code_decode,tag_decode] = LT_decode_BP(code_send{1,2},code_send{1,1},H_decode,code_decode);
            
                
                rank_statistic = [rank_statistic find_rank(H_decode)];
            elseif decode_tag == 2
                %��˹����
                %������յ��İ���
                receive_packet = receive_packet + 1; 
                
                [H_decode,code_decode,tag_decode,rank_statistic] = LT_decode_Guassian(code_send{1,2},code_send{1,1},H_decode,code_decode,rank_statistic); 
            end

            if tag_decode == 1                       
                disp('decode success');
                disp('receive packet num is');
                disp(receive_packet);
                disp('send packet num is');
                disp(send_packet);
                if decode_tag == 1
                    %BP����
                elseif decode_tag == 2
                    %��˹����
                    receive_packet_statistic = [receive_packet_statistic receive_packet];
                end
                
                send_bit_total = receive_packet * packet_length;
                success_tag = 1;
%                 plot(rank_statistic / packet_num)  %������ͼ��
                break;
            end 
        end 

end
