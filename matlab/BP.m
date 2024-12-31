function [ H_in,code_in] = BP( H_in,code_in,degree_1_pos)    

%H_in ��code_in������º���������ź����ݼ�
%degree_1_pos���������м����µı����
%row����H_in��������col��������
    row = size(H_in,1);
    % col = size(H_in,2);
    
    %�õ�����Ϊ1�Ĳ���Ӧ���ھ��������
    %ѡ���±����һ����Ϊ1��ԭʼ�������,��һ����
    col_index_of_new_H = find(H_in(degree_1_pos,:) == 1);
    %H_in�е����н���
    temp_H = H_in(col_index_of_new_H,:);
    H_in(col_index_of_new_H,:) = H_in(degree_1_pos,:);
    H_in(degree_1_pos,:) = temp_H;
    %code_in�е�������Ҳͬ������
    temp_code = code_in(col_index_of_new_H,:);
    code_in(col_index_of_new_H,:) = code_in(degree_1_pos,:);
    code_in(degree_1_pos,:) = temp_code;
    
    %%��������Ϊ1����һ���������е�1
    for i = 1:row
        if H_in(i,col_index_of_new_H) == 1 && i ~= col_index_of_new_H
            %��һ����������1ȫ������
            H_in(i,col_index_of_new_H) = 0;
            %�Ѷ�Ϊ1����һ�мӵ���i����
            code_in(i,:) = rem(code_in(i,:) + code_in(col_index_of_new_H,:),2);
            %�ж����º�Ķ���
            degree_pos = find(H_in(i,:) == 1);
            %�����i�о������֮�����ҲΪ1����ô�Ե�i�н�����BP
            if size(degree_pos,2) == 1
                [H_in,code_in] = BP(H_in,code_in,i);
            end
        end
    end
    

end

