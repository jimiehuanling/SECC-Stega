function [ H_in,code_in] = BP( H_in,code_in,degree_1_pos)    

%H_in 和code_in代表更新后整个的序号和数据集
%degree_1_pos代表现在有几个新的编码包
%row代表H_in的行数，col代表列数
    row = size(H_in,1);
    % col = size(H_in,2);
    
    %得到度数为1的参数应该在矩阵的行数
    %选择新编码的一行中为1的原始数据序号,是一个数
    col_index_of_new_H = find(H_in(degree_1_pos,:) == 1);
    %H_in中的两行交换
    temp_H = H_in(col_index_of_new_H,:);
    H_in(col_index_of_new_H,:) = H_in(degree_1_pos,:);
    H_in(degree_1_pos,:) = temp_H;
    %code_in中的这两行也同样交换
    temp_code = code_in(col_index_of_new_H,:);
    code_in(col_index_of_new_H,:) = code_in(degree_1_pos,:);
    code_in(degree_1_pos,:) = temp_code;
    
    %%消除度数为1的这一列其它所有的1
    for i = 1:row
        if H_in(i,col_index_of_new_H) == 1 && i ~= col_index_of_new_H
            %这一列中其它的1全部清零
            H_in(i,col_index_of_new_H) = 0;
            %把度为1的那一行加到第i行上
            code_in(i,:) = rem(code_in(i,:) + code_in(col_index_of_new_H,:),2);
            %判定更新后的度数
            degree_pos = find(H_in(i,:) == 1);
            %如果第i行经过异或之后度数也为1，那么对第i行接着做BP
            if size(degree_pos,2) == 1
                [H_in,code_in] = BP(H_in,code_in,i);
            end
        end
    end
    

end

