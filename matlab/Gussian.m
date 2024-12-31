function [ H_in,code_in ] = Gussian( H_in,code_in )
%H_in是一个01矩阵
%H_in代表新编码包的异或序号
%code_in代表新编码包
%在H_in的行和列中选择一个较小者作为循环的终点
%选择列数因为接收到的包一定比原来数量的包要多
%循环开始，循环到行数列数的最小值，但最后行数一定大于列数
%还是把矩阵变为单位矩阵的思路
    for col_index = 1:min(size(H_in,1),size(H_in,2))
        
        %row_pos等于H_in中第i列中不等于0的行号，为一个集合
        %相当于选择所有的非零行
        row_pos = find(H_in(:,col_index) ~= 0);
        
        %判断row_pos是否为空集，如果为空集，则说明矩阵中出现了零列
        %零列代表矩阵无解，则跳出整个函数，代表译码失败
        if(isempty(row_pos))
            return;
        end 
        %以上是第i次循环中必做的事情，即找出第i列中的非零行行号，放在row_pos中
        
        %判断
        %这个矩阵中，如果（i,i）主对角线元素为零
        if(H_in(col_index,col_index) == 0)
            %对角元不为1则找为1的行交换  
            %第i列不动，从第i行的下一行开始寻找不为零的行，一直到最后一行
            %row_pos_below应该是一个非零元素那行行号的集合
            row_pos_below = find(H_in(col_index+1:size(H_in,1),col_index) ~= 0);
            %同样，如果非零元素集合为空，说明该列为零列，矩阵无解
            if(isempty(row_pos_below))
                return;
            end            
            
            %row_pos和row_pos_below都是满足要求的行号
            %取出row_pos_below中的第一个元素为row_pos_below(1)
            %和这个循环次数col_index相加
            row_pointer = row_pos_below(1) + col_index;    
            
            %取出相加结果的这一行
            %交换这两行
            temp_H = H_in(row_pointer,:);
            H_in(row_pointer,:) = H_in(col_index,:);
            H_in(col_index,:) = temp_H;
            temp_code = code_in(row_pointer,:);
            code_in(row_pointer,:) = code_in(col_index,:);
            code_in(col_index,:) = temp_code;
            
            %单位矩阵化
            for row_index = row_pos'
                if(row_index ~= row_pointer)                
                    H_in(row_index,:) = rem(H_in(col_index,:) + H_in(row_index,:),2);
                    code_in(row_index,:)  = rem(code_in(col_index,:) + code_in(row_index,:),2);
                end
            end 
        %对角线元素不为零
        else
            for row_index = row_pos'
                %除去本行，其他的非零行与这行做异或运算，相当于把这一列中其他位置的1变成了零
                %只保留循环的那个
                if(row_index ~= col_index)                
                    H_in(row_index,:) = rem(H_in(col_index,:) + H_in(row_index,:),2);
                    code_in(row_index,:)  = rem(code_in(col_index,:) + code_in(row_index,:),2);
                end
            end                        
        end       
    end  
end

