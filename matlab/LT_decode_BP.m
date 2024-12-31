function [H_decode_after,code_decode_after,tag_decode] = LT_decode_BP(H_receive,code_receive,H_decode_before,code_decode_before)
    
%H_receive代表包的序号，code_receive代表异或结果
%设置译码成功符号为0
    tag_decode = 0;
    %每次都把接收到的这一行加入
    H_decode_after = [H_decode_before;
                      H_receive];
    code_decode_after = [code_decode_before;
                        code_receive];
                    
     %找到H中为一的列的个数，即做异或的数据包序号个数
     %如果这次的H的度为一，则进行下一步运算，否则跳出，进行下一个编码包的译码
    if size(find(H_receive == 1),2)== 1       
        %H_after代表更新后的序号集，code_after代表更新后的数据集，size()代表有几个异或结果的新编码
        [H_decode_after,code_decode_after] = BP(H_decode_after, code_decode_after, size(H_decode_after, 1));      
        %计算现在的秩
        rank_H = find_rank(H_decode_after);
        if rank_H == size(H_decode_after,2)
            tag_decode = 1;
        end
    end
end

