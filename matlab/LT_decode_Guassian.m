function [ H_decode_after,code_decode_after,tag_decode,rank_statistic ] = LT_decode_Guassian( H_receive,code_receive,H_decode_before,code_decode_before,rank_statistic)
    %tag_decode的初始值设置为0，表示没有译码成功
    tag_decode = 0;
    %H_decode_after、code_decode_after代表更新后的
    H_decode_after = [H_decode_before;
                      H_receive];
    code_decode_after = [code_decode_before;
                        code_receive];
                    
    [H_decode_after,code_decode_after] = Gussian(H_decode_after,code_decode_after);
    %计算现在的秩
    rank_H = find_rank(H_decode_after);
    rank_statistic = [rank_statistic rank_H];
    %如果现在满秩了，说明译码成功，tag_decode改为1
    if rank_H == size(H_decode_after,2)
        tag_decode = 1;
    end


end

