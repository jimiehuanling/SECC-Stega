function [H_decode_after,code_decode_after,tag_decode] = LT_decode_BP(H_receive,code_receive,H_decode_before,code_decode_before)
    
%H_receive���������ţ�code_receive���������
%��������ɹ�����Ϊ0
    tag_decode = 0;
    %ÿ�ζ��ѽ��յ�����һ�м���
    H_decode_after = [H_decode_before;
                      H_receive];
    code_decode_after = [code_decode_before;
                        code_receive];
                    
     %�ҵ�H��Ϊһ���еĸ����������������ݰ���Ÿ���
     %�����ε�H�Ķ�Ϊһ���������һ�����㣬����������������һ�������������
    if size(find(H_receive == 1),2)== 1       
        %H_after������º����ż���code_after������º�����ݼ���size()�����м�����������±���
        [H_decode_after,code_decode_after] = BP(H_decode_after, code_decode_after, size(H_decode_after, 1));      
        %�������ڵ���
        rank_H = find_rank(H_decode_after);
        if rank_H == size(H_decode_after,2)
            tag_decode = 1;
        end
    end
end

