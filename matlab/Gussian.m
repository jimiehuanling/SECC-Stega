function [ H_in,code_in ] = Gussian( H_in,code_in )
%H_in��һ��01����
%H_in�����±������������
%code_in�����±����
%��H_in���к�����ѡ��һ����С����Ϊѭ�����յ�
%ѡ��������Ϊ���յ��İ�һ����ԭ�������İ�Ҫ��
%ѭ����ʼ��ѭ����������������Сֵ�����������һ����������
%���ǰѾ����Ϊ��λ�����˼·
    for col_index = 1:min(size(H_in,1),size(H_in,2))
        
        %row_pos����H_in�е�i���в�����0���кţ�Ϊһ������
        %�൱��ѡ�����еķ�����
        row_pos = find(H_in(:,col_index) ~= 0);
        
        %�ж�row_pos�Ƿ�Ϊ�ռ������Ϊ�ռ�����˵�������г���������
        %���д�������޽⣬������������������������ʧ��
        if(isempty(row_pos))
            return;
        end 
        %�����ǵ�i��ѭ���б��������飬���ҳ���i���еķ������кţ�����row_pos��
        
        %�ж�
        %��������У������i,i�����Խ���Ԫ��Ϊ��
        if(H_in(col_index,col_index) == 0)
            %�Խ�Ԫ��Ϊ1����Ϊ1���н���  
            %��i�в������ӵ�i�е���һ�п�ʼѰ�Ҳ�Ϊ����У�һֱ�����һ��
            %row_pos_belowӦ����һ������Ԫ�������кŵļ���
            row_pos_below = find(H_in(col_index+1:size(H_in,1),col_index) ~= 0);
            %ͬ�����������Ԫ�ؼ���Ϊ�գ�˵������Ϊ���У������޽�
            if(isempty(row_pos_below))
                return;
            end            
            
            %row_pos��row_pos_below��������Ҫ����к�
            %ȡ��row_pos_below�еĵ�һ��Ԫ��Ϊrow_pos_below(1)
            %�����ѭ������col_index���
            row_pointer = row_pos_below(1) + col_index;    
            
            %ȡ����ӽ������һ��
            %����������
            temp_H = H_in(row_pointer,:);
            H_in(row_pointer,:) = H_in(col_index,:);
            H_in(col_index,:) = temp_H;
            temp_code = code_in(row_pointer,:);
            code_in(row_pointer,:) = code_in(col_index,:);
            code_in(col_index,:) = temp_code;
            
            %��λ����
            for row_index = row_pos'
                if(row_index ~= row_pointer)                
                    H_in(row_index,:) = rem(H_in(col_index,:) + H_in(row_index,:),2);
                    code_in(row_index,:)  = rem(code_in(col_index,:) + code_in(row_index,:),2);
                end
            end 
        %�Խ���Ԫ�ز�Ϊ��
        else
            for row_index = row_pos'
                %��ȥ���У������ķ�������������������㣬�൱�ڰ���һ��������λ�õ�1�������
                %ֻ����ѭ�����Ǹ�
                if(row_index ~= col_index)                
                    H_in(row_index,:) = rem(H_in(col_index,:) + H_in(row_index,:),2);
                    code_in(row_index,:)  = rem(code_in(col_index,:) + code_in(row_index,:),2);
                end
            end                        
        end       
    end  
end

