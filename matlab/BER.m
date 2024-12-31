function [berating] = BER(Im,I2)
%T是原水印图，I是提取水印图
[m,n]=size(Im);
[a,b]=size(I2);

%判断原图和水印的长宽是否一致
if (m~=a || n~=b)
    error('原图<>提取水印');
    y=0;
    return ;
end

%转换成double类型的数据进行nc运算
Im = double(Im);
I2= double(I2);

difference_matrix = (Im ~= I2);
% 统计不同元素的数量
num_different_elements = sum(difference_matrix(:));
% 计算误码率

length_of_wm = m*n;
berating = num_different_elements/length_of_wm;
end