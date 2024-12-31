function [ nc ] = NC( T,I )

%T是原水印图，I是提取水印图
[m,n]=size(T);
[a,b]=size(I);

%判断原图和水印的长宽是否一致
if (m~=a || n~=b)
    error('原图<>提取水印');
    y=0;
    return ;
end
%转换成double类型的数据进行nc运算
T = double(T);
I= double(I);

nc1=0;
nc2=0;
nc3=0;
for i=1:m
   for j=1:n
       nc1=nc1+T(i*j)*I(i*j);
       nc2=nc2+T(i*j)^2;
       nc3=nc3+I(i*j)^2;
   end
end

nc2=double(nc2);
nc3=double(nc3);
nc=nc1/(sqrt(nc2*nc3));

end

