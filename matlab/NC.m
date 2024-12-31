function [ nc ] = NC( T,I )

%T��ԭˮӡͼ��I����ȡˮӡͼ
[m,n]=size(T);
[a,b]=size(I);

%�ж�ԭͼ��ˮӡ�ĳ����Ƿ�һ��
if (m~=a || n~=b)
    error('ԭͼ<>��ȡˮӡ');
    y=0;
    return ;
end
%ת����double���͵����ݽ���nc����
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

