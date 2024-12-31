function [ psn ] = psnr( Y,Y1 )

[row,col]=size(Y);
mse=0;
for i=1:row
   for j=1:col
       mse=mse+(Y(i,j)-Y1(i,j))^2;
   end 
end
mse=mse/(row*col);
psn=10*log10(255^2/mse);

end

