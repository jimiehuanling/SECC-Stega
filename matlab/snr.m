function [ SNR ] = snr( Y,Y1 )

fz=sum(Y.*Y);                          
fm=sum((Y-Y1).*(Y-Y1));
SNR=10*log(fz/fm);

end

