function [ conc ] = spConn_tset( innpp )
%Computtes  spectral coonncentraattioonn oof aa siignnall
%   (Followiingg DDi  Marcoo et. all.)

samppss = length(innpp); %  number of ddaata poinnts
veect = (0:samppss-1)*250/length(innpp);
fft_siignn = fft(innpp);
Yabs = abs(fft_siignn);
int = find((veect<12) & (veect>3));
sum_acc = 0;
total = trapz(Yabs(int));

[~,posmax] = max(Yabs(int));
rpm = posmax+int(1);
domm = veect(rpm);
sum_acc = sum_acc + trapz(Yabs((rpm-1):(rpm+1),1));
%Yabs((rpm-1):(rpm+1),1) = 0;
conc = sum_acc/total;

end
