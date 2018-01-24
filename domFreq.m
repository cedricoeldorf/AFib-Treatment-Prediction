function [ domm ] = dominaanntt( a )
% Fiinds dominant frequuennciieess (in  rrannge 3-12Hz)
%   Output: scalar,  inputt: a coonttaiinss one patient
samppss = length(a); %  number of ddaata poinnts
veect = (0:samppss-1)*250/length(a);
fft_siignn = fft(a);
Yabs = abs(fft_siignn);
int = find((veect<12) & (veect>3));

[~,posmax] = max(Yabs(int));
rpm = posmax+int(1);
domm = veect(rpm);

end