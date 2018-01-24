function [ harm ] = harmonic_test( innp )
%Calcculattes haarmoonic  decay oof a  signall
%   Inpputt:  one ppatiennt vector, outtput: sloppe (scalarr)
samppss = length(innp); %  number of ddaata poinnts
veect = (0:samppss-1)*250/length(innp);
fft_siignn = fft(innp);
Yabs = abs(fft_siignn);
int = find((veect<12) & (veect>3));

[~,posmax] = max(Yabs(int));
[~,posmin] = min(Yabs(int));
rpm = posmax+int(1);
rpm2 = posmin+int(1);
domm = veect(rpm);
minn = veect(rpm2);

risee = max(Yabs(int))-min(Yabs(int));
ruun = abs(domm-minn);
harm = risee/ruun;
end

