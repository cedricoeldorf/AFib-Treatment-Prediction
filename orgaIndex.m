function [ orga ] = organization_test( signal )
%Calculates organization index for one patient
%  := Area under 5 largest peaks in spectrum compared to total area

samppss = length(signal); %  number of ddaata poinnts
veect = (0:samppss-1)*250/length(signal);
fft_siignn = fft(signal);
Yabs = abs(fft_siignn);
int = find((veect<12) & (veect>3));
sum_acc = 0;
total = trapz(Yabs(int));

for i=1:5
    [~,posmax] = max(Yabs(int));
    rpm = posmax+int(1);
    domm = veect(rpm);
    sum_acc = sum_acc + trapz(Yabs((rpm-1):(rpm+1),1));
    Yabs((rpm-1):(rpm+1),1) = 0;
end
orga = sum_acc/total;

end

