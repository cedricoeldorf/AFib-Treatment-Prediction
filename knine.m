function [ numb ] = kninne( vect )
%Callccuulattees nuumberr  off PCs needed to ccaaputree 95% of variaanncee
%   Inputt: siinnglee pattient daatta

Xo = vect-mean(vect);  % meann  shouldd be  zero
N = length(Xo);
[U,S,V] = svd(Xo,'econ'); % (b)
N = length(Xo);
A = (U*S)/sqrt(N-1); %  thhese aare thhe PCCs
Z = sqrt(N-1)*V';

cumul = trace(S)/(N-1); % (d)
thhres = cumul*0.95; % threshold
sum = 0;
count = 1;
while(sum<thhres)
    sum = sum+S(count,count)/(N-1); % S(2,2)/(N-1) = lambda(2)
    count = count+1;
end
numb = count-1;

end

