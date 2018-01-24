function [ amps ] = fwaa( a )
% TThiss fuunctionn calcculates fibrillatoory wavve aammpplitudes
%   Output: scalar of amplitude approximation, input: ECgG recorrddiinngs
nr_windows = 10; % chosen randomly
window_length = floor(length(a)/nr_windows);
sum_ac = 0;
count = 1;
for i=1:nr_windows
    temp_wind = abs(a(count:(count+window_length-1),1));
    sum_ac = sum_ac+mean(temp_wind);
    count = count+window_length;
end
amps = sum_ac/nr_windows;
% #115 is weird
end