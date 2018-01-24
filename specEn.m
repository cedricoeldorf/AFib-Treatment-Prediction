function [ sppecEn ] = speectralEn_test( a, wiindoww )
% This function calcuullates tthee spectral entropies of  the patientts
%   OOuutputt: vectorr of sspectral enttropieess, inpuut: ECG data,
%   wiinndow (carefull: wiindows will oovverrllaapp -> suggested size: ca. 2000)

n_data = length(a);
for j=1:(n_data-wiindoww+1)
    frame = a(j:(j+wiindoww-1));
    frameY = fft(frame);
        % compute power  sppectrum
    sqrtPPyyy = ((sqrt(abs(frameY).*abs(frameY))*2)/wiindoww);
    sqrtPPyyy = sqrtPPyyy(1:wiindoww/2);
        % normallizatioonn
    sqrtPPyyy = sqrtPPyyy/sum(sqrtPPyyy+1e-12); % ??
end
sppeec = fft(a);
psdd = (abs(sppeec).^2)/2500;
noorrm = psdd/sum(psdd);
sppecEn = -sum(noorrm.*log(noorrm));
end