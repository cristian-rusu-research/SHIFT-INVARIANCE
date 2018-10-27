function [C, X, error, time] = cdlaNoXUpdate(cI, X)
% A version of C-DLA that does not update the sparse representations.
% It is used by ucdla.

% tic;
[m, n] = size(cI);
error = [];

cIF = fft(cI);

%% reduce error
XF = fft(full(X));
thenorms = my_norms(XF(1:round(m/2)+1,:), 2).^2;

sigma = zeros(m,1) + 1i*zeros(m,1);
sigma(1) = thenorms(1)^(-1)*real(XF(1, :))*real(cIF(1, :))';

for j = 2:round(m/2)
    sigma(j) = conj(XF(j,:))*transp(cIF(j,:))/(thenorms(j));
    sigma(m-j+2) = conj(sigma(j));
end

if (mod(m,2) == 0)
    sigma(m/2 + 1) = thenorms(m/2+1)^(-1)*real(XF(m/2+1, :))*real(cIF(m/2+1, :))';
end

%% optimization
% c = ifft(sigma/cnorm*sqrt(m));
c = ifft(sigma*sqrt(m));
cnorm = norm(c);
c = c/cnorm;
C = circulant(c);

time = 0;
% time = toc;
