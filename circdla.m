function [C, X, error, time] = circdla(Y, k0, C)
%CDLA The Circulant Dictionary Learning Algorithm - A proof of concept implementation
%of the C-DLA.
%
%[C, X, error, time] = CDLA(Y, k0)
%
% Input:
%   Y - dataset (m x n)
%   k0 - target sparsity: 1 <= k0 <= m
%   C - initialization
%
% Output:
%   C - circulant dictionary
%   X - representation matrix
%   error - Frobenius norm
%   time - running time
%
% Reference: C. Rusu, B. Dumitrescu and S. Tsaftaris,
%	 Explicit shift-invariant dictionary learning, IEEE Signal Processing Letters,
%    21 (1), pp. 6-9, 2013.

% start timer
tic;
[m, n] = size(Y);
steps = 100;

k0 = max(1, round(k0)); k0 = min(m, k0);

%% find good starting point
if (nargin < 3)
    [t1, ~, ~] = svds(Y, 1);
    sigma = fft(t1);
else
    sigma = fft(C(:, 1));
end

cIF = fft(Y);

%% optimization
for k = 1:steps
    %% sparse representation
%     P = real(ifft(diag(conj(sigma))*cIF));
%     g = ifft(abs(sigma).^2); g = g/norm(sigma)^2*m; G = circulant(g);
%     X = omp(P, G, k0);
    C = circulant(ifft(sigma));
    X = omp(C'*Y, C'*C, k0);
    
    %% reduce error
    XF = fft(full(X));
    thenorms = my_norms(XF(1:round(m/2)+1,:), 2).^2;
    
    sigma = zeros(m,1) + 1i*zeros(m,1);
    sigma(1) = thenorms(1)^(-1)*real(XF(1, :))*real(cIF(1, :))';
    
    for j = 2:round(m/2)
          sigma(j) = conj(XF(j,:))*transpose(cIF(j,:))/(thenorms(j));
          sigma(m-j+2) = conj(sigma(j));
    end

    if (mod(m,2) == 0)
      sigma(m/2 + 1) = thenorms(m/2+1)^(-1)*real(XF(m/2+1, :))*real(cIF(m/2+1, :))';
    end

    %% new dictionary C = circulant(ifft(sigma));
    sigma = sigma/norm(sigma)*sqrt(m);
end

C = circulant(ifft(sigma));
X = omp(C'*Y, C'*C, k0);
error = norm(Y-C*X, 'fro');

time = toc;
