function [bestA, bestS, errors, time] = ucircdlasu(cI, k0, L)
%UCDLA - Union of Ciculant Dictionary Learning Simultaneous Update -
% A proof of concept implementation of the UCDLA.
%
% [A, S, errors, time] = UCIRCDLASU(cI, k0, L)
%
% Input:
%   Y - dataset (m x n)
%   k0 - target sparsity: 1 <= k0 <= m
%   L - number of basis in the union
% 
% Output:
%   A - union of circulants dictionary
%   S - representation matrix
%   errors - Frobenius norm
%   time - execution time
%
% Reference: C. Rusu, On learning with circulant matrices, 2018

% start timer
tic;
[m, n] = size(cI);
steps = 100;

%% find good starting point
U = cI(:, randsample(n, L));
U = bsxfun(@rdivide, U, sqrt(sum(U.^2)));
A = zeros(m, m*L);
for j = 1:L
    A(:, (j-1)*m+1:j*m) = circulant(U(:, j));
end

S = omp(A'*cI, A'*A, k0);
Sigmas = zeros(m, L);

errors = norm(cI - A*S, 'fro');

bestA = [];
bestS = [];

cItilde = fft(cI);
%% optimization
Sltilde = zeros(size(S));
A = zeros(m, m*L);
for i = 1:steps
    gountil = round(m/2);
    if (mod(m,2) == 0)
        gountil = m/2 + 1;
    end
    
    for workingL = 1:L
        Sltilde((workingL-1)*m+1:workingL*m, :) = fft(full(S((workingL-1)*m+1:workingL*m, :)));
    end
    
    for j = 1:gountil
        Sigmas(j, :) = sqrt(m)*Sltilde((0:L-1).*m+j, :).'\cItilde(j, :).';
        
        if (j > 1)
            Sigmas(m-j+2, :) = conj(Sigmas(j, :));
        end
    end
    
    for j = 1:L
        Sigmas(:,j) = Sigmas(:,j)/norm(Sigmas(:,j))*sqrt(m);
        A(:, (j-1)*m+1:j*m) = circulant(ifft(Sigmas(:,j)));
    end
    
    S = omp(A'*cI, A'*A, k0);
    
    error = norm(cI-A*S, 'fro');
    if (error < errors(end))
        bestA = A;
        bestS = S;
    end
    errors = [errors error];
    
    if (error <= 10e-5)
        break;
    end
end

time = toc;
