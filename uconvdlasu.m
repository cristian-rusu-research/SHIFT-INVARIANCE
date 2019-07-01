function [bestA, bestS, errors, time] = uconvdlasu(cI, k0, L, n, m)
%UCDLA - Union of Convolutional Dictionary Learning Simultaneous Update -
% A proof of concept implementation of the UCONVDLA-SU.
%
% [A, S, errors, time] = UCONVDLASU(Y, k0, L, n, m)
%
% Input:
%   Y - dataset (p x N)
%   k0 - target sparsity: 1 <= k0 <= m
%   L - number of basis in the union
%   n - length of filters
%   m - size of inputs
%   p = n + m - 1
% 
% Output:
%   A - union of convolutional dictionary
%   S - representation matrix
%   errors - Frobenius norm
%   time - execution time
%
% Reference: C. Rusu, On learning with circulant matrices, 2018

% start timer
tic;
[p, N] = size(cI);
steps = 100;

if (p~=(n+m-1))
    error('Dimensions (p,n,m) do not match!');
end

if (n>m)
    error('Dimensions (n,m) do not match!');
end

%% find good starting point
U = cI(:, randsample(N, L));
U = bsxfun(@rdivide, U, sqrt(sum(U.^2)));
A = zeros(p, m*L);
for j = 1:L
    C = circulant([U(1:n, j)/norm(U(1:n, j)); zeros(m-1, 1)]);
    A(:, (j-1)*m+1:j*m) = C(:, 1:m);
end
S = omp(A'*cI, A'*A, k0);

errors = norm(cI - A*S, 'fro');

bestA = [];
bestS = [];

cItilde = fft(cI);
cItilde = cItilde(1:round(p/2)+1, :);
%% optimization
Sltilde = zeros(p*L, N);
A = zeros(p, m*L);
BHB = zeros(n*L, n*L);
v = zeros(n*L, 1);
opts.POSDEF = true; opts.SYM = true;
for i = 1:steps
    % Fourier transformation of the components in X
    for workingL = 1:L
        Sltilde((workingL-1)*p+1:workingL*p, :) = fft(full([S((workingL-1)*m+1:workingL*m, :); zeros(n-1, N)]));
    end
    
    % build v
    for workingL = 1:L
        vv = sum(conj(Sltilde((workingL-1)*p+1:(workingL-1)*p+1+round(p/2), :)).*cItilde, 2);
        for j = p:-1:round(p/2)+2
            vv(j) = conj(vv(p-j+2));
        end
        vv = ifft(vv);
        
        v((workingL-1)*n+1:workingL*n) = vv(1:n);
    end
    
    %build B^HB
    for workingL1 = 1:L
        for workingL2 = workingL1:L
            vv = sum(conj(Sltilde((workingL1-1)*p+1:(workingL1-1)*p+1+round(p/2), :)).*Sltilde((workingL2-1)*p+1:(workingL2-1)*p+1+round(p/2), :), 2);
            for j = p:-1:round(p/2)+2
                vv(j) = conj(vv(p-j+2));
            end
            vv = ifft(vv);
            
            if (workingL1 == workingL2)
                BHB((workingL1-1)*n+1:workingL1*n, (workingL2-1)*n+1:workingL2*n) = toeplitz(vv(1:n), [vv(1); flipud(vv(p-n+2:1:end))]);
            else
                BHB((workingL1-1)*n+1:workingL1*n, (workingL2-1)*n+1:workingL2*n) = toeplitz(vv(1:n), [vv(1); flipud(vv(p-n+2:1:end))]);
                BHB((workingL2-1)*n+1:workingL2*n, (workingL1-1)*n+1:workingL1*n) = BHB((workingL1-1)*n+1:workingL1*n, (workingL2-1)*n+1:workingL2*n)';
            end
        end
    end
    
    c = linsolve(BHB, v, opts);
    Cs = reshape(c, n, L);
    Cs = bsxfun(@rdivide, Cs, sqrt(sum(Cs.^2)));
    
    for j = 1:L
        C = circulant([Cs(:, j); zeros(m-1, 1)]);
        A(:, (j-1)*m+1:j*m) = C(:, 1:m);
    end
    S = omp(A'*cI, A'*A, k0);
    
    error2 = norm(cI-A*S, 'fro');
    if (error2 < errors(end))
        bestA = A;
        bestS = S;
    end
    errors = [errors error2];
    
    if (error2 <= 10e-5)
        break;
    end
end

time = toc;
