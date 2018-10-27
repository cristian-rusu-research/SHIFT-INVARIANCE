function [C, X, error, time] = convdla(Y, k0, n, m)
%ConvDLA The Convolutional Dictionary Learning Algorithm - A proof of concept implementation
%of the Conv-DLA.
%
%[Cconv, X, error, time] = ConvDLA(Y, k0, n, m)
%
% Input:
%   Y - dataset (p x N)
%   k0 - target sparsity: 1 <= k0 <= m
%   n - length of filters
%   m - size of inputs
%   p = n + m - 1
%
% Output:
%   C - convolutional dictionary
%   X - representation matrix
%   error - Frobenius norm
%   time - running time
%
% Reference: C. Rusu, On learning with circulant matrices, 2018

% start timer
tic;
[p, N] = size(Y);
steps = 100;

if (p~=(n+m-1))
    error('Dimensions (p,n,m) do not match!');
end

if (n>m)
    error('Dimensions (n,m) do not match!');
end

k0 = max(1, round(k0)); k0 = min(m, k0);

%% find good starting point
[c, ~, ~] = svds(Y, 1);
c = c(1:n);
c = c./norm(c);

cIF = fft(Y);

%% optimization
for k = 1:steps
    %% sparse representation
    C = circulant([c; zeros(m-1,1)]);
    C = C(:, 1:m);
    X = omp(C'*Y, C'*C, k0);
    
    %% reduce error
    XF = fft(full([X; zeros(n-1, N)]));
    thenorms = my_norms(XF(1:round(p/2)+1,:), 2).^2;
    
    w = zeros(p, 1);
    w(1) = thenorms(1);
    for j = 2:round(p/2)
        w(j) = thenorms(j);
        w(p-j+2) = w(j);
    end
    if (mod(p,2) == 0)
        w(p/2 + 1) = thenorms(p/2+1);
    end

    %% new dictionary
    v = zeros(p,1);
    v(1) = XF(1, :)*transp(cIF(1, :));
    for j = 2:round(p/2)
        v(j) = conj(XF(j, :))*transp(cIF(j, :));
        v(p-j+2) = conj(v(j));
    end
    if (mod(p,2) == 0)
        v(p/2 + 1) = XF(p/2+1, :)*transp(cIF(p/2+1, :));
    end
    v = ifft(v); v = v(1:n); v = v/norm(v);
    
    t = ifft(w); t = t(1:n); t = t/norm(t);
    
    c = toeplitz(t)\v;
    c = c/norm(c);
end

C = circulant([c; zeros(m-1,1)]);
C = C(:, 1:m);
X = omp(C'*Y, C'*C, k0);
% C = circulant([c; zeros(m-1,1)]);
error = norm(Y-C*X, 'fro');

time = toc;
