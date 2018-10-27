function [bestW, best_filters_g, best_filters_h, bestD, bestX, errors, time] = wdla(Y, n, m, k0, filters_g, filters_h)
%W-DLA - Wavelet-like Dictionary Learning -
% A proof of concept implementation of the W-DLA. This implementation is
% optimized for clarity, not performance. See the paper for the details.
%
% [W, filters_g, filters_h, D, X, errors, time] = wdla(Y, n, m, k0)
%
% Input:
%   Y - dataset (p x N)
%   n - length of the filter n <= p
%   m - number of filtering stages
%   k0 - target sparsity: 1 <= k0 <= m
% 
% Output:
%   W - wavelet-like dictionary
%   filters_g, filters_h - filter coefficients
%   D - normalization matrix
%   X - sparse representation matrix
%   errors - Frobenius norm
%   time - execution time
%
% Reference: C. Rusu, On learning with circulant matrices, 2018

% start timer
tic;
[p, N] = size(Y);

if (m > log2(p))
    error('Too many stages m.');
end

if (mod(p/(2^(m)), 1) ~= 0)
    error('2^(m) does not divide p exactly.');
end

if (n > p/(2^(m-1)))
    error('Dimensions n and m mismatch.');
end

if (n > p)
    error('Dimensions n and p mismatch.');
end

init_filters = 1;
if (~exist('filters_g','var'))
    filters_g = randn(m, n);
    init_filters = 0;
end
if (~exist('filters_h','var'))
    filters_h = randn(m, n);
    init_filters = 0;
end

[mg, ng] = size(filters_g); [mh, nh] = size(filters_h);
if (mg ~= m) || (ng~= n)
    error('Filter g sizes mismatch');
end
if (mh ~= m) || (nh~= n)
    error('Filter h sizes mismatch');
end
clear mg ng mh nh;

% [c, ~, ~] = svds(Y, 1);
% C = circulant(c);
% X = omp(C'*Y, C'*C, k0);

if (init_filters == 1)
    W = eye(p);
    for j = 1:m
        the_size = p/(2^(j-1));
        aux_g = circulant([filters_g(j, :) zeros(1, the_size-n)]); aux_g = aux_g(:, 1:2:end);
        aux_h = circulant([filters_h(j, :) zeros(1, the_size-n)]); aux_h = aux_h(:, 1:2:end);
        aux = eye(p);
        aux(1:the_size, 1:the_size) = [aux_g aux_h];

        W = W*aux;
    end
    D = diag(1./my_norms(W, 1));
    WD = W*D;
    X = omp(WD'*Y, WD'*WD, k0);
else
    [W, ~, ~] = svd(Y, 'econ');
    D = eye(p);
    X = omp(W'*Y, eye(p), k0);
end

best = inf;
totalenergy = norm(Y, 'fro')^2;
K = 100;
errors = zeros(K+1, 1);
errors(1) = norm(Y - W*D*X, 'fro')^2/totalenergy*100;
for k = 1:K
    for i = 1:m

        Wa = eye(p);
        for j = 1:i-1
            the_size = p/(2^(j-1));
            aux_g = circulant([filters_g(j, :) zeros(1, the_size-n)]); aux_g = aux_g(:, 1:2:end);
            aux_h = circulant([filters_h(j, :) zeros(1, the_size-n)]); aux_h = aux_h(:, 1:2:end);
            aux = eye(p);
            aux(1:the_size, 1:the_size) = [aux_g aux_h];

            Wa = Wa*aux;
        end

        Wb = eye(p);
        for j = i+1:m
            the_size = p/(2^(j-1));
            aux_g = circulant([filters_g(j, :) zeros(1, the_size-n)]); aux_g = aux_g(:, 1:2:end);
            aux_h = circulant([filters_h(j, :) zeros(1, the_size-n)]); aux_h = aux_h(:, 1:2:end);
            aux = eye(p);
            aux(1:the_size, 1:the_size) = [aux_g aux_h];

            Wb = Wb*aux;
        end

        the_size = p/(2^(i-1));

        Xbar = Wb*D*X; Xbar1 = Xbar(1:the_size, :); Xbar2 = Xbar(the_size+1:end, :);
        Wa1 = Wa(:, 1:the_size); Wa2 = Wa(:, the_size+1:end);
        ybar = vec(Y - Wa2*Xbar2);

        F = 1/sqrt(the_size)*dftmtx(the_size);
        V = transp(kron(eye(2), F(:, 1:2:end))*Xbar1);
        B = Wa1*F';

        clear Wa Wb;
        clear Xbar Xbar1 Xbar2 Wa1 Wa2;
        J = zeros(p*N, 2*the_size);
        for j = 1:the_size
            J(:, j) = kron(V(:,j), B(:,j));
            J(:, the_size+j) = kron(V(:,the_size+j), B(:,j));
        end
        
        clear V B;
        % in Matlab, this is fastest - see paper for the optimized step
        x = real(sqrt(the_size)*J*kron(eye(2), F(:, 1:n)))\ybar;
        
        % when filters are not used in the sparse representations, keep the
        % old ones
        if (norm(x(1:n)) >= 10e-5)
            filters_g(i, :) = x(1:n);
        end
        if (norm(x(n+1:end)) >= 10e-5)
            filters_h(i, :) = x(n+1:end);
        end
    end
    
    W = eye(p);
    for j = 1:m
        the_size = p/(2^(j-1));
        aux_g = circulant([filters_g(j, :) zeros(1, the_size-n)]); aux_g = aux_g(:, 1:2:end);
        aux_h = circulant([filters_h(j, :) zeros(1, the_size-n)]); aux_h = aux_h(:, 1:2:end);
        aux = eye(p);
        aux(1:the_size, 1:the_size) = [aux_g aux_h];

        W = W*aux;
    end
    D = diag(1./my_norms(W, 1));
    
    WD = W*D;
    X = omp(WD'*Y, WD'*WD, k0);

    errors(k+1) = norm(Y - WD*X, 'fro')^2/totalenergy*100;
    
    if (k > 3)
        if (errors(k+1) < best)
            bestW = W;
            bestX = X;
            bestD = D;
            best_filters_h = filters_h;
            best_filters_g = filters_g;
        end
    end
end

time = toc;
