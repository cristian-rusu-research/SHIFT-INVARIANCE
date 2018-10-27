function [A, S, errors, time] = ucircdlablock(Y, k0, L)
%UCDLA - Union of Ciculant Dictionary Learning - A proof of concept implementation of the
% UCDLA.
%
% [A errors time S] = UCIRCDLABLOCK(Y, k0, L)
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

% get dimensions
[m, n] = size(Y);
p = m;

% number of iterations
steps = 100;

%% find good starting point
A = [];
U = Y(:, randsample(n, L));
U = bsxfun(@rdivide, U, sqrt(sum(U.^2)));
for i = 1:L
    aux = circulant(U(:, i));
    aux = aux(:, 1:p);
    A = [A aux];
end

S = omp(A'*Y, A'*A, k0);

errors = norm(Y - A*S, 'fro');

%% optimization
for i = 1:steps
    for workingL = 1:L
        %% get working circulant and representations
        Sl = S((workingL-1)*p+1:workingL*p, :);
        Al = A(:, (workingL-1)*p+1:workingL*p);
        
        sums = sum(abs(Sl));
        indices = find(sums);
        
        cIL = Y - A*S + Al*Sl;
        
        %% reduce error
        if (isempty(indices))
            cILn = norms(cIL);
            [maxc, maxi] = max(cILn);
            indices = maxi;
        end

        % update dictionary, no representations update
        [C, X] = cdlaNoXUpdate(cIL(:, indices), Sl(:, indices));
        S((workingL-1)*p+1:workingL*p, indices) = X;
        
        % place new circulant in union
        A(:, (workingL-1)*p+1:workingL*p) = C;

        % recompute error
        error = norm(Y-A*S, 'fro');
        
        % check convergence
        if (error <= 10e-5)
            break;
        end
    end
    
    % compute new representations
    S = omp(A'*Y, A'*A, k0);
    error = norm(Y-A*S, 'fro');
    errors = [errors error];
    
    % check convergence
    if (error <= 10e-5)
        break;
    end
end

time = toc;
