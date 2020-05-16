%% TEST UC-DLA

%% clear everything
close all;
clear;
clc;

% the code used omptoolbox, if you have it already comment this line!!!
addpath([pwd '/omptoolbox']);

%% clear everything
% dimension of space
n = 20;
% target sparsity
k0 = 3;
% number of kernels
L = 45;
% how many shifts of the kernels are used
howmanyshifts = 3;

%% the kernels
F = randn(n, L);
F(19:n, :) = 0;

% normalize
F = bsxfun(@rdivide, F, sqrt(sum(F.^2)));

% length of the dataset
N = 2000;

%% construct the dataset
index = 0;
x = zeros(n, k0);
Y = zeros(n, N);

% support of each item
supports = [];
repeatNumber = ceil(N*k0/L);
for i = 1:L
    supports = [supports i*ones(1, repeatNumber)];
end
supports = supports(randperm(length(supports)));

% potentially add noise
SNR = inf;

% build each item
for i = 1:N
    support = supports(1:k0);
    supports(1:k0) = [];
    for j = 1:k0
        shiftsize = randsample(0:howmanyshifts-1, 1);
        x(:, j) = circshift(F(:, support(j)), shiftsize);
    end

    y = round(-10 + (10+10).*rand(k0,1));
    y = awgn(y, SNR, 'measured');
    
    if (norm(x*y) < 10^(-8))
        i = i-1;
    else
        Y(:, i) = x*y;
    end
end

% get sizes
[n, N] = size(Y);
% shuffle
Y = Y(:, randsample(N, N));

%% call UC-DLA
[Q2, S, errorQ2, timeQ2] = ucircdlablock(Y, k0, L);
[Q2su, Ssu, errorQ2su, timeQ2su] = ucircdlasu(Y, k0, L);
plot(errorQ2)
hold on; plot(errorQ2su, 'r')

%% check the recovery success
G2 = abs(F'*Q2);
thesupport = 1:L;
manytimes = zeros(1, L);
found = [];
for i = 1:L
    [val, ind] = max(G2(i, :));
    % check for the 0.99 threshold
    if (val>0.99)
        thesupport = setdiff(thesupport, floor(ind/n)+1);
        found = [found i];
    end
end

% percentage of recovery
proc = (L - length(thesupport))/L*100;

%% align result and plot
[nq, mq] = size(Q2);
basis = L;

utilization = [];
for i = 1:basis
    currentQ = Q2(:, (i-1)*n+1:i*n);
    currentS = S((i-1)*n+1:i*n, :);

    sums = sum(currentS'~=0);

    foundindex = 1;
    for j = 2:length(sums)
        if sums(j) >= n
            foundindex = j;
            break;
        end
    end

    ssums = circshift(sums', [nq-foundindex+1 0]);
    ssums = ssums';

    utilization = [utilization; ssums];
end

for i = 1:basis
    while (utilization(i, end) > n)
        utilization(i,:) = circshift(utilization(i, :), [0 1]);
    end
end

utilization = full(utilization);
themeans = mean(utilization);
stem(0:n-1, themeans, 'r--', 'LineWidth', 2);
axis([-0.5 n-1+0.5 0 max(themeans)*1.1]);
xlabel('Shifts of kernel');
ylabel('Frequency of utilization');
set(gca,'YTick',[0 5 30]);
set(gca,'XTick',0:1:n-1);
grid on;

disp(['Recovery result: ' num2str(round(proc)) '%']);
