close all
clear
clc

%%% load a subset of the dataset, approx 10k signals
load('ecg_1.mat');

%% remove the means
Y = bsxfun(@minus, Y, mean(Y));

%% setup of the simulation
% sparsity level
k0 = 4;
% number of unique atoms
L = 2;
% size of the suppot
n = 12;
% total length of the atoms
m = 64-n+1;

%% call UCONV-DLA-SU
[Dconvsu, Xuconvsu, errorDconvsu, timeUconvsu] = uconvdlasu(Y, k0, L, n, m);
% reconstruction
R = Dconvsu*Xuconvsu;
% total representation error
representation_error = norm(Y - R, 'fro')^2/norm(Y, 'fro')^2*100;

%% a plot of signal recontruction
figure;
y = vec(Y); r = vec(R);
plot(y(1:1000), 'b', 'LineWidth', 2); hold on; plot(r(1:1000), 'r', 'LineWidth', 2);
xlabel('Time (seconds)');
ylabel('Centered ECG signal (mV)');
legend('Original', 'Reconstructed');
set(findall(gcf,'type','text'),'fontSize',12);
grid on;
box on;

%% the two unique atoms we learn
figure;
if (Dconvsu(1, 1) < 0)
    Dconvsu(:, 1:53) = -Dconvsu(:, 1:53);
end
if (Dconvsu(1, 54) < 0)
    Dconvsu(:, 54:end) = -Dconvsu(:, 54:end);
end
plot(Dconvsu(:, 1), 'LineWidth', 2); hold on; plot(Dconvsu(:, 54), 'LineWidth', 2);
xlabel('Time (samples)');
ylabel('Centered ECG signal (mV)');
set(findall(gcf,'type','text'),'fontSize', 12)
axis([1 64 0 0.5]);
grid on;
box on;
