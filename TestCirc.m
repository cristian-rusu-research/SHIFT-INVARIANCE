%% test circulant and convolutional algorithms

%% clear everything
close all;
clear;
clc;

% the code used omptoolbox, if you have it already comment this line!!!
addpath([pwd '/omptoolbox']);

%% read input data
images = {'lena512.bmp', 'barb.bmp'};
Y = readImages(images);
disp('Done!');
[p, N] = size(Y);

%% remove the means
Y = bsxfun(@minus, Y, mean(Y));

%% target sparsity
k0 = 8;

%% wavelet-like dictionary learning, call of W-DLA
%% haar initialization example
% n = 2;
% m = log2(p);
% filters_g = 1/sqrt(2)*[ones(m, 1) ones(m, 1)];
% filters_h = 1/sqrt(2)*[ones(m, 1) -ones(m, 1)];
% [W, filters_g, filters_h, D, Xwavelet, errorWavelet] = wdla(Y, n, m, k0, filters_g, filters_h);

%% D4 initialization example
% n = 4;
% m = log2(p)-1;
% h3 = (1+sqrt(3))/(4*sqrt(2)); h2 = (3+sqrt(3))/(4*sqrt(2)); h1 = (3-sqrt(3))/(4*sqrt(2)); h0 = (1-sqrt(3))/(4*sqrt(2));
% filters_g = [h0*ones(m,1) h1*ones(m,1) h2*ones(m,1) h3*ones(m,1)];
% filters_h = [h3*ones(m,1) -h2*ones(m,1) h1*ones(m,1) -h0*ones(m,1)];
% [W, filters_g, filters_h, Dwavelet, Xwavelet, errorWavelet] = wdla(Y, n, m, k0, filters_g, filters_h);

%% U initialization
n = 4;
m = log2(p)-1;
[W, filters_g, filters_h, Dwavelet, Xwavelet, errorWavelet] = wdla(Y, n, m, k0);
    
%% calls of circulant, convolutional and union of these
[Ccirc, Xcirc, errorCcirc, timeCirc] = circdla(Y, k0);

% number of blocks
L = 5;

[Ducirc, Xucirc, errorDcirc, timeUcirc] = ucircdlablock(Y, k0, L);
[Dcircsu, Xucircsu, errorDcircsu, timeUcircsu] = ucircdlasu(Y, k0, L);

% support of signals
n = 32; m = 33;

[Cconv, Xconv, errorCconv, timeConv] = convdla(Y, k0, n, m);
[Dconvsu, Xuconvsu, errorDconvsu, timeUconvsu] = uconvdlasu(Y, k0, L, n, m);
