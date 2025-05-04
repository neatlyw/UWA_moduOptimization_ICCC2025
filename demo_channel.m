% rng(2025) ;
clear all ;
clc ;
N_train = 20000 ;
N_test = 1000 ;
roll_off = 0.65 ; % the rolling off factor of RRC
P = 8 ; % the number of paths
fc = 12.5e3 ; % the carrier frequency (Hz)
B = 5e3 ;
fs = B / (1+roll_off) ;
Ts = 1 / fs ;
M_tx = 256 ; % the number of sampling points at Tx
M_g = 64 ; % the number of sampling points in guard
M_rx = M_tx + M_g ;
H_train = zeros(N_train,2,M_rx,M_tx) ;
H_test = zeros(N_test,2,M_rx,M_tx) ;
v_max = 20 ; % maximum mobility (kn)
tau_interval = 1e-3 ; % maximum delay spread (s)
tau_max = 10e-3 ;
T_g = M_g * Ts ; % duration of guard
decay_dB = 20 ; % the power difference from 0 to tau_max
for n_mc = 1:N_train
    [a_taps,tau_taps,A_taps] = ...
        Gen_para(tau_interval,v_max,P,decay_dB,T_g) ;
    H = Gen_channel_mtx...
        (a_taps,tau_taps,A_taps,P,fc,Ts,M_tx,M_rx,roll_off) ;
    H_train(n_mc,1,:,:) = real(H) ;
    H_train(n_mc,2,:,:) = imag(H) ;
end
save('H_train.mat','H_train') ;
for n_mc = 1:N_test
    [a_taps,tau_taps,A_taps] = ...
        Gen_para(tau_interval,v_max,P,decay_dB,T_g) ;
    H = Gen_channel_mtx...
        (a_taps,tau_taps,A_taps,P,fc,Ts,M_tx,M_rx,roll_off) ;
    H_test(n_mc,1,:,:) = real(H) ;
    H_test(n_mc,2,:,:) = imag(H) ;
end
save('H_test.mat','H_test') ;