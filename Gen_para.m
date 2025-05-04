function [a_taps,tau_taps,A_taps] = ...
    Gen_para(tau_interval,v_max,P,decay_dB,T_g)
% Generate UWA channel parameters
%% input arguments
% tau_max: maximum time delay (s)
% v_max: maximum mobility velocity (kn)
% P: the number of paths
% decay_dB: the power difference from 0 to tau_max
%% generate a_taps - scaling factors
v_max = v_max * 1.852 / 3.6 ;
v_taps = v_max * cos(2*pi*rand(P,1)-pi) ;
c = 1500 ; % the velocity of sound
a_taps = v_taps / c ;
%% generate tau_taps - delay (s)
tau_pre = exprnd(tau_interval,P,1) ;
tau_taps = tau_pre(1) * ones(P,1) ;
for p = 2:P
    tau_taps(p) = tau_taps(p-1) + tau_pre(p) ;
end
% tau_taps = tau_max * sort(rand(P,1)) ;
%% generate A_taps - path amplitude
decay_coe = 10^(decay_dB/10) ;
power_taps = exp(-log(decay_coe)*tau_taps/T_g) ;
power_taps = power_taps ./ sum(power_taps) ;
A_taps = raylrnd(sqrt(power_taps/2)) ;


