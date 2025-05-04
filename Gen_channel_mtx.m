function H = Gen_channel_mtx...
    (a_taps,tau_taps,A_taps,P,fc,Ts,M_tx,M_rx,roll_off)
% generate channel matrix H in the time domain
%% Input arguments
% a_taps: time scaling for each path
% tau_taps: delay time for each path (s)
% A_taps: path amplitude for each path
% P: the number of paths
% fc: the carrier frequency (Hz)
% Ts: the sampling interval (s) 
% M_tx: the number of sampling points at Tx
% M_rx: the number of sampling points at Rx (M_rx>M_tx)
% roll_off: rolling off factor for the RRC
H = zeros(M_rx,M_tx) ;
% M_g = M_rx - M_tx ;
% equivalent baseband path gain
h_taps = A_taps .* ...
    exp(-1j*2*pi*fc*tau_taps) ;
% discrete delay
l_taps = tau_taps / Ts ;
% sampling index at Tx
m_tx = 0:1:M_tx-1 ;
% sampling index at Rx
m_rx = (0:1:M_rx-1).' ;
for p = 1:P
    % fetch parameters for p-th path
    hp = h_taps(p) ; 
    lp = l_taps(p) ; 
    ap = a_taps(p) ;
    Doppler_p = exp(1j*2*pi*ap*fc*Ts*m_rx) ;
    G_index = (1+ap)*m_rx - m_tx - lp ;
    G_cos = cos(roll_off * pi * G_index) ;
    G_frac = 1 - ((2*roll_off*G_index) ...
        .* (2*roll_off*G_index)) ;
    G_pulse = sinc(G_index) .* (G_cos ./G_frac) ;
    H = H + hp * G_pulse .* Doppler_p ;
end

