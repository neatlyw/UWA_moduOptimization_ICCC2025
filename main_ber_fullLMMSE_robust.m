clear all ;
clc ;
warning off ;
P = 15 ; % the number of paths
fc = 35e3 ; % the carrier frequency (Hz)
roll_off = 0.65 ; % the rolling off factor of RRC
B = 5e3 ;
fs = B / (1+roll_off) ; % the sampling rate (Hz), equivalent to the bandwidth
Ts = 1 / fs ;
M_tx = 256 ;
M_g = 64 ;
% M_g = M_tx / 4 ;
M_rx = M_tx + M_g ;
M_data = M_tx ; % the number of data symbols
v_max = 10 ; % maximum mobility (kn)
tau_interval = 1e-3 ;
tau_max = 10e-3 ;
T_g = M_g*Ts ; % duration of guard
decay_dB = 20 ; % the power difference from 0 to Tg
% ZP-OFDM modulation matrix
F_ofdm = dftmtx(M_tx) / sqrt(M_tx) ;
F_ofdm = F_ofdm' ;
load('F.mat','F') ;
F_propose = squeeze(F(1,:,:)+1j*F(2,:,:)) ;
% F_propose = 2*eye(M_data) ;
F_sc = eye(M_data) ;
eye_M = eye(M_data) ;
M_ODDM = 64 ;
N_ODDM = M_data / M_ODDM ;
eye_ODDM = eye(M_ODDM) ;
dft_ODDM = dftmtx(N_ODDM) / sqrt(N_ODDM) ;
F_ODDM = kron(dft_ODDM',eye_ODDM) ;
qam_mod = 4 ; % number of elements in QAM alphabets
qam_bit = log2(qam_mod) ; % bits per symbol for QPSK
num_bit = M_data * qam_bit ; % number of bits per transmission
% power per symbol
% eng_sqrt = (qam_mod==2)+(qam_mod~=2)*sqrt((qam_mod-1)/6*(2^2)) ;
% sigmas2 = eng_sqrt * eng_sqrt ;
SNR_dB = 0:5:30 ;
SNR_linear = 10.^(SNR_dB/10) ;
sigma2_cand = 1 ./ SNR_linear / qam_bit * (2/(1+roll_off)) ;
length_SNR = length(SNR_dB) ;
ber_ofdm = zeros(length_SNR,1) ;
ber_sc = zeros(length_SNR,1) ;
ber_oddm = zeros(length_SNR,1) ;
ber_propose = zeros(length_SNR,1) ;
N_mc = 100000 ; % the number of monte-carlo
for n_mc = 1:N_mc
    if mod(n_mc-1,N_mc/100) == 0
        fprintf('%3.2f%% finished \n',(n_mc-1)/(N_mc/100)) 
    end 
    % generate channel parameters
    [a_taps,tau_taps,A_taps] = ...
        Gen_para(tau_interval,v_max,P,decay_dB,T_g) ;
    % channel matrix in the delay domain
    H = Gen_channel_mtx...
        (a_taps,tau_taps,A_taps,P,fc,Ts,M_tx,M_rx,roll_off) ;
    H_ofdm = H * F_ofdm ; % channel matrix for OFDM
    H_propose = H * F_propose ;
    H_sc = H * F_sc ;
    H_oddm = H * F_ODDM ;
    % generate data vector
    data_bit = randi([0 1],qam_bit,M_data) ;
    data_sym = qammod(data_bit,qam_mod,'InputType','bit','UnitAveragePower',true) ;
    data_sym = data_sym.' ;
    % sequence-level modulation
    s_tx_ofdm = F_ofdm * data_sym ; 
    s_tx_sc = F_sc * data_sym ;
    s_tx_oddm = F_ODDM * data_sym ;
    s_tx_propose = F_propose * data_sym ; 
    % transmission
    r_rx_ofdm_wn = H * s_tx_ofdm ;
    r_rx_propose_wn = H * s_tx_propose ;
    r_rx_sc_wn = H * s_tx_sc ;
    r_rx_oddm_wn = H * s_tx_oddm ;
    n_norm = sqrt(1/2) * (randn(M_rx,1)...
        + 1j*randn(size(M_rx,1)));
    for b = 1:length(sigma2_cand)
        sigma2 = sigma2_cand(b) ;
        % sigma2 = 0 ;
        % additive noise
        r_rx_ofdm = r_rx_ofdm_wn + sqrt(sigma2) * n_norm ;
        r_rx_propose = r_rx_propose_wn + sqrt(sigma2) * n_norm ;
        r_rx_sc = r_rx_sc_wn + sqrt(sigma2) * n_norm ;
        r_rx_oddm = r_rx_oddm_wn + sqrt(sigma2) * n_norm ;
        % BER computation
        % fully ICI-aware ZP-OFDM
        data_rx = (H_ofdm'*H_ofdm+sigma2*eye_M) \ (H_ofdm'*r_rx_ofdm) ;
        bit_est = qamdemod(data_rx.',qam_mod,'OutputType','bit','UnitAveragePower',true) ;
        ber_ofdm(b) = ber_ofdm(b) ...
            + sum(sum(data_bit~=bit_est)) ;
        % SC
        data_rx = (H_sc'*H_sc+sigma2*eye_M) \ (H_sc'*r_rx_sc) ;
        bit_est = qamdemod(data_rx.',qam_mod,'OutputType','bit','UnitAveragePower',true) ;
        ber_sc(b) = ber_sc(b) ...
            + sum(sum(data_bit~=bit_est)) ;
        % ODDM
        data_rx = (H_oddm'*H_oddm+sigma2*eye_M) \ (H_oddm'*r_rx_oddm) ;
        bit_est = qamdemod(data_rx.',qam_mod,'OutputType','bit','UnitAveragePower',true) ;
        ber_oddm(b) = ber_oddm(b) ...
            + sum(sum(data_bit~=bit_est)) ;
        % proposed modulation
        data_rx = (H_propose'*H_propose+sigma2*eye_M) \ (H_propose'*r_rx_propose) ;
        bit_est = qamdemod(data_rx.',qam_mod,'OutputType','bit','UnitAveragePower',true) ;
        ber_propose(b) = ber_propose(b) ...
            + sum(sum(data_bit~=bit_est)) ;
    end
end
ber_ofdm = ber_ofdm / N_mc / num_bit ;
ber_oddm = ber_oddm / N_mc / num_bit ;
ber_sc = ber_sc / N_mc / num_bit ;
ber_propose = ber_propose / N_mc / num_bit ;
figure ;
plot(SNR_dB,ber_ofdm,'-+','Color',...
    [0.4940 0.1840 0.5560],'LineWidth',1.5) ;
hold on ;
plot(SNR_dB,ber_sc,'-*','Color',...
    [0.8500 0.3250 0.0980],'LineWidth',1.5) ;
hold on ;
plot(SNR_dB,ber_oddm,'-^','Color',...
    [0.4660 0.6740 0.1880],'LineWidth',1.5) ;
hold on ;
plot(SNR_dB,ber_propose,'m-p'...
    ,'LineWidth',1.5) ;
hold on ;
% plot(SNR_dB,ber_propose_5diag,'-s','Color',...
%     [0.4940 0.1840 0.5560],'LineWidth',1.5) ;
% hold on ;
% plot(SNR_dB,ber_propose_3diag,'m-p'...
%     ,'LineWidth',1.5) ;
% hold on ;
xlabel('SNR in dB') ;
ylabel('BER') ;
legend('OFDM','single carrier'...
    ,'ODDM','proposed modulation') ;
% legend('ICI-aware ZP-OFDM','only nearby ICI ZP-OFDM'...
%     ,'ICI-ignorant ZP-OFDM') ;
grid on ;
set(gca,'YScale','log') ;  