%% AMATH 582 HW4

clc; clear all; close all;
fig = 1;

%% Load Music

% Mac Miller (rap)---------------------------------------------------------

[mac1, Fs_mac1] = audioread("MacMiller_WhatsTheUse.mp3");
[mac2, Fs_mac2] = audioread("MacMiller_Ladders.mp3");
[mac3, Fs_mac3] = audioread("MacMiller_GoodNews.mp3");

% Avicii (EDM)-------------------------------------------------------------

[avicii1, Fs_avicii1] = audioread("Avicii_TheNights.mp3");
[avicii2, Fs_avicii2] = audioread("Avicii _SOS.mp3");
[avicii3, Fs_avicii3] = audioread("Avicii_HeyBrother.mp3");

% Khalid (R&B)-------------------------------------------------------------

[khalid1, Fs_khalid1] = audioread("Khalid_UpAllNight.mp3");
[khalid2, Fs_khalid2] = audioread("Khalid _Eleven.mp3");
[khalid3, Fs_khalid3] = audioread("Khalid_Better.mp3");

%% Create music 5 sec music vectors

Fs = Fs_mac1; % sample rate is same for all songs

t = 0:1/Fs:5;

% Truncated (5 sec) music samples

mac1_truncated = mac1(1:length(t), 1);
mac2_truncated = mac2(1:length(t), 1);
mac3_truncated = mac3(1:length(t), 1);

avicii1_truncated = avicii1(1:length(t), 1);
avicii2_truncated = avicii2(1:length(t), 1);
avicii3_truncated = avicii3(1:length(t), 1);

khalid1_truncated = khalid1(1:length(t), 1);
khalid2_truncated = khalid2(1:length(t), 1);
khalid3_truncated = khalid3(1:length(t), 1);

figure(fig); fig = fig+1
plot(t, mac1_truncated)

figure(fig); fig = fig+1
plot(t, avicii1_truncated)

figure(fig); fig = fig+1
plot(t, khalid1_truncated)

%% Spectrograms

Spec_mac1 = spectrogram(mac1_truncated);
Spec_mac2 = spectrogram(mac2_truncated);
Spec_mac3 = spectrogram(mac3_truncated);
Spec_mac = [Spec_mac1 Spec_mac2 Spec_mac3];

Spec_avicii1 = spectrogram(avicii1_truncated);
Spec_avicii2 = spectrogram(avicii2_truncated);
Spec_avicii3 = spectrogram(avicii3_truncated);
Spec_avicii = [Spec_avicii1 Spec_avicii2 Spec_avicii3];

Spec_khalid1 = spectrogram(khalid1_truncated);
Spec_khalid2 = spectrogram(khalid2_truncated);
Spec_khalid3 = spectrogram(khalid3_truncated);
Spec_khalid = [Spec_khalid1 Spec_khalid2 Spec_khalid3];

%%

[U_mac,S_mac,V_mac] = svd(abs(Spec_mac), 'econ');
[U_avicii,S_avicii,V_avicii] = svd(abs(Spec_avicii), 'econ');
[U_khalid,S_khalid,V_khalid] = svd(abs(Spec_khalid), 'econ');

figure(fig); fig = fig+1;
subplot(2,1,1) 
plot(diag(S_mac),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80]) 
subplot(2,1,2) 
semilogy(diag(S_mac),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80])

figure(fig); fig = fig+1;
subplot(2,1,1) 
plot(diag(S_avicii),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80]) 
subplot(2,1,2) 
semilogy(diag(S_avicii),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80])

figure(fig); fig = fig+1;
subplot(2,1,1) 
plot(diag(S_khalid),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80]) 
subplot(2,1,2) 
semilogy(diag(S_khalid),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80])

%% Classification

train = [V_mac(1:20, 1:6); V_avicii(1:20, 1:6); V_khalid(1:20, 1:6)];
label = [1*ones(20, 1); 2*ones(20,1); 3*ones(20,1)];
test = [V_mac(21:24, 1:6); V_avicii(21:24, 1:6); U_khalid(21:24, 1:6)];
class = classify(test, train, label);
figure(fig); fig=fig+1;
bar(class);







