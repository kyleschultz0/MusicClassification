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

for i = 0:9
    mac1_truncated(:, i+1) = mac1((1+10*i)*Fs:length(t)+(1+10*i)*Fs, 1);
    mac2_truncated(:, i+1) = mac2((1+10*i)*Fs:length(t)+(1+10*i)*Fs, 1);
    mac3_truncated(:, i+1) = mac3((1+10*i)*Fs:length(t)+(1+10*i)*Fs, 1);
    
    avicii1_truncated(:, i+1) = avicii1((1+10*i)*Fs:length(t)+(1+10*i)*Fs, 1);
    avicii2_truncated(:, i+1) = avicii2((1+10*i)*Fs:length(t)+(1+10*i)*Fs, 1);
    avicii3_truncated(:, i+1) = avicii3((1+10*i)*Fs:length(t)+(1+10*i)*Fs, 1);

    khalid1_truncated(:, i+1) = khalid1((1+10*i)*Fs:length(t)+(1+10*i)*Fs, 1);
    khalid2_truncated(:, i+1) = khalid2((1+10*i)*Fs:length(t)+(1+10*i)*Fs, 1);
    khalid3_truncated(:, i+1) = khalid3((1+10*i)*Fs:length(t)+(1+10*i)*Fs, 1);
end



%% Spectrograms

for i = 1:10
    Spec_mac1(:, i) = reshape(spectrogram(mac1_truncated(:, i), 1.1025e+04), [8193*38 1]);
    Spec_mac2(:, i) = reshape(spectrogram(mac2_truncated(:, i), 1.1025e+04), [8193*38 1]);
    Spec_mac3(:, i) = reshape(spectrogram(mac3_truncated(:, i), 1.1025e+04), [8193*38 1]);
    
    Spec_avicii1(:, i) = reshape(spectrogram(avicii1_truncated(:, i), 1.1025e+04), [8193*38 1]);
    Spec_avicii2(:, i) = reshape(spectrogram(avicii2_truncated(:, i), 1.1025e+04), [8193*38 1]);
    Spec_avicii3(:, i) = reshape(spectrogram(avicii3_truncated(:, i), 1.1025e+04), [8193*38 1]);
    
    Spec_khalid1(:, i) = reshape(spectrogram(khalid1_truncated(:, i), 1.1025e+04), [8193*38 1]);
    Spec_khalid2(:, i) = reshape(spectrogram(khalid2_truncated(:, i), 1.1025e+04), [8193*38 1]);
    Spec_khalid3(:, i) = reshape(spectrogram(khalid3_truncated(:, i), 1.1025e+04), [8193*38 1]);
end

Spec_mac = [Spec_mac1 Spec_mac2 Spec_mac3];
Spec_avicii = [Spec_avicii1 Spec_avicii2 Spec_avicii3];
Spec_khalid = [Spec_khalid1 Spec_khalid2 Spec_khalid3];



%% Assembling matrix with all samples and taking SVD

Spec_mac_normal = abs(Spec_mac) - abs(mean(Spec_mac));
Spec_avicii_normal = abs(Spec_avicii) - abs(mean(Spec_avicii));
Spec_khalid_normal = abs(Spec_khalid) - abs(mean(Spec_khalid));
Spec_train = [Spec_mac_normal(:, 1:20) Spec_avicii_normal(:, 1:20) Spec_khalid_normal(:, 1:20)];
Spec_test = [Spec_mac_normal(:, 21:30) Spec_avicii_normal(:, 21:30) Spec_khalid_normal(:, 21:30)];


[U,S,V] = svd(Spec_train, 'econ');

figure(fig); fig = fig+1;
subplot(2,1,1) 
plot(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80]) 
subplot(2,1,2) 
semilogy(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80])

%% Classification

figure(fig); fig=fig+1;

plot3(V(1:20, 2), V(1:20, 3), V(1:20, 4), 'ko')
hold on
plot3(V(21:40, 2), V(21:40, 3), V(21:40, 4), 'ro')
hold on
plot3(V(41:60, 2), V(41:60, 3), V(41:60, 4), 'go')
legend("Mac Miller", "Avicii", "Khalid")
title("Projection onto Principle Components")
xlabel("Component 2"); ylabel("Component 3"); zlabel("Component 4");


figure(fig); fig=fig+1;
train1 = ((U(:, 2:6))'*Spec_train)';
label1 = [1*ones(20, 1); 2*ones(20,1); 3*ones(20,1)];
test1 = ((U(:, 2:6))'*Spec_test)';
class1 = classify(test1, train1, label1);
bar(class1);
title("Song Classification")
xlabel("Song Sample")
ylabel("Classification")

truth = [1*ones(10, 1); 2*ones(10,1); 3*ones(10,1)];

num_correct_case1 = 0;

for i = 1:length(class1)
    if class1(i) == truth(i)
        num_correct_case1 = num_correct_case1 + 1;
    end
end

percent_correct_1 = (num_correct_case1/length(class1))*100;




%% Bands of same genre (rap)

% Reusing Mac Miller

% Drake -------------------------------------------------------------

drake1 = audioread("Drake_Nonstop.mp3");
drake2 = audioread("Drake_InMyFeelings.mp3");
drake3 = audioread("Drake_GodsPlan.mp3");

% Lil Yachty-------------------------------------------------------------

yachty1 = audioread("Lil Yachty_ImTheMac.mp3");
yachty2 = audioread("Lil Yachty_Broccoli.mp3");
yachty3 = audioread("Lil Yachty_1Night.mp3");

%% Create music 5 sec music vectors

% Truncated (5 sec) music samples

for i = 0:9
    
    drake1_truncated(:, i+1) = drake1((1+10*i)*Fs:length(t)+(1+10*i)*Fs, 1);
    drake2_truncated(:, i+1) = drake2((1+10*i)*Fs:length(t)+(1+10*i)*Fs, 1);
    drake3_truncated(:, i+1) = drake3((1+10*i)*Fs:length(t)+(1+10*i)*Fs, 1);

    yachty1_truncated(:, i+1) = yachty1((1+10*i)*Fs:length(t)+(1+10*i)*Fs, 1);
    yachty2_truncated(:, i+1) = yachty2((1+10*i)*Fs:length(t)+(1+10*i)*Fs, 1);
    yachty3_truncated(:, i+1) = yachty3((1+10*i)*Fs:length(t)+(1+10*i)*Fs, 1);
    
end

%% Spectrograms

for i = 1:10
    
    Spec_drake1(:, i) = reshape(spectrogram(drake1_truncated(:, i), 1.1025e+04), [8193*38 1]);
    Spec_drake2(:, i) = reshape(spectrogram(drake2_truncated(:, i), 1.1025e+04), [8193*38 1]);
    Spec_drake3(:, i) = reshape(spectrogram(drake3_truncated(:, i), 1.1025e+04), [8193*38 1]);
    
    Spec_yachty1(:, i) = reshape(spectrogram(yachty1_truncated(:, i), 1.1025e+04), [8193*38 1]);
    Spec_yachty2(:, i) = reshape(spectrogram(yachty2_truncated(:, i), 1.1025e+04), [8193*38 1]);
    Spec_yachty3(:, i) = reshape(spectrogram(yachty3_truncated(:, i), 1.1025e+04), [8193*38 1]);
end

Spec_drake = [Spec_drake1 Spec_drake2 Spec_drake3];
Spec_yachty = [Spec_yachty1 Spec_yachty2 Spec_yachty3];



%% Assembling matrix with all samples and taking SVD

Spec_drake_normal = abs(Spec_drake) - abs(mean(Spec_drake));
Spec_yachty_normal = abs(Spec_yachty) - abs(mean(Spec_yachty));

Spec_train = [Spec_mac_normal(:, 1:20) Spec_drake_normal(:, 1:20) Spec_yachty_normal(:, 1:20)];
Spec_test = [Spec_mac_normal(:, 21:30) Spec_drake_normal(:, 21:30) Spec_yachty_normal(:, 21:30)];


[U,S,V] = svd(Spec_train, 'econ');

figure(fig); fig = fig+1;
subplot(2,1,1) 
plot(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80]) 
subplot(2,1,2) 
semilogy(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80])

%% Classification

figure(fig); fig=fig+1;

plot3(V(1:20, 1), V(1:20, 2), V(1:20, 3), 'ko')
hold on
plot3(V(21:40, 1), V(21:40, 2), V(21:40, 3), 'ro')
hold on
plot3(V(41:60, 1), V(41:60, 2), V(41:60, 3), 'go')
legend("Mac Miller", "Drake", "Lil Yachty")
title("Projection onto Principle Components")
xlabel("Component 1"); ylabel("Component 2"); zlabel("Component 3");

figure(fig); fig=fig+1;
train1 = ((U(:, 2:5))'*Spec_train)';
label1 = [1*ones(20, 1); 2*ones(20,1); 3*ones(20,1)];
test1 = ((U(:, 2:5))'*Spec_test)';
class2 = classify(test1, train1, label1);
bar(class2);
title("Song Classification")
xlabel("Song Sample")
ylabel("Classification")

truth = [1*ones(10, 1); 2*ones(10,1); 3*ones(10,1)];

num_correct_case2 = 0;

for i = 1:length(class2)
    if class2(i) == truth(i)
        num_correct_case2 = num_correct_case2 + 1;
    end
end

percent_correct_2 = (num_correct_case2/length(class2))*100;



%% Different Genres

% Hour long rap mix
rap = audioread("rap_1hr.mp3");

% Hour long EDM mix
edm = audioread("house_1hr.mp3");

% Hour long classical mix
classical = audioread("classical_1hr.mp3");

%% 5 second samples

for i = 0:89
    
    edm_truncated(:, i+1) = edm((1+30*i)*Fs:length(t)+(1+30*i)*Fs, 1);
    classical_truncated(:, i+1) = classical((1+30*i)*Fs:length(t)+(1+30*i)*Fs, 1);
    rap_truncated(:, i+1) = rap((1+30*i)*Fs:length(t)+(1+30*i)*Fs, 1);

end

%% Spectograms

for i = 1:90
    
    Spec_edm(:, i) = reshape(spectrogram(edm_truncated(:, i), 1.1025e+04), [8193*38 1]);
    Spec_classical(:, i) = reshape(spectrogram(classical_truncated(:, i), 1.1025e+04), [8193*38 1]);
    Spec_rap(:, i) = reshape(spectrogram(rap_truncated(:, i), 1.1025e+04), [8193*38 1]);
    
end


%% Assembling matrix with all samples and taking SVD

Spec_edm_normal = abs(Spec_edm) - abs(mean(Spec_edm));
Spec_classical_normal = abs(Spec_classical) - abs(mean(Spec_classical));
Spec_rap_normal = abs(Spec_rap) - abs(mean(Spec_rap));
Spec_train = [Spec_edm_normal(:, 1:70) Spec_classical_normal(:, 1:70) Spec_rap_normal(:, 1:70)]; 
Spec_test = [Spec_edm_normal(:, 71:90) Spec_classical_normal(:, 71:90) Spec_rap_normal(:, 71:90)]; 

[U,S,V] = svd(Spec_train, 'econ');

figure(fig); fig = fig+1;
subplot(2,1,1) 
plot(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80]) 
subplot(2,1,2) 
semilogy(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80])

%% Classification

figure(fig); fig=fig+1;

plot3(V(1:70, 1), V(1:70, 2), V(1:70, 3), 'ko')
hold on
plot3(V(71:140, 1), V(71:140, 2), V(71:140, 3), 'ro')
hold on
plot3(V(141:210, 1), V(141:210, 2), V(141:210, 3), 'go')
legend("EDM", "Classical", "Rap")
title("Projection onto Principle Components")
xlabel("Component 1"); ylabel("Component 2"); zlabel("Component 3");

figure(fig); fig=fig+1;
train1 = ((U(:, 1:3))'*Spec_train)';
label1 = [1*ones(70, 1); 2*ones(70,1); 3*ones(70,1)];
test1 = ((U(:, 1:3))'*Spec_test)';
class3 = classify(test1, train1, label1);
bar(class3);
title("Song Classification")
xlabel("Song Sample")
ylabel("Classification")

truth = [1*ones(20, 1); 2*ones(20,1); 3*ones(20,1)];

num_correct_case3 = 0;

for i = 1:length(class3)
    if class3(i) == truth(i)
        num_correct_case3 = num_correct_case3 + 1;
    end
end

percent_correct_3 = (num_correct_case3/length(class3))*100;
