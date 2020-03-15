%% AMATH 582 HW4

clc; clear all; close all;
fig = 1;

%% Create 4-D uint8 array with all cropped images for easy access

listing_cropped = dir('yalefaces_cropped/CroppedYale');
listing_uncropped = dir('yalefaces_uncropped/yalefaces');

cropped_subjects = extractfield(listing_cropped, 'name');
cropped_subjects = cropped_subjects(3:40);
size_cropped = size(cropped_subjects);
format = 'yalefaces_cropped/CroppedYale/%s';

for i = 1:size_cropped(2)
    str = sprintf(format,char(cropped_subjects(i)));
    subject_listing(:, :, i) = extractfield(dir(str), 'name'); 
end

subject_listing = subject_listing(:, 3:66, :);
size_subjects = size(subject_listing);
format = 'yalefaces_cropped/CroppedYale/%s/%s';

for i = 1:size_subjects(3)
    for j = 1:size_subjects(2)
        str = sprintf(format, char(cropped_subjects(i)), char(subject_listing(:, j, i)));
        cropped_images(:, :, j, i) = imread(str);
    end
end

%% SVD on first image of all subjects

figure(fig); fig = fig+1;
im_size = size(cropped_images);
im_length = im_size(1)*im_size(2);
for i = 1:size_subjects(3)
    subplot(5,8,i)
    imshow(cropped_images(:, :, 1, i))
    image_reshaped(i, :) = reshape(cropped_images(:, :, 1, i), [1, im_length]);
end
sgtitle("All Subjects in Baseline Lighting")

[U, S, V] = svd(double(image_reshaped), 'econ');

for j = 19
    figure(fig); fig = fig+1;
    im_rank1 = U(:, 1:j)*S(1:j, 1:j)*V(:, 1:j)';
    im_rank1s = reshape(im_rank1, [38, 192, 168]);
    for i = 1:size_subjects(3)
        subplot(5,8,i)
        imshow(squeeze(uint8(im_rank1s(i, :, :))))
    end
end

figure(fig); fig = fig+1;
for i = 1:8
    subplot(2, 4, i)
    Vt1 = reshape(V(:, i), 192, 168);
    Vt2=Vt1(192:-1:1,:); 
    pcolor(Vt2), colormap(hot)
    set(gca,'Xtick',[],'Ytick',[])
end

figure(fig); fig = fig+1;
subplot(2,1,1) 
plot(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80]) 
subplot(2,1,2) 
semilogy(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80])

%% SVD on first subject

for i = 1:size_subjects(2)
    subplot(8,8,i)
    imshow(cropped_images(:, :, i, 1))
    image_reshaped(i, :) = reshape(cropped_images(:, :, i, 1), [1, im_length]);
end
sgtitle("First Subject in Different Poses")
[U, S, V] = svd(double(image_reshaped), 'econ');

for j = 10
    figure(fig); fig = fig+1;
    im_rank1 = U(:, 1:j)*S(1:j, 1:j)*V(:, 1:j)';
    im_rank1s = reshape(im_rank1, [64, 192, 168]);
    for i = 1:size_subjects(2)
        subplot(8,8,i)
        imshow(uint8(squeeze(im_rank1s(i, :, :))))
    end
end

figure(fig); fig = fig+1;
for i = 1:8
    subplot(2, 4, i)
    Vt1 = reshape(V(:, i), 192, 168);
    Vt2=Vt1(192:-1:1,:); 
    pcolor(Vt2), colormap(hot)
    set(gca,'Xtick',[],'Ytick',[])
end

figure(fig); fig = fig+1;
subplot(2,1,1) 
plot(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80]) 
subplot(2,1,2) 
semilogy(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80])





