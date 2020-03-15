%% AMATH 582 HW4

clc; clear all; close all;
fig = 1;

%% Create 4-D uint8 array with all cropped images for easy access

listing_cropped = dir('yalefaces_cropped/CroppedYale');

uncropped_subjects = extractfield(listing_cropped, 'name');
uncropped_subjects = uncropped_subjects(3:40);
size_cropped = size(uncropped_subjects);
format = 'yalefaces_cropped/CroppedYale/%s';

for i = 1:size_cropped(2)
    str = sprintf(format,char(uncropped_subjects(i)));
    subject_listing(:, :, i) = extractfield(dir(str), 'name'); 
end

subject_listing = subject_listing(:, 3:66, :);
size_subjects = size(subject_listing);
format = 'yalefaces_cropped/CroppedYale/%s/%s';

for i = 1:size_subjects(3)
    for j = 1:size_subjects(2)
        str = sprintf(format, char(uncropped_subjects(i)), char(subject_listing(:, j, i)));
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

for j = 30
    figure(fig); fig = fig+1;
    im_rank1 = U(:, 1:j)*S(1:j, 1:j)*V(:, 1:j)';
    im_rank1s = reshape(im_rank1, [38, 192, 168]);
    for i = 1:size_subjects(3)
        subplot(5,8,i)
        imshow(squeeze(uint8(im_rank1s(i, :, :))))
    end
end

figure(fig); fig = fig+1;
for i = 1:6
    subplot(2, 3, i)
    Vt1 = reshape(V(:, i), 192, 168);
    Vt2=Vt1(192:-1:1,:); 
    pcolor(Vt2), shading interp, colormap hot
    title(sprintf("Component %i", i))
    set(gca,'Xtick',[],'Ytick',[])
end

figure(fig); fig = fig+1;
semilogy(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80])
title("Singular Values of Different People")
ylabel("Singular Value")
xlabel("Component")

%% SVD on first subject

figure(fig); fig = fig+1;
for i = 1:size_subjects(2)
    subplot(8,8,i)
    imshow(cropped_images(:, :, i, 1))
    image_reshaped(i, :) = reshape(cropped_images(:, :, i, 1), [1, im_length]);
end
sgtitle("First Subject in Different Poses")
[U, S, V] = svd(double(image_reshaped), 'econ');

for j = 9
    figure(fig); fig = fig+1;
    im_rank1 = U(:, 1:j)*S(1:j, 1:j)*V(:, 1:j)';
    im_rank1s = reshape(im_rank1, [64, 192, 168]);
    for i = 1:size_subjects(2)
        subplot(8,8,i)
        imshow(uint8(squeeze(im_rank1s(i, :, :))))
    end
end

figure(fig); fig = fig+1;
for i = 1:6
    subplot(2, 3, i)
    Vt1 = reshape(V(:, i), 192, 168);
    Vt2=Vt1(192:-1:1,:); 
    pcolor(Vt2), shading interp, colormap hot
    set(gca,'Xtick',[],'Ytick',[])
    title(sprintf("Component %i", i))
end

figure(fig); fig = fig+1;
semilogy(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80])

title("Singular Values of Same Person")
ylabel("Singular Value")
xlabel("Component")

%% Uncropped faces

clc; clear all; close all;

%% Same person, different  poses

fig = 1;

listing_uncropped = dir('yalefaces_uncropped/yalefaces');
uncropped_subjects = extractfield(listing_uncropped, 'name');
uncropped_subjects = uncropped_subjects(3:167);
size_cropped = size(uncropped_subjects);
format = 'yalefaces_uncropped/yalefaces/%s';


for i = 1:165
        str = sprintf(format, char(uncropped_subjects(i)));
        uncropped_images(:, :, i) = imread(str);
        uncropped_images_shape(i, :) = reshape(imread(str), [1, 243*320]);
end

figure(fig); fig = fig+1;

for i = 1:11
    subplot(3,4,i)
    imshow(uncropped_images(:, :, i))
end

[U,S,V] = svd(double(uncropped_images_shape(1:11, :)), 'econ');

figure(fig); fig = fig+1;
for i = 1:11
    subplot(3, 4, i)
    Vt1 = reshape(V(:, i), 243, 320);
    Vt2=Vt1(243:-1:1,:); 
    pcolor(Vt2), shading interp, colormap hot
    title(sprintf("Component %i", i))
    set(gca,'Xtick',[],'Ytick',[])
end

figure(fig); fig = fig+1;
semilogy(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 11])
title("Subject in Different Poses")
ylabel("Singular Value")
xlabel("Component")

for j = 9
    figure(fig); fig = fig+1;
    im_rank1 = U(:, 1:j)*S(1:j, 1:j)*V(:, 1:j)';
    im_rank1s = reshape(im_rank1, [11, 243, 320]);
    for i = 1:11
        subplot(3,4,i)
        imshow(uint8(squeeze(im_rank1s(i, :, :))))
    end
end

%% Different people, same  poses

figure(fig); fig = fig+1;
for i = 1:15
    subplot(3,5,i)
    faces(:, :, i) = uncropped_images(:, :, 10+ 11*(i-1));
    imshow(faces(:, :, i))
end

faces_shape = reshape(faces, [243*320, 15]);
[U,S,V] = svd(double(faces_shape'), 'econ');

figure(fig); fig = fig+1;
for i = 1:15
    subplot(3, 5, i)
    Vt1 = reshape(V(:, i), 243, 320);
    Vt2=Vt1(243:-1:1,:); 
    pcolor(Vt2), shading interp, colormap hot
    title(sprintf("Component %i", i))
    set(gca,'Xtick',[],'Ytick',[])
end

figure(fig); fig = fig+1;
semilogy(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 15])
title("Subject in Different Poses")
ylabel("Singular Value")
xlabel("Component")

for j = 12
    figure(fig); fig = fig+1;
    im_rank1 = U(:, 1:j)*S(1:j, 1:j)*V(:, 1:j)';
    im_rank1s = reshape(im_rank1, [15, 243, 320]);
    for i = 1:15
        subplot(3,5,i)
        imshow(uint8(squeeze(im_rank1s(i, :, :))))
    end
end


