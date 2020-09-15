clear;
file = dir('D:\add_noise\*.jpg');
for j = 1:size(file, 1)
    img = imread(['D:\add_noise\', file(j).name]);
    bw_img = im2bw(img);
    noise_img = bw_img;
    tic;
    for i = 1:800
        noise_img = random_noise(noise_img);
    end
    imwrite(noise_img, ['D:\result\', file(j).name])
    toc;
end
imshow(noise_img, []);