clear all; 
clc; 

I = imread('20191010_133753.jpg');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Thresholding using Lab values
[bw, lab] = thresh_backgroundremoval_lab(I);

% hold on; subplot(4,3,1); imshow(bw); title('bw_lab');
% hold on; subplot(4,3,2); imshow(lab); title('lab');

% %Eliminate small unconnected pixels
% BW1ab = bwareaopen(bw, 1500); % removes objects comprised of < 500 pixels
% %Fill holes in the image
% BWdfill_lab = imfill(BW1ab,'holes');
% 
% BW_lab = BWdfill_lab;
% hold on; subplot(4,3,3); imshow(BW_lab); title('segmented using lab');

I = rgb2gray(lab);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%TEXTURE 
glcm = graycomatrix(I,'Offset',[2 0;0 2]);
GLCMTextureFeatures = graycoprops(glcm,{'contrast','correlation', 'energy', 'homogeneity'})
entropy_im = entropy(I)

imshow(I)

%Get the co occurence matrix (in Matlab called GLCM: Gray Level Co-Occurence Matrix)
glcm = graycomatrix(I, 'offset', [0 1], 'Symmetric', true);

%Haralick texture features
xFeatures = 1:14;
x = haralickTextureFeatures(glcm, 1:14);
HaralickTextureFeatures = x( xFeatures )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%