%%%      Maher Nadar    Advanced Topics in Computer Vision
%%                      Image stiching

clear all
close all
clc


run('vl_setup')% launching the SIFT package

% Reading the pair of images 
image_left=imread('mountains1.jpg'); image_right=imread('mountains2.jpg');
%image_left=imread('jungfrau1.jpg'); image_right=imread('jungfrau2.jpg');
%image_left=imread('beach1.jpg'); image_right=imread('beach2.jpg');


[r,c,b]=size(image_left);
image_right=imresize(image_right,[r,c]);% letting both images have the same size in order to match the resulting coordinates properly



%% SIFT
%applying SIFT on both images and diplaying 100 features in each of them
%(simpy for visualisation
%Left image SIFT
figure;
subplot(1,2,1)
imshow(image_left,[]);
hold on
[fa,da]=vl_sift(im2single(rgb2gray(image_left)));
rand_p = randperm(size(fa,2)) ; % displaying 100 features.
rand_features = rand_p(1:50) ;
h1 = vl_plotframe(fa(:,rand_features)) ;
h2 = vl_plotframe(fa(:,rand_features)) ;
set(h1,'color','k','linewidth',3) ;
set(h2,'color','y','linewidth',2) ;
title('Left image SIFT features');

%Right image SIFT
subplot(1,2,2)
imshow(image_right,[]);
hold on
[fb,db]=vl_sift(im2single(rgb2gray(image_right)));
rand_p = randperm(size(fb,2)) ; % displaying 100 features.
rand_features = rand_p(1:50) ;
h1 = vl_plotframe(fb(:,rand_features)) ;
h2 = vl_plotframe(fb(:,rand_features)) ;
set(h1,'color','k','linewidth',3) ;
set(h2,'color','y','linewidth',2) ;
title('Right image SIFT features');


[ind_match, distance]=vl_ubcmatch(da,db);  %estimating potential matches


[sorted, ind_sort]=sort(distance,'descend');% Sorting the potential matches 
ind_match=ind_match(:,ind_sort);

distance=distance(ind_sort);
match_left=fa(1:2,ind_match(1,:)); 
match_right=fb(1:2,ind_match(2,:));

%% RANSAC and HOMOGRAPHY

%choosing the parameters
e=0.4; p=0.99; s=6;  
n=ceil((log(1-p))/(log(1-(1-e)^s))); % Number of iterations

%initialising some values
inlier_count_max=0;
s = size(match_left, 2);
inlier_count = 0;
H_best=zeros(3,3);

% performing RANSAC iterations 
for i=1:n    
    
    [~,index]=datasample(match_left(1,:),4);
    l=match_left(:,index)'; % picking 4 random points from left image features to perform H matrix
    r=match_right(:,index)'; % picking the corresponding 4 matches from the right image 
       
    l1=l(1,:); l2=l(2,:); l3=l(3,:); l4=l(4,:); 
    r1=r(1,:); r2=r(2,:); r3=r(3,:); r4=r(4,:);
    
    % Direct Linear Transformation
M=[r1(1) r1(2)  1    0     0     0  -r1(1)*l1(1) -r1(2)*l1(1) -l1(1);
    0     0     0   r1(1) r1(2)  1  -r1(1)*l1(2) -r1(2)*l1(2) -l1(2);
   r2(1) r2(2)  1    0     0     0  -r2(1)*l2(1) -r2(2)*l2(1) -l2(1);
    0     0     0   r2(1) r2(2)  1  -r2(1)*l2(2) -r2(2)*l2(2) -l2(2);
   r3(1) r3(2)  1     0     0    0  -r3(1)*l3(1) -r3(2)*l3(1) -l3(1);
    0     0     0   r3(1) r3(2)  1  -r3(1)*l3(2) -r3(2)*l3(2) -l3(2);
   r4(1) r4(2)  1     0     0    0  -r4(1)*l4(1) -r4(2)*l4(1) -l4(1);
    0     0     0   r4(1) r4(2)  1  -r4(1)*l4(2) -r4(2)*l4(2) -l4(2)];


[~,~,V]=svd(M); % Solve system using singular value decomposition              
x=V(:,end); %retreiving eigenvector corresponding to smallest eigenvalue

% Homography matrix
H=[x(1:3,1)';x(4:6,1)';x(7:9,1)'];
H=H/H(end);

% calculating the error in order to keep the H matrix correspoding to
% biggest number of inliers

    points1 = H*[match_right;ones(1,s)];
    
    % Normalizing
    points1=[points1(1,:)./points1(3,:); points1(2,:)./points1(3,:); ones(1,s)];
             
    error=(points1-[match_left;ones(1,s)]);
                                                   
    error = sqrt(sum(error.^2,1));   % Computing the eucledian distance
    inliers = error < 10;      
    inlier_count = size(find(inliers),2); %calculating number of inliers
    
    if inlier_count > inlier_count_max     % Save the H with greatest number of inliers
        inlier_count_max=inlier_count;
        ind = find(inliers);
        H_best=H;
    end
end


% Plotting potential correspondences
figure
 showMatchedFeatures(image_left,image_right,match_left',match_right','montage')
 title('Set of potential matches');
legend('matchedPts_left','matchedPts_right');

% Plotting inlier correspondences
figure
 showMatchedFeatures(image_left,image_right,match_left(:,ind)',match_right(:,ind)','montage')
 title('Set of inliers');
legend('matchedPts_left','matchedPts_right');

%% Final Mosaic
% Compute mosaic composition using the homography H
[im_out]=make_mosaic(image_left,image_right,H_best);
figure;
imshow(im_out,[]);
title('Final Mosaic')


