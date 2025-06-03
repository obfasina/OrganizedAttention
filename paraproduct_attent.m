
% Load Matrix, set initial parameters
data = load('/home/of56/Documents/Yale/Coifman/Interpretability_transformer/holdermat_attn/organmat_lzo.mat');
lyrholder = data.holdermat;

%%
Ny = size(lyrholder,1);
Nx = size(lyrholder,2);
onesmat = ones(Ny,Nx);
avgopone = zeros(Ny,Nx);
avgoptwo = zeros(Ny,Nx);
avgopthree = zeros(Ny,Nx);

% Set Parameters
j = 1;
jp = 1;
dx = 2^(-j)*Nx;
dy = 2^(-jp)*Ny;



%% Generate and save Softmax of attention matrices

%{

data = load('holder_attnmat_0.mat');
holdermat_lzro = data.holder_attnmat_layer0;

data = load('holder_attnmat_15.mat');
holdermat_lfift = data.holder_attnmat_layer15;

soft_holder_lzro = softmax(holdermat_lzro);
soft_holder_lfiftn = softmax(holdermat_lfift);


save(['/home/of56/Documents/Yale/Coifman/Interpretability_transformer/softmax_attnmat_0.mat'],'soft_holder_lzro')
save(['/home/of56/Documents/Yale/Coifman/Interpretability_transformer/softmax_attnmat_15.mat'],'soft_holder_lfiftn')


%% Example of averaging operator computation

for k=1:dx:Nx
    for kp=1:dy:Ny

        xstart = k;
        xend  = xstart + (dx - 1);
        ystart = kp;
        yend = ystart + (dy - 1);

        avgopone(ystart:yend,xstart:xend) = mean(mean(holdermat(ystart:yend,xstart:xend))).*onesmat(ystart:yend,xstart:xend);

    end
end 




%% Test functions

jinp = 1;
jinp_p = 1; 
avgopone_fcn = compute_avgop(holdermat,jinp,jinp_p);

figure();
imagesc(avgopone_fcn)

disp(sum(sum(avgopone - avgopone_fcn)))


%}



%% Compute Approximation

tknidx = 3;
hdindx = 5;
holdermat = lyrholder(:,:,tknidx,hdindx);
soft_holder = softmax(holdermat);

Nxmat = size(holdermat,2);
Nymat = size(holdermat,1);
nxscl = log2(Nxmat) - 1;
nyscl = log2(Nymat) - 1;

Tenswave = 0;
Xwave = 0;
Ywave = 0;
totalapprox = 0;
firstapprox = 0;
scndapprox = 0;

for jx=1:nxscl
    for jy=1:nyscl

        
        avgop_jxpjyp = compute_avgop(holdermat,jx + 1,jy + 1);
        avgop_jxjyp = compute_avgop(holdermat,jx,jy + 1);
        avgop_jxpjy = compute_avgop(holdermat,jx + 1,jy);
        avgop_jxjy = compute_avgop(holdermat,jx,jy);

        Tenswave = Tenswave + (avgop_jxpjyp - avgop_jxjyp - avgop_jxpjy + avgop_jxjy);
        Xwave = Xwave + (avgop_jxpjy - avgop_jxjy);
        Ywave = Ywave + (avgop_jxjyp - avgop_jxjy);

        firstord = fderiv(avgop_jxjy);
        scndord = sderiv(avgop_jxjy);
        curapprox = (firstord.*Tenswave) + (scndord.*Xwave.*Ywave);

        firstapprox = firstapprox + (firstord.*Tenswave);
        scndapprox = scndapprox + (scndord.*Xwave.*Ywave);
        totalapprox = totalapprox + curapprox;


    end
end


%% Compute Approximation and Perform Visualization

% Specify Token
tknidx = 3;
storedecomps_tknthree=struct();

% Specify Layer Index
nheads=8;
attnheadlyr = [0,1,2,3,4,5,6,10,11,13,14,15];

for k=1:length(attnheadlyr)
    
    lyrindx = attnheadlyr(k);
    fload=strcat('/home/of56/Documents/Yale/Coifman/Interpretability_transformer/holdermat_attn/organmat_',num2str(lyrindx),'.mat');
    data = load(fload);
    lyrholder=data.holdermat;
    
    for j=1:nheads
    
        holder = lyrholder(:,:,tknidx,j);
        softholder = softmax(holder);
        decompmat = zeros(4,size(holder,1),size(holder,2));
        [fapprx,sapprx,tapprx,resapprx] = computeapprox(holder,softholder);
    
    
        decompmat(1,:,:) = holder;
        decompmat(2,:,:) = softholder;
        decompmat(3,:,:) = fapprx;
        decompmat(4,:,:) = resapprx;
        
        
        fieldname = strcat('l',num2str(lyrindx),'_h',num2str(j));
        storedecomps_tknthree.(fieldname) = decompmat;
    
    end


end 

fsave=strcat('/home/of56/Documents/Yale/Coifman/Interpretability_transformer/attenhead_decomp/tkn_idx',num2str(tknidx) ,'.mat');
save(fsave, 'storedecomps_tknthree');



%% Visualization

figure();
subplot(2,2,1)
imagesc(holder);colorbar;
title("Holder Matrix")

subplot(2,2,2)
imagesc(softholder);colorbar;
title("Softmax of Holder matrix")

subplot(2,2,3)
imagesc(fapprx);colorbar;
title("Total Approximation")

subplot(2,2,4)
imagesc(resapprx);colorbar;
title("Residual")



%% Function Definitions





function fordout = fderiv(matrix)
fordout = exp(matrix);
end



function sordout = sderiv(matrix)
sordout = exp(matrix);
end


function [firstapprox, scndapprox, totalapprox, resid] = computeapprox(matrix,nonlinmat)


Nxmat = size(matrix,2);
Nymat = size(matrix,1);
nxscl = log2(Nxmat) - 1;
nyscl = log2(Nymat) - 1;

Tenswave = 0;
Xwave = 0;
Ywave = 0;
totalapprox = 0;
firstapprox = 0;
scndapprox = 0;



for jx=3:nxscl
    for jy=3:nyscl

        
        avgop_jxpjyp = compute_avgop(matrix,jx + 1,jy + 1);
        avgop_jxjyp = compute_avgop(matrix,jx,jy + 1);
        avgop_jxpjy = compute_avgop(matrix,jx + 1,jy);
        avgop_jxjy = compute_avgop(matrix,jx,jy);

        Tenswave = Tenswave + (avgop_jxpjyp - avgop_jxjyp - avgop_jxpjy + avgop_jxjy);
        Xwave = Xwave + (avgop_jxpjy - avgop_jxjy);
        Ywave = Ywave + (avgop_jxjyp - avgop_jxjy);

        firstord = fderiv(avgop_jxjy);
        scndord = sderiv(avgop_jxjy);
        curapprox = (firstord.*Tenswave) + (scndord.*Xwave.*Ywave);

        firstapprox = firstapprox + (firstord.*Tenswave);
        scndapprox = scndapprox + (scndord.*Xwave.*Ywave);
        totalapprox = totalapprox + curapprox;


    end
end

resid = abs(totalapprox - nonlinmat);

end




function sftout = softmax(inpmat)

nrows=size(inpmat,1);
ncols=size(inpmat,2);

emat = exp(inpmat);
rowsum = sum(emat,2);

sftout = zeros(nrows,ncols);
for i=1:size(inpmat,1)
    sftout(i,:) = emat(i,:)/rowsum(i);
end

end



% PROCEDURE for computing averaging operator across grid at multiple scales
function avgop = compute_avgop(inpmat,j,jp)


% Set Parameters
Ny = size(inpmat,1);
Nx = size(inpmat,2);
dx = 2^(-j)*Nx;
dy = 2^(-jp)*Ny;
onesmat = ones(Ny,Nx);

for k=1:dx:Nx
    for kp=1:dy:Ny

        xstart = k;
        xend  = xstart + (dx - 1);
        ystart = kp;
        yend = ystart + (dy - 1);

        avgop(ystart:yend,xstart:xend) = mean(mean(inpmat(ystart:yend,xstart:xend))).*onesmat(ystart:yend,xstart:xend);

    end
end 

end



