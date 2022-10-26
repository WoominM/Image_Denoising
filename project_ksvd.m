clear all;
sigma=25;lambda=30/sigma;C=1.15;T=(C*sigma)^6;
var=sigma^2/255^2;
I=imread('Lena.png');
% I=I(1:63,1:63);
I_noise=imnoise(I,'gaussian',0,var);
I_noise=double(I_noise);
I=double(I);
[M,N]=size(I_noise);
n=8;
MSE_noise=sum(sum((I-I_noise).^2))/(M*N);
PSNR_noise=10*log10(255^2/MSE_noise);
%% Image data to column data
Data=reshape(I_noise,M*N,1);
Patch=im2col(I_noise,[n,n],'sliding');
%% DCT Dictionary
Dm=8;Dn=16;
for i=0:Dm-1
    for j=0:Dn-1
        if i==0
            C=sqrt(1/Dm);
0 0 0 0     else 
            C=sqrt(2/Dm);
        end
        A(i+1,j+1)=C*cos((2*j+1)*pi*i/2/Dm);
    end
end
DCT=kron(A,A);
%% OMP sparse coding
[Dm,Dn]=size(DCT);   
for i=1:Dn       %% Normalization
    s=sum(DCT(:,i).^2);
    DCT(:,i)=DCT(:,i)./s;
end

[~,num]=size(Patch);
del=1e-3;
x=zeros(Dn,num);
for i=1:num      %% OMP
    Patch_in=Patch(:,i)-repmat(mean(Patch(:,i)),Dm,1);
    Ak=zeros(Dm,Dn);
    f=Patch_in;
    Rkf=f;
    for k=1:Dn
        [~,ind_k]=max(abs(DCT'*Rkf));
        index(k)=ind_k;
        Ak(:,k)=DCT(:,ind_k);
        x_ls=(Ak(:,1:k)'*Ak(:,1:k))\Ak(:,1:k)'*f;
        Rkf=f-Ak(:,1:k)*x_ls;
%         if max(abs(DCT'*Rkf))<del
%             break
%         end
        if sqrt(sum(Rkf.^2))<T
            break
        end
    end
    x(index(1:k),i)=x_ls;
    K(i)=length(x(x(:,i)>0));
    Patch_out(:,i)=DCT*x(:,i)+ones(Dm,1)*mean(Patch(:,i));
end
Patch_dn=Patch_out;
%% Image reconstruction
rltrl=zeros(M,N);
rltda=zeros(M,N);
[indr,indc]=ind2sub([M-n+1,N-n+1],1:num);
for i=1:(M-n+1)*(N-n+1)
    rltrl(indr(i):indr(i)+n-1,indc(i):indc(i)+n-1)=rltrl(indr(i):indr(i)+n-1,indc(i):indc(i)+n-1)+ones(n,n);
    rltda(indr(i):indr(i)+n-1,indc(i):indc(i)+n-1)=rltda(indr(i):indr(i)+n-1,indc(i):indc(i)+n-1)+reshape(Patch_dn(:,i),[8,8]);
end
X_DCT=(lambda*I_noise+rltda)./(lambda*ones(M,N)+rltrl);
%% Result of DCT
Atom_meanDCT=mean(K);
MSE_DCT=sum(sum((I-X_DCT).^2))/(M*N);
PSNR_DCT=10*log10(255^2/MSE_DCT);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% KSVD Algorithm
KSVD=DCT;          %Initialization
[Km,Kn]=size(KSVD);
J=20;
for repeat=1:J
    %% Normalization 
%     for i=1:Kn       
%         s=sum(KSVD(:,i).^2);
%         KSVD(:,i)=KSVD(:,i)./s;
%     end
    %% OMP
    x=zeros(Kn,num);
    for i=1:num
        Patch_in(:,i)=Patch(:,i)-repmat(mean(Patch(:,i)),Km,1);
        Ak=zeros(Km,Kn);
        f=Patch_in(:,i);
        Rkf=f;
        for k=1:Kn
            [~,ind_k]=max(abs(KSVD'*Rkf));
            index(k)=ind_k;
            Ak(:,k)=KSVD(:,ind_k);
            x_ls=(Ak(:,1:k)'*Ak(:,1:k))\Ak(:,1:k)'*f;
            Rkf=f-Ak(:,1:k)*x_ls;
            %         if max(abs(DCT'*Rkf))<del
            %             break
            %         end
            if sqrt(sum(Rkf.^2))<T
                break
            end
        end
        x(index(1:k),i)=x_ls;
    end
    %% KSVD
    Y=Patch_in;dx=0;   
    for k=1:Kn
        Ek=Y-KSVD*x+KSVD(:,k)*x(k,:);
        index=find(x(k,:)>0);
        if ~isempty(index)
            EkR=Ek(:,index);
            [U,S,V]=svd(EkR);
            KSVD(:,k)=U(:,1);
            x(k,index)=V(:,1)*S(1,1);
        end
    end
    for i=1:Kn
        K(i)=length(x(x(:,i)>0));
        Patch_out(:,i)=KSVD*x(:,i)+ones(Km,1)*mean(Patch(:,i));
    end
    K_KSVD(repeat)=mean(K);
    Patch_dn=Patch_out;
    %% Image reconstruction
    rltrl=zeros(M,N);
    rltda=zeros(M,N);
    [indr,indc]=ind2sub([M-n+1,N-n+1],1:num);
    for i=1:(M-n+1)*(N-n+1)
        rltrl(indr(i):indr(i)+n-1,indc(i):indc(i)+n-1)=rltrl(indr(i):indr(i)+n-1,indc(i):indc(i)+n-1)+ones(n,n);
        rltda(indr(i):indr(i)+n-1,indc(i):indc(i)+n-1)=rltda(indr(i):indr(i)+n-1,indc(i):indc(i)+n-1)+reshape(Patch_dn(:,i),[8,8]);
    end
    X_KSVD(:,:,repeat)=(lambda*I_noise+rltda)./(lambda*ones(M,N)+rltrl);
    MSE_KSVD=sum(sum((I-X_KSVD(:,:,repeat)).^2))/(M*N);
    PSNR_KSVD(repeat)=10*log10(255^2/MSE_KSVD);
end
Atom_meanKSVD=mean(K_KSVD);
%% Result
figure(1)
subplot(2,2,1)
imshow(uint8(I));
title('Original Image');
subplot(2,2,2)
imshow(uint8(I_noise));
title('Noise Image');
subplot(2,2,3)
imshow(uint8(X_DCT))
title('Denoising Image of DCT');
subplot(2,2,4)
imshow(uint8(X_KSVD(:,:,end)))
title('Denoising Image of KSVD');
suptitle('Noise Reduction');
figure(2)
plot(PSNR_KSVD); hold on
plot(repmat(PSNR_DCT,1,20));hold on
plot(repmat(PSNR_noise,1,20));hold on
legend('KSVD','DCT','Noise')
