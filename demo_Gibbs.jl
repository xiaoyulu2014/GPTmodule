using GPTinf
using DataFrames
data=DataFrames.readtable("Folds5x2_pp.csv", header = true);
data = convert(Array,data);
N=size(data,1);
D=4;
Ntrain=5000;
length_scale=1.4332;
sigma=0.2299;
Xtrain = data[1:Ntrain,1:D];
ytrain = data[1:Ntrain,D+1];
XtrainMean=mean(Xtrain,1);
XtrainStd=zeros(1,D);
for i=1:D
  XtrainStd[1,i]=std(Xtrain[:,i]);
end
ytrainMean=mean(ytrain);
ytrainStd=std(ytrain);

Xtrain = datawhitening(Xtrain);
ytrain = datawhitening(ytrain);
Xtest = (data[Ntrain+1:end,1:D]-repmat(XtrainMean,N-Ntrain,1))./repmat(XtrainStd,N-Ntrain,1);
ytest = (data[Ntrain+1:end,D+1]-ytrainMean)/ytrainStd;
burnin=0;
r = 20;n = 150;Q = 100;seed = 17;I=samplenz(r,D,Q,seed);numiter = 100;
phitrain=GPTinf.feature(Xtrain,n,length_scale,seed,1);
phitest=GPTinf.feature(Xtest,n,length_scale,seed,1);
tic()
w_store,U_store=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter);
timer = toq();

RMSEtrain = ytrainStd*RMSE(w_store[:,end-5:end],U_store[:,:,:,end-5:end],I,phitrain,ytrain);
RMSEtest = ytrainStd*RMSE(w_store[:,end-5:end],U_store[:,:,:,end-5:end],I,phitest,ytest);
RMSEtrainvec = ytrainStd*GPTinf.RMSESGLDvec(w_store,U_store,I,phitrain,ytrain);
RMSEtestvec = ytrainStd*GPTinf.RMSESGLDvec(w_store,U_store,I,phitest,ytest);


using HDF5
c=h5open("GibbsRMSEvec.h5","w") do file
	write(file,"RMSEtest",RMSEtest);
	write(file,"RMSEtrain",RMSEtrain);
	write(file,"timer",timer);
	write(file,"RMSEtrainve",RMSEtrainvec);
	write(file,"RMSEtestvec",RMSEtestvec);
end

outfile=open("tmp","a") #append to file
    println(outfile,"RMSEtestvec=",RMSEtestvec,";RMSEtrainvec=",RMSEtrainvec,";timer=",timer,
            ";RMSEtest=",RMSEtest,";RMSEtrain=",RMSEtrain);
    close(outfile)
