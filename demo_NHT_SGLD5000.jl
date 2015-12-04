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
burnin=0;epsw = 2e-6; epsU = 1e-8;
r = 20;n = 150;Q = 200;seed = 17;m = 5000; L = 10;
maxepoch = 10;I=samplenz(r,D,Q,seed);
phitrain=GPTinf.feature(Xtrain,n,length_scale,seed,sqrt(n/(Q^(1/D))));
phitest=GPTinf.feature(Xtest,n,length_scale,seed,sqrt(n/(Q^(1/D))));
tic()
w_store,U_store=GPTinf.GPNHT_SGLDERM(phitrain,ytrain,sigma,I,r,Q,m,epsw,epsU,burnin,maxepoch,L,1,1);
timer_learn = toc();
RMSEtrain = ytrainStd*RMSE(w_store[:,end-5:end],U_store[:,:,:,end-5:end],I,phitrain,ytrain);
tic();
RMSEtest = ytrainStd*RMSE(w_store[:,end-5:end],U_store[:,:,:,end-5:end],I,phitest,ytest);
timer_test = toc();
RMSEtrainvec = ytrainStd*GPTinf.RMSESGLDvec(w_store,U_store,I,phitrain,ytrain);
RMSEtestvec = ytrainStd*GPTinf.RMSESGLDvec(w_store,U_store,I,phitest,ytest);

###traceplot####

using HDF5
c=h5open("res_batchsize5000.h5","w") do file
	write(file,"RMSEtest5000",RMSEtest);
	write(file,"RMSEtrain5000",RMSEtrain);
	write(file,"timer_learn5000",timer_learn);
	write(file,"timer_test5000",timer_test);
	write(file,"RMSEtrainve5000",RMSEtrainvec);
	write(file,"RMSEtestvec5000",RMSEtestvec);
end
#=
figure()
subplot(121)
plot(collect(w_store[1,:]));plot(collect(w_store[10,:]));plot(collect(w_store[100,:]));
title("NHT_SGLD, traceplots of w, \n n = 150, Q = 200, r = 20, batchsize = 5000")
subplot(122)
plot(collect(U_store[1,1,1,:]));plot(collect(U_store[2,2,2,:]));plot(collect(U_store[3,3,3,:]));
title("NHT_SGLD, traceplots of U, \n n = 150, Q = 200, r = 20, batchsize = 5000")
savefig("NHT_SGLD_m5000")=#

