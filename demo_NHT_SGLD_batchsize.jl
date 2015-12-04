@everywhere using GPTinf
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
r = 20;n = 150;Q = 200;seed = 17; L = 10;
maxepoch = 10;I=samplenz(r,D,Q,seed);
phitrain=GPTinf.feature(Xtrain,n,length_scale,seed,sqrt(n/(Q^(1/D))));
phitest=GPTinf.feature(Xtest,n,length_scale,seed,sqrt(n/(Q^(1/D))));
mvec = [50,500,5000];

timer_learn = SharedArray(Float64,3,1);
timer_test = SharedArray(Float64,3,1);
RMSEtrain = SharedArray(Float64,3,1);
RMSEtest = SharedArray(Float64,3,1);
RMSEtrainvec = SharedArray(Float64,1000,3);
RMSEtestvec = SharedArray(Float64,1000,3);

@parallel for i in 1:3
  m = mvec[i];
  tic()
  w_store,U_store=GPTinf.GPNHT_SGLDERM(phitrain,ytrain,sigma,I,r,Q,m,epsw,epsU,burnin,maxepoch,L,1,1);
  timer_learn[i,1] = toc();
  RMSEtrain[i,1] = ytrainStd*RMSE(w_store[:,end-50:end],U_store[:,:,:,end-50:end],I,phitrain,ytrain);
  tic();
  RMSEtest[i,1] = ytrainStd*RMSE(w_store[:,end-50:end],U_store[:,:,:,end-50:end],I,phitest,ytest);
  timer_test[i,1] = toc();
  tmp = ytrainStd*GPTinf.RMSESGLDvec(w_store,U_store,I,phitrain,ytrain);
  RMSEtrainvec[1:length(tmp),i] = tmp;
  tmp = ytrainStd*GPTinf.RMSESGLDvec(w_store,U_store,I,phitest,ytest);
  RMSEtestvec[1:length(tmp),i] = tmp;
  println("RMSEtest = ", RMSEtest )
end

#=
using HDF5
c=h5open("res_batchsize.h5","w") do file
	write(file,"RMSEtest",RMSEtest);
	write(file,"RMSEtrain",RMSEtrain);
	write(file,"timer_learn",timer_learn);
	write(file,"timer_test",timer_test);
	write(file,"RMSEtrainve",RMSEtrainvec);
	write(file,"RMSEtestvec",RMSEtestvec);
end=#
