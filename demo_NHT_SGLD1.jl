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
burnin=10;
r = 20;
n = 50;
Q = 100;
seed = 17;
m = 500;
#epsw = 10.0^(-6); epsU = 10.0^(-8);I=samplenz(r,D,Q,seed)
epsw = 2e-6; epsU = 1e-8; I=samplenz(r,D,Q,seed)
r = 20;n = 150;Q = 200;seed = 17;m = 500; L = 10;
maxepoch = 10;I=samplenz(r,D,Q,seed);
phitrain=GPTinf.feature(Xtrain,n,length_scale,seed,sqrt(n/(Q^(1/D))));
phitest=GPTinf.feature(Xtest,n,length_scale,seed,sqrt(n/(Q^(1/D))));
tic()
w_store,U_store=GPTinf.GPNHT_SGLDERM(phitrain,ytrain,sigma,I,r,Q,m,epsw,epsU,burnin,maxepoch,L);
timer_learn = toc();
tic();
restest = ytrainStd*GPTinf.RMSESGLD(w_store,U_store,I,phitest,ytest)
timer_test = toc();



