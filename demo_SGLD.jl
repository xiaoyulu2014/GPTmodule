@everywhere using DataFrames
@everywhere cd("C:\\Users\\Xiaoyu Lu\\Dropbox\\GP\\GPTmodule")
@everywhere include("GPTinf.jl")
@everywhere using GPTinf

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

Xtrain = GPTinf.datawhitening(Xtrain);
ytrain = GPTinf.datawhitening(ytrain);
Xtest = (data[Ntrain+1:end,1:D]-repmat(XtrainMean,N-Ntrain,1))./repmat(XtrainStd,N-Ntrain,1);
ytest = (data[Ntrain+1:end,D+1]-ytrainMean)/ytrainStd;
burnin=50;
r = 20;
n = 150;
Q = 100;
seed = 17;
m = 100;
phitrain=GPTinf.feature(Xtrain,n,length_scale,seed);
phitest=GPTinf.feature(Xtest,n,length_scale,seed);

epsw = 0.00001; epsU = 0.000001;maxepoch = 10;I=GPTinf.samplenz(r,D,Q,seed);
tic();w_store,U_store=GPTinf.GPTSGLD(phitrain,ytrain,sigma,I,r,Q,m,epsw,epsU,burnin,maxepoch);timer = toc()
tic();
restest = ytrainStd*GPTinf.RMSESGLD(w_store,U_store,I,phitest,ytest);
timertrain = toc();
