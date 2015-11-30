@everywhere cd("C:\\Users\\Xiaoyu Lu\\Dropbox\\GP\\GPTmodule")
@everywhere include("GPTinf.jl")
@everywhere using GPTinf
@everywhere using DataFrames

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
burnin=150;
r = 20;
#n = 150;
Q = 100;
seed = 17;
m = 500;
phitrain=GPTinf.feature(Xtrain,n,length_scale,seed,1);
phitest=GPTinf.feature(Xtest,n,length_scale,seed,1);
I=GPTinf.samplenz(r,D,Q,seed);
maxepoch = 50;

#tune the hyperparameters
epsw_vec = linspace(-8,-6,10); epsU_vec = linspace(-6,-4,10);
restest = SharedArray(Float64,10,10);restrain = SharedArray(Float64,10,10);
timer= SharedArray(Float64,10,10);timertrain= SharedArray(Float64,10,10);
timertest = SharedArray(Float64,10,10);

for i in 6:10
 for j in 1:5
    epsw = convert(Float64,10.0^epsw_vec[i]); epsU = convert(Float64,10.0^epsU_vec[j]);
    tic();w_store,U_store=GPTinf.GPTSGLD(phitrain,ytrain,sigma,I,r,Q,m,epsw,epsU,burnin,maxepoch);
    timer[i,j] = toc();
    tic();
    restest[i,j] = ytrainStd*GPTinf.RMSESGLD(w_store,U_store,I,phitest,ytest);
    timertrain = toc();
    tic();
    restrain[i,j] = ytrainStd*GPTinf.RMSESGLD(w_store,U_store,I,phitest,ytest);
    timertest = toc();
    println("j = ", j, "RMSE_test = ",restest[i,j]  );
end
  println("i = ", i);
end

