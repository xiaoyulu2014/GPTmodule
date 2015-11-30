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
burnin=50
r = 20;
n = 50;
Q = 100;
seed = 17;
m = 500;
L = 10;
maxepoch = 10
I=samplenz(r,D,Q,seed);
phitrain=feature(Xtrain,n,length_scale,seed);
phitest=feature(Xtest,n,length_scale,seed);
epsw = 10.0^(-8);epsU = 10.0^(-6);

restrain_n=SharedArray(Float64,5,1);timertrain_n=SharedArray(Float64,5,1);timer_n = SharedArray(Float64,5,1)
restest_n=SharedArray(Float64,5,1);timertest_n=SharedArray(Float64,5,1);
nvec = round(linspace(r,120,5))

for i in 1:5
  n = convert(Int,nvec[i])
  seed = i; tic();
	I=samplenz(r,D,Q,seed);
 	phitrain=feature(Xtrain,n,length_scale,seed);
 	phitest=feature(Xtest,n,length_scale,seed);
  w_store,U_store=GPTinf.GPNHT_SGLDERM(phitrain,ytrain,sigma,I,r,Q,m,epsw,epsU,burnin,maxepoch,L);
  timer_n[i,1] = toq();
	tic();
	restrain_n[i,1] = ytrainStd*RMSE(w_store,U_store,I,phitrain,ytrain);
	timertrain_n[i,1] = toq();
	tic();
	restest_n[i,1] = ytrainStd*RMSE(w_store,U_store,I,phitest,ytest);
	timertest_n[i,1] = toq();
  println(n)
end


outfile=open("RMSENHTSGLD_n","a") #append to file
    println(outfile,"restrain_n=",restrain_n,";timertrain_n=",timertrain_n,";timer_n=",timer_n,
            ";restest_n=",restest_n,";timertest_n=",timertest_n);
    close(outfile)
