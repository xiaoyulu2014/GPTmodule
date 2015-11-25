using GPNotensor
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

seed = 17; nvec = round(linspace(50,5000,10)); 
restrain_n = Array(Float64,10,1); restest_n = Array(Float64,10,1); timer_n = Array(Float64,10,1)
for i in 1:length(nvec)
    n = convert(Int,nvec[i]);
    tic()
    phitrain = featureNotensor(Xtrain,n,length_scale,seed);
    phitest = featureNotensor(Xtest,n,length_scale,seed);
    theta = FullTheta(phitrain,ytrain,sigma);
    restrain_n[i] = ytrainStd*RMSE(theta,phitrain,ytrain);
    restest_n[i] = ytrainStd*RMSE(theta,phitest,ytest);
    timer_n[i] = toq();
    println(i)
end