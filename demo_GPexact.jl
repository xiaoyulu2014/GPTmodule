using GPexact

using DataFrames
cd("C:/Users/Xiaoyu lu/Dropbox/GP/GPTmodule")
data=DataFrames.readtable("Folds5x2_pp.csv", header = true);
data = convert(Array,data);
data = data[1:1000,:];
N=size(data,1);
D=4;
Ntrain=500;
#N=size(data,1);
#Ntrain=5000;
Xtrain = data[1:Ntrain,1:D];
ytrain = data[1:Ntrain,D+1];
XtrainMean=mean(Xtrain,1);
XtrainStd=zeros(1,D);
for i=1:D
  XtrainStd[1,i]=std(Xtrain[:,i]);
end
ytrainMean=mean(ytrain);
ytrainStd=std(ytrain);
# centre and normalise data X so that each col has sd=1

Xtrain = datawhitening(Xtrain);
ytrain = datawhitening(ytrain);
Xtest = (data[Ntrain+1:end,1:D]-repmat(XtrainMean,N-Ntrain,1))./repmat(XtrainStd,N-Ntrain,1);
ytest = (data[Ntrain+1:end,D+1]-ytrainMean)/ytrainStd;

length_scale=2.435;sigma=0.253;
f =  GPexact.SECov(length_scale,sigmaRBF);
gp = GPexact.GP(0,f,size(Xtrain,2));

#training RMSE = 3.846 ; test RMSE = 4.006; timer = 519.021
tic();yfittrain = GPpost(gp,Xtrain,ytrain,Xtrain,sigma);timer_train = toc();
RMSEtrain = ytrainStd* (norm(ytrain-yfittrain)/sqrt(Ntrain));
tic();yfittest = GPpost(gp,Xtrain,ytrain,Xtest,sigma);timer_test = toc();
RMSEtest = ytrainStd* (norm(ytest-yfittest)/sqrt(N - Ntrain));
