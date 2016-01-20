using GPexact

using DataFrames

data=DataFrames.readtable("Folds5x2_pp.csv", header = true);
data = convert(Array,data);
data = data[1:1000,:];
N=size(data,1);
D=4;
Ntrain=500;
#N=size(data,1);
#Ntrain=5000;
Xtrain = data[1:Ntrain,1:D];
ytrain = reshape(data[1:Ntrain,D+1],Ntrain,1);
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

@everywhere length_scale=2.57;
@everywhere sigma=3.11;
@everywhere sigma_RBF=0.65;

f =  GPexact.SECov(length_scale,sigma_RBF);
gp = GPexact.GP(0,f,size(Xtrain,2));

tic();yfittrain = GPpost(gp,Xtrain,ytrain,Xtrain,sigma);timer_train = toc();
RMSEtrain = ytrainStd* (norm(ytrain-yfittrain)/sqrt(Ntrain));
tic();yfittest = GPpost(gp,Xtrain,ytrain,Xtest,sigma);timer_test = toc();
RMSEtest = ytrainStd* (norm(ytest-yfittest)/sqrt(N - Ntrain));

