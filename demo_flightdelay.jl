using DataFrames
data_original=DataFrames.readtable("/homes/xlu/Downloads/FlightDelays2008.csv", header = true);
data = data_original[:,[:Month,:DayofMonth,:DayOfWeek,:CRSDepTime,:CRSArrTime,:AirTime,:Distance,:ArrDelay]]
data = convert(Array,complete_cases!(data));
subindex = randperm(size(data,1));
data = float64(data[subindex,:]);


N = 10000;
data = data[1:N,:];

D=7;
Ntrain=5000;
length_scale=0.1; #0.3
sigma=0.2;
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
r = 20;n = 100;Q = 200;seed = 17;I=samplenz(r,D,Q,seed);numiter = 10;
phitrain=GPTinf.feature(Xtrain,n,length_scale,seed,1);
phitest=GPTinf.feature(Xtest,n,length_scale,seed,1);
tic()
w_store,U_store=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter);
timer = toq();

RMSEtrain = ytrainStd*RMSE(w_store[:,end-5:end],U_store[:,:,:,end-5:end],I,phitrain,ytrain);
RMSEtest = ytrainStd*RMSE(w_store[:,end-5:end],U_store[:,:,:,end-5:end],I,phitest,ytest);
RMSEtrainvec = ytrainStd*GPTinf.RMSESGLDvec(w_store,U_store,I,phitrain,ytrain);
RMSEtestvec = ytrainStd*GPTinf.RMSESGLDvec(w_store,U_store,I,phitrain,ytest);
function yhat(w_store::Array,U_store::Array,I::Array,phitest::Array)
Ntest=size(phitest,1);
    T=size(w_store,2);

    meanfhat= @parallel (+) for i=1:T
        pred(w_store[:,i],U_store[:,:,:,i],I,phitest);
    end
    meanfhat=meanfhat/T;
    return meanfhat;
end

yfit = yhat(w_store,U_store,I,phitrain);




