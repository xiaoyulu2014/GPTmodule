@everywhere using DataFrames
@everywhere using GPTinf

### data processing
@everywhere function getdummy{R}(df::DataFrame, cname::Symbol, ::Type{R})
    darr = df[cname]
    vals = sort(levels(darr))[2:end]
    namedict = Dict(vals, 1:length(vals))   
    arr = zeros(R, length(darr), length(namedict))
    for i=1:length(darr)
        if haskey(namedict, darr[i])
            arr[i, namedict[darr[i]]] = 1
        end        
    end
    newdf = convert(DataFrame, arr)
    names!(newdf, [symbol("$(cname)_$k") for k in vals])
    return newdf
end

@everywhere function convertdummy{R}(df::DataFrame, cnames::Array{Symbol}, ::Type{R})
    # consider every variable from cnames as categorical
    # and convert them into set of dummy variables,
    # return new dataframe
    newdf = DataFrame()
    for cname in names(df)
        if !in(cname, cnames)
            newdf[cname] = df[cname]
        else
            dummydf = getdummy(df, cname, R)
            for dummyname in names(dummydf)
                newdf[dummyname] = dummydf[dummyname]
            end
        end
    end
    return newdf
end

@everywhere convertdummy(df::DataFrame, cnames::Array{Symbol}) = convertdummy(df, cnames, Int32)

##data clearing
@everywhere UserData = readdlm("/homes/xlu/Downloads/ml-100k/u.user", '|');
@everywhere MovieData = readdlm("/homes/xlu/Downloads/ml-100k/u.item",'|');
@everywhere Rating = readdlm("/homes/xlu/Downloads/ml-100k/u.data",Float64);
#UserData[:,end] = map(string,UserData[:,end]);
#convertdummy(convert(DataFrame,UserData),[:x5]);
@everywhere UserData = convertdummy(convert(DataFrame,UserData),[:x3,:x4])[:,1:end-1]
@everywhere MovieData = MovieData[:,[1,6:end]];


@everywhere User = UserData[Rating[:,1],2:end];
@everywhere User = convert(Array{Float64,2},User);
@everywhere Movie = MovieData[Rating[:,1],2:end];
@everywhere Movie = convert(Array{Float64,2},Movie);
@everywhere data = hcat(User,Movie);
@everywhere data[isnan(data)] = 0.0
@everywhere N,D = size(data);
@everywhere Ntrain = 500;
@everywhere Ntest = 500;
@everywhere Xtrain = data[1:Ntrain,:];
@everywhere ytrain = Rating[1:Ntrain,3];
@everywhere Xtest = data[Ntrain+1:Ntrain+Ntest,:];
@everywhere ytest = Rating[Ntrain+1:Ntrain+Ntest,3];
@everywhere XtrainMean=mean(Xtrain,1); 
@everywhere XtrainStd=zeros(1,D);
@everywhere for i=1:D
	    XtrainStd[1,i]=std(Xtrain[:,i]);
	    end
@everywhere ytrainMean=mean(ytrain);
@everywhere ytrainStd=std(ytrain);
#data whitening
@everywhere Xtrain = datawhitening(Xtrain);
@everywhere ytrain=datawhitening(ytrain);
@everywhere Xtest = (Xtest-repmat(XtrainMean,Ntest,1))./repmat(XtrainStd,Ntest,1);
@everywhere ytest = (ytest-ytrainMean)/ytrainStd;
@everywhere seed = 10;
## learning hyperparameters
@everywhere n = 150; 
#GPNT_hyperparameters(Xtrain,ytrain,n,0.5,0.5,0.5,seed)
@everywhere length_scale=0.4909;
@everywhere sigma_RBF=0.1773;
@everywhere sigma=0.9859;

@everywhere burnin=10;
@everywhere numiter=50;
@everywhere Q=100;   #200
@everywhere scale=1;
@everywhere M = 50
@everywhere phitrain=GPTinf.hash_feature(Xtrain,n,M,0.2,0.5,seed,21,19);
@everywhere phitrain[isnan(phitrain)] = 0.0;
@everywhere phitest=GPTinf.hash_feature(Xtest,n,M,0.2,0.5,seed,21,19);
@everywhere phitest[isnan(phitest)] = 0.0;
@everywhere r = 30


#SGLD
@everywhere using Distributions
@everywhere D = 2;
@everywhere length_scale = exp(randn(1))[1];  @everywhere sigma_RBF = exp(randn(1))[1]; @everywhere tau = rand(Gamma(1,1))[1];  @everywhere signal_var = 1/tau;
@everywhere scale = sqrt(n/(Q^(1/D)));
#@everywhere I = rand(DiscreteUniform(1, r),Q,D)
@everywhere I=samplenz(r,D,Q,seed); 
@everywhere m = 500;
@everywhere maxepoch = 10;

@everywhere using Iterators
@everywhere t=Iterators.product(2:6,3:8)
@everywhere myt=Array(Any,30);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
        it+=1;
        end

@sync @parallel for Tuple in myt
    i,j=Tuple;
    epsw=float(string("1e-",i)); epsU=float(string("1e-",j));
    w_store,U_store=GPTinf.GPT_SGLDERM(phitrain, ytrain,signal_var, I, r, Q, m, epsw, epsU, burnin, maxepoch);
    testRMSE=Array(Float64,maxepoch)
    numbatches=int(ceil(Ntrain/m))
    for epoch=1:maxepoch
        testpred=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitest)
        testRMSE[epoch]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
    end
    
println("r=",r,";minRMSE=",minimum(testRMSE),";minepoch=",indmin(testRMSE),";epsw=",epsw,";epsU=",epsU,";burnin=",burnin,";maxepoch=",maxepoch);
end



# fixed U
rvec = convert(Array{Int,1},round(linspace(10,50,5)));
@everywhere function func_w(r::Real)
	I=samplenz(r,D,Q,seed);
	w_store,U_store=GPT_w(phitrain,ytrain,sigma,I,r,Q);
	RMSEtest = ytrainStd*RMSE(w_store,U_store,I,phitest,ytest);
	return(RMSEtest)
end
RMSE_w = pmap(func_w,rvec)


@everywhere function func(r::Real)
	I=samplenz(r,D,Q,seed);
	w_store,U_store=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter)
	RMSEtest = ytrainStd*RMSE(w_store,U_store,I,phitest,ytest);
	return(RMSEtest)
end
RMSE_wU = pmap(func,rvec)




#GP exact using learnt hyperparameters using GPkit, memory issue
include("/homes/xlu/Downloads/GPkit/src/GPKit.jl")
using GPkit
cov=CovSEiso(length_scale,sigma_RBF);    # use ; to suppress return being printed in repl
lik=LikGauss(sn2);
gp=GPmodel(InfExact(), cov, lik, MeanZero(), Xtrain, ytrain);
# now we can run the model - you'll get deprecation warnings in julia 0.4,
# use julia --depwarn=no when starting the repl to supporess them
(post,nlZ,dnlZ)=inference(gp, with_dnlz=false); # posterior and derivatives if requested
(ymu,ys2,fmu,fs2,lp)=prediction(gp, post, Xtrain);  # predictions and variances (see gpml)

#= using GPexact
f =  GPexact.SECov(length_scale,sigma_RBF);
gp = GPexact.GP(0,f,size(Xtrain,2));

tic();yfittrain = GPpost(gp,Xtrain,ytrain,Xtrain,sigma);timer_train = toc();
RMSEtrain = ytrainStd* (norm(ytrain-yfittrain)/sqrt(Ntrain));
tic();yfittest = GPpost(gp,Xtrain,ytrain,Xtest,sigma);timer_test = toc();
RMSEtest = ytrainStd* (norm(ytest-yfittest)/sqrt(N - Ntrain));
=#

