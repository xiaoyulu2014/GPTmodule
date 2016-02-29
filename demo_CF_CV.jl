@everywhere using DataFrames
@everywhere using GPTinf
using Iterators
### data processing
@everywhere function getdummy{R}(df::DataFrame, cname::Symbol, ::Type{R})
    darr = df[cname]
    vals = sort(levels(darr))#[2:end]
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

@everywhere function bin_age(age::Array)
	q=quantile(age,[0.2,0.4,0.6,0.8,1.0])
	indmin(q.<UserData[30,2])
        map(x->indmin(q.<x),age)
end

##data clearing
@everywhere UserData = readdlm("/homes/xlu/Downloads/ml-100k/u.user", '|');
@everywhere MovieData = readdlm("/homes/xlu/Downloads/ml-100k/u.item",'|');
@everywhere Rating = readdlm("/homes/xlu/Downloads/ml-100k/u.data",Float64);

@everywhere Ntrain = 60000;
@everywhere Ntest = 20000;
@everywhere UserData[:,2] = bin_age(UserData[:,2])
@everywhere UserData = convertdummy(convert(DataFrame,UserData),[:x2,:x3,:x4])[:,1:end-1];
@everywhere MovieData = MovieData[:,[1,6:end]];
@everywhere UserData = convert(Array{Float64,2},UserData)[:,2:end];
@everywhere MovieData = convert(Array{Float64,2},MovieData)[:,3:end]; 

@everywhere ytrain = Rating[1:Ntrain,3];
@everywhere ytest = Rating[Ntrain+1:Ntrain+Ntest,3];
@everywhere ytrainMean=mean(ytrain);
@everywhere ytrainStd=std(ytrain);
@everywhere ytrain=datawhitening(ytrain);
@everywhere ytest = (ytest-ytrainMean)/ytrainStd;
@everywhere n = 500; 
@everywhere M = 5;
@everywhere burnin=50;
@everywhere r = 8
@everywhere using Distributions
@everywhere param_seed=17;
@everywhere m = 100;
@everywhere maxepoch = 100;

1.0,0.2,0.2,0.001,0.05,1.0e-5,1.0e-7


@everywhere avec=[0.1,1,2];
@everywhere b1vec=[0.0];
@everywhere b2vec=[0.0];
@everywhere signal_varvec=[0.001,0.01];
@everywhere var_uvec=[0.01,0.05];
@everywhere epswvec=[0.0001,0.00001];
@everywhere epsUvec=[0.000001,0.0000001];
@everywhere numbatches=int(ceil(maximum(Ntrain)/m));
@everywhere myn=3*2*2*2*2
@everywhere t=Iterators.product(avec,b1vec,b2vec,signal_varvec,var_uvec,epswvec,epsUvec)
@everywhere myt=Array(Any,myn);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
        it+=1;
        end
##cross validation

testRMSE = SharedArray(Float64,maxepoch-burnin+1,myn); trainRMSE= SharedArray(Float64,maxepoch-burnin+1,myn);timer=SharedArray(Float64,myn);
@sync @parallel for k=1:myn
        a,b1,b2,signal_var,var_u,epsw,epsU=myt[k]
	phiUser = GPTinf.hash_feature(UserData,n,M,a,b1);
	phiMovie = GPTinf.hash_feature(MovieData,n,M,1,b2);
	phiUtrain=Array(Float64,size(phiUser,1),Ntrain);phiVtrain=Array(Float64,size(phiMovie,1),Ntrain);
	phiUtest=Array(Float64,size(phiUser,1),Ntest);phiVtest=Array(Float64,size(phiMovie,1),Ntest);
	for i=1:Ntrain
	    phiUtrain[:,i]=phiUser[:,Rating[i,1]]
	    phiVtrain[:,i]=phiMovie[:,Rating[i,2]]
	end
	 for i=1:Ntest
		phiUtest[:,i]=phiUser[:,Rating[Ntrain+i,1]]
		phiVtest[:,i]=phiMovie[:,Rating[Ntrain+i,2]]
	 end
	tic(); 
	w_store,U_store,V_store=GPTinf.GPT_CF(phiUtrain, phiVtrain, ytrain,signal_var, var_u, r, m, epsw, epsU, maxepoch,param_seed,true,false);
	numbatches=int(ceil(Ntrain/m));trainpred=zeros(Ntrain);testpred=zeros(Ntest)
	#make predictions for every epoch
	for epoch=burnin:maxepoch
		trainpred = (trainpred + GPTinf.pred_CF(w_store[:,epoch],U_store[:,:,epoch],V_store[:,:,epoch],phiUtrain,phiVtrain))/2
		trainRMSE[epoch-burnin+1,k]=ytrainStd*norm(ytrain-trainpred)/sqrt(Ntrain)
		testpred = (testpred + GPTinf.pred_CF(w_store[:,epoch],U_store[:,:,epoch],V_store[:,:,epoch],phiUtest,phiVtest))/2
		testRMSE[epoch-burnin+1,k]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
	end
	timer[k] = toq(); 
	println("a=",a,";signal_var=",signal_var,";var_u=",var_u,";b1=",b1,";b2=",b2,"; epsw=",epsw,"; epsU=",epsU, ";minRMSE=",minimum(testRMSE[:,k]),";minepoch=",indmin(testRMSE[:,k]),";timer=",timer[k]);
end


#=
@everywhere avec=[1];
@everywhere b1vec=[0.05,10];
@everywhere b2vec=[0.05,10];
@everywhere signal_varvec=[0.001];
@everywhere var_uvec=[0.01];
@everywhere var_wvec=[0.01,0.1,1]
@everywhere numbatches=int(ceil(maximum(Ntrain)/m));
@everywhere t1=Iterators.product(avec,b1vec,b2vec,signal_varvec,var_uvec,var_wvec)
@everywhere myt1=Array(Any,myn);@everywhere myn1=2*2*3
@everywhere it=1;
@everywhere for prod in t1
	myt1[it]=prod;
        it+=1;
        end
testRMSEgibbs = SharedArray(Float64,maxepoch-burnin+1,myn1); trainRMSEgibbs = SharedArray(Float64,maxepoch-burnin+1,myn1);timergibbs =SharedArray(Float64,myn1)
@sync @parallel for k=1:myn1
	a,b1,b2,signal_var,var_u,var_w=myt1[k];
        phiUser = GPTinf.hash_feature(UserData,n,M,a,b1);
	phiMovie = GPTinf.hash_feature(MovieData,n,M,1,b2);
	phiUtrain=Array(Float64,size(phiUser,1),Ntrain);phiVtrain=Array(Float64,size(phiMovie,1),Ntrain);
	phiUtest=Array(Float64,size(phiUser,1),Ntest);phiVtest=Array(Float64,size(phiMovie,1),Ntest);
	for i=1:Ntrain
	    phiUtrain[:,i]=phiUser[:,Rating[i,1]]
	    phiVtrain[:,i]=phiMovie[:,Rating[i,2]]
	end
	 for i=1:Ntest
		phiUtest[:,i]=phiUser[:,Rating[Ntrain+i,1]]
		phiVtest[:,i]=phiMovie[:,Rating[Ntrain+i,2]]
	 end
	tic();
	 wGibbs,UGibbs,VGibbs=GPTinf.GPT_CFgibbs(phiUtrain, phiVtrain,ytrain, signal_var, var_u, var_w, r, maxepoch);
         println("w[1,1]=", wGibbs[1,1])
         trainpred=zeros(Ntrain);testpred=zeros(Ntest)
	 for epoch=burnin:maxepoch   ##average over samples
	        trainpred = (trainpred + GPTinf.pred_CF(wGibbs[:,epoch],UGibbs[:,:,epoch],VGibbs[:,:,epoch],phiUtrain,phiVtrain))/2
		trainRMSEgibbs[epoch-burnin+1,k]=ytrainStd*norm(ytrain-trainpred)/sqrt(Ntrain)
		testpred = (testpred + GPTinf.pred_CF(wGibbs[:,epoch],UGibbs[:,:,epoch],VGibbs[:,:,epoch],phiUtest,phiVtest))/2
		testRMSEgibbs[epoch-burnin+1,k]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
	end
	timergibbs[k]=toq(); 
	println("a=",a,";signal_var=",signal_var,";var_u=",var_u,";minRMSEgibbs=",minimum(testRMSEgibbs[:,k]),";minepoch=",indmin(testRMSEgibbs[:,k]),";timergibbs=",timergibbs[k]);
end
=#

#=
outfile=open("CF_CV_parafac","a") #append to file
    println(outfile,"testRMSE=",testRMSE,"; trainRMSE=",trainRMSE, "; timer=", timer, 
		"; testRMSEgibbs=", testRMSEgibbs, "; trainRMSEgibbs=", trainRMSEgibbs,"; timergibbs=", timergibbs,"; myt=", myt)
close(outfile)
=#

#=
##after CV, on the test set:
@everywhere ytrain = Rating[1:Ntrain+Ntest,3];
@everywhere ytrainMean=mean(ytrain);
@everywhere ytrainStd=std(ytrain);
@everywhere ytrain=datawhitening(ytrain);
@everywhere ytest = Rating[Ntrain+Ntest+1:end,3];
@everywhere ytest = (ytest-ytrainMean)/ytrainStd;
a1,b1,a2,b2,signal_var,epsw,epsU=myt[k]
phiUser = GPTinf.hash_feature(UserData,n,M,a1,b1);
phiMovie = GPTinf.hash_feature(MovieData,n,M,a2,b2);
phiMovie = vcat(phiMovie,zeros(size(UserData,2)-size(MovieData,2),size(MovieData,1)));
phitrain=Array(Float64,size(phiUser,1),2,Ntrain+Ntest);
phitest=Array(Float64,size(phiUser,1),2,size(Rating,1)-Ntrain-Ntest);
for i=1:(Ntrain+Ntest)
    phitrain[:,1,i]=phiUser[:,Rating[i,1]]
    phitrain[:,2,i]=phiMovie[:,Rating[i,2]]
end
 for i=1:(size(Rating,1)-Ntrain-Ntest)
	phitest[:,1,i]=phiUser[:,Rating[Ntrain+Ntest+i,1]]
	phitest[:,2,i]=phiMovie[:,Rating[Ntrain+Ntest+i,2]]
 end
tic(); 
w_store,U_store=GPTinf.GPTregression(phitrain, ytrain,signal_var, I, r, Q, m, epsw, epsU, burnin, maxepoch,param_seed,langevin=false,stiefel=true);
numbatches=int(ceil((Ntrain+Ntest)/m));trainRMSE_test= SharedArray(Float64,maxepoch);testRMSE_test= SharedArray(Float64,maxepoch);
@everywhere trainpred=zeros(size(ytrain,1),1);@everywhere testpred=zeros(size(ytest,1),1);
@sync @parallel for epoch=1:maxepoch
		#traintmp=pred(w_store[:,epoch*numbatches],U_store[:,:,:,epoch*maxepoch],I,phitrain)
		#if epoch > burnin
		#	trainpred = (trainpred + traintmp)/2
		#else 	trainpred = traintmp
		#end
		#trainRMSE_test[epoch]=ytrainStd*norm(ytrain-trainpred)/sqrt(Ntrain)
		testtmp=pred(w_store[:,epoch*numbatches],U_store[:,:,:,epoch*numbatches],I,phitest)
		if epoch > burnin
			testpred = (testpred + testtmp)/2
		else 	testpred = testpred
		end
		testRMSE_test[epoch]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
	end

##Gibbs
a,b,signal_var,epsw,epsU=myt[k]
phiUser = GPTinf.hash_feature(UserData,n,M,a,b);
phiMovie = GPTinf.hash_feature(MovieData,n,M,a,b);
phiMovie = vcat(phiMovie,zeros(size(UserData,2)-size(MovieData,2),size(MovieData,1)));
phitrain=Array(Float64,size(phiUser,1),2,Ntrain+Ntest);
phitest=Array(Float64,size(phiUser,1),2,Ntrain+Ntest);
for i=1:(Ntrain+Ntest)
    phitrain[:,1,i]=phiUser[:,Rating[i,1]]
    phitrain[:,2,i]=phiMovie[:,Rating[i,2]]
end
 for i=1:(size(Rating,1)-Ntrain-Ntest)
	phitest[:,1,i]=phiUser[:,Rating[Ntrain+Ntest+i,1]]
	phitest[:,2,i]=phiMovie[:,Rating[Ntrain+Ntest+i,2]]
 end
numiter = 20;trainRMSEgibbs_test= SharedArray(Float64,numiter);testRMSEgibbs_test= SharedArray(Float64,numiter)
wGibbs,UGibbs=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter);
@sync @parallel for epoch=1:numiter
    trainpred=pred(wGibbs[:,epoch],UGibbs[:,:,:,epoch],I,phitrain)
    trainRMSEgibbs_test[epoch]=ytrainStd*norm(ytrain-trainpred)/sqrt(Ntrain)
    testpred=pred(wGibbs[:,epoch],UGibbs[:,:,:,epoch],I,phitest)
    testRMSEgibbs_test[epoch]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
end

=#









