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

#=
@everywhere User = UserData[Rating[:,1],2:end];
@everywhere User = convert(Array{Float64,2},User);
@everywhere Movie = MovieData[Rating[:,2],2:end];
@everywhere Movie = convert(Array{Float64,2},Movie);
@everywhere data = hcat(User,Movie);
@everywhere data[isnan(data)] = 0.0
@everywhere N,D = size(data);
=#
@everywhere Ntrain = 50000;
@everywhere Ntest = 50000;

@everywhere UserData = convertdummy(convert(DataFrame,UserData),[:x3,:x4])[:,1:end-1];
@everywhere MovieData = MovieData[:,[1,6:end]];
@everywhere UserData = convert(Array{Float64,2},UserData)[:,2:end];
@everywhere MovieData = convert(Array{Float64,2},MovieData)[:,2:end];
@everywhere UserData=datawhitening(UserData);
@everywhere MovieData=datawhitening(MovieData);
@everywhere n = 150; @everywhere M = 50
@everywhere a,b=0.5,0.5;
@everywhere phiUser = GPTinf.hash_feature(UserData,n,M,a,b);
@everywhere phiMovie = GPTinf.hash_feature(MovieData,n+22-19,M,a,b);

@everywhere phitrain=Array(Float64,n+22,2,Ntrain)
@everywhere phitest=Array(Float64,n+22,2,Ntest)
@everywhere   for i=1:Ntrain
			phitrain[:,1,i]=phiUser[:,Rating[i,1]]
			phitrain[:,2,i]=phiMovie[:,Rating[i,2]]
		end

@everywhere   for i=1:Ntest
			phitest[:,1,i]=phiUser[:,Rating[Ntrain+i,1]]
			phitest[:,2,i]=phiMovie[:,Rating[Ntrain+i,2]]
		end



@everywhere ytrain = Rating[1:Ntrain,3];
@everywhere ytest = Rating[Ntrain+1:Ntrain+Ntest,3];
@everywhere ytrainMean=mean(ytrain);
@everywhere ytrainStd=std(ytrain);
@everywhere ytrain=datawhitening(ytrain);
@everywhere ytest = (ytest-ytrainMean)/ytrainStd;

## learning hyperparameters

#GPNT_hyperparameters(Xtrain,ytrain,n,0.5,0.5,0.5,seed)
@everywhere burnin=0;
@everywhere numiter=5;
@everywhere Q=200;   #200
@everywhere r = 30
@everywhere D = 2;
@everywhere signal_var = 0.1;
@everywhere sigma = 0.1;
#SGLD
@everywhere using Distributions
@everywhere seed=17;
@everywhere I=samplenz(r,D,Q,seed); 
@everywhere m = 500;
@everywhere maxepoch = 8;

@everywhere using Iterators
@everywhere t=Iterators.product(3:5,5:7)
@everywhere myt=Array(Any,9);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
        it+=1;
        end

#=
#tuning epsw and epsU
numbatches=int(ceil(Ntrain/m));myn=size(myt,1);
testRMSE = SharedArray(Float64,maxepoch*numbatches,myn); testNMAE= SharedArray(Float64,maxepoch*numbatches,myn);trainNMAE= SharedArray(Float64,maxepoch*numbatches,myn);
epsvec=SharedArray(Float64,myn,2);timer=SharedArray(Float64,myn);

#epsw=0.01;epsU=2*float(string("1e-",5)),j=4 is the best
@sync @parallel for k=1:size(myt,1)
    i,j = myt[k]
    tic()
    epsw=float(string("1e-",i)); epsU=float(string("1e-",j));
    w_store,U_store=GPTinf.GPT_SGLDERM(phitrain, ytrain,signal_var, I, r, Q, m, epsw, epsU, burnin, maxepoch);
   # testRMSE=Array(Float64,maxepoch*numbatches)
    numbatches=int(ceil(Ntrain/m))
    for epoch=1:maxepoch*numbatches
        trainpred=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitrain)
	trainNMAE[epoch,k]=ytrainStd/(1.6*Ntrain) * sum(abs(ytrain-trainpred))
        testpred=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitest)
        testRMSE[epoch,k]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
	testNMAE[epoch,k]=ytrainStd/(1.6*Ntest) * sum(abs(ytest-testpred))
    end
    epsvec[k,:] = [epsw,epsU];
    timer[k] = toq();
println("r=",r,";minRMSE=",minimum(testNMAE[:,j]),";minepoch=",indmin(testNMAE[:,j]),";epsw=",epsw,";epsU=",epsU,";burnin=",burnin,";maxepoch=",maxepoch);
end
=#

### as a function of r
@everywhere rvec=[15,20,30,50,100];@everywhere epsw=0.001; @everywhere epsU=0.0000005;
numbatches=int(ceil(Ntrain/m)); myn=size(rvec,1);
testRMSE = SharedArray(Float64,maxepoch*numbatches,myn); testNMAE= SharedArray(Float64,maxepoch*numbatches,myn);trainNMAE= SharedArray(Float64,maxepoch*numbatches,myn)
timer=SharedArray(Float64,myn)
@sync @parallel for k=1:myn
         r = convert(Int,rvec[k])
   	 tic(); 
  	 I=samplenz(r,D,Q,seed);
         w_store,U_store=GPTinf.GPT_SGLDERM(phitrain, ytrain,signal_var, I, r, Q, m, epsw, epsU, burnin, maxepoch);
	numbatches=int(ceil(Ntrain/m))
	for epoch=1:maxepoch*numbatches
		trainpred=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitrain)
		trainNMAE[epoch,k]=ytrainStd/(1.6*Ntrain) * sum(abs(ytrain-trainpred))
		testpred=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitest)
		testRMSE[epoch,k]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
		testNMAE[epoch,k]=ytrainStd/(1.6*Ntest) * sum(abs(ytest-testpred))
	end
    timer[k] = toq();
println("r=",r,";minRMSE=",minimum(testNMAE[:,j]),";minepoch=",indmin(testNMAE[:,j]),";epsw=",epsw,";epsU=",epsU,";burnin=",burnin,";maxepoch=",maxepoch);
end



#=
@everywhere burnin=0; @everywhere numiter=50; @everywhere rvec = round(linspace(30,80,5));@everywhere seed=17
testNMAE=SharedArray(Float64,numiter,5);trainNMAE=SharedArray(Float64,numiter,5);testRMSE = SharedArray(Float64,numiter,5);timer=SharedArray(Float64,5);
@sync @parallel for i in 1:5
    r = convert(Int,rvec[i])
    tic();
    I=samplenz(r,D,Q,seed);
    wGibbs,UGibbs=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter);
    for epoch=1:numiter
        trainpred=pred(wGibbs[:,epoch],UGibbs[:,:,:,epoch],I,phitrain)
	trainNMAE[epoch,i]=ytrainStd/(1.6*Ntrain) * sum(abs(ytrain-trainpred))
        testpred=pred(wGibbs[:,epoch],UGibbs[:,:,:,epoch],I,phitest)
        testRMSE[epoch,i]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
	testNMAE[epoch,i]=ytrainStd/(1.6*Ntest) * sum(abs(ytest-testpred))
    end
    timer[i] = toq();
println("r=",r,";minRMSE=",minimum(testNMAE[:,i]),";minepoch=",indmin(testNMAE[:,i]));
end
=#

#=
outfile=open("CF","a") #append to file
    println(outfile,"testNMAE=",testNMAE,"; trainNMAE=",trainNMAE,
		    "; testRMSE=", testRMSE, "; testRMSE=", testRMSE, 
			"; timer=", timer)
    close(outfile)


outfile=open("CF_SGLD","a") #append to file
    println(outfile,"testNMAE=",testNMAE,"; trainNMAE=",trainNMAE,
		    "; testRMSE=", testRMSE, "; epsvec=", epsvec, "; timer=", timer)
    close(outfile)
=#

