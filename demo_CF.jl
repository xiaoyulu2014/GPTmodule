@everywhere using DataFrames
@everywhere using GPTinf

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
@everywhere Ntrain = 8000;
@everywhere Ntest = 20000;
@everywhere UserData[:,2] = bin_age(UserData[:,2])
@everywhere UserData = convertdummy(convert(DataFrame,UserData),[:x2,:x3,:x4])[:,1:end-1];
@everywhere MovieData = MovieData[:,[1,6:end]];
@everywhere UserData = convert(Array{Float64,2},UserData)[:,2:end];
@everywhere MovieData = convert(Array{Float64,2},MovieData)[:,3:end];  #first column is "unknown" genre, discard
@everywhere ytrain = Rating[1:Ntrain,3];
@everywhere ytest = Rating[Ntrain+1:Ntrain+Ntest,3];
@everywhere ytrainMean=mean(ytrain);
@everywhere ytrainStd=std(ytrain);
@everywhere ytrain=datawhitening(ytrain);
@everywhere ytest = (ytest-ytrainMean)/ytrainStd;

@everywhere n = 150;
@everywhere M = 50;
@everywhere a,b=0.5,0.5;

@everywhere phiUser = GPTinf.hash_feature(UserData,n,M,a,b);
@everywhere phiMovie = GPTinf.hash_feature(MovieData,n+size(UserData,2)-size(MovieData,2),M,a,b);
@everywhere phitrain=Array(Float64,size(phiUser,1),2,Ntrain)
@everywhere phitest=Array(Float64,size(phiUser,1),2,Ntest)
@everywhere   for i=1:Ntrain
			phitrain[:,1,i]=phiUser[:,Rating[i,1]]
			phitrain[:,2,i]=phiMovie[:,Rating[i,2]]
		end

@everywhere   for i=1:Ntest
			phitest[:,1,i]=phiUser[:,Rating[Ntrain+i,1]]
			phitest[:,2,i]=phiMovie[:,Rating[Ntrain+i,2]]
		end

@everywhere param_seed=17;
@everywhere burnin=0;
@everywhere numiter=20;
@everywhere Q=200;   
@everywhere r = 30
@everywhere D = 2;
@everywhere sigma = 0.3;
@everywhere signal_var = sigma^2;
@everywhere using Distributions
@everywhere seed=17;
@everywhere I=samplenz(r,D,Q,seed); 
@everywhere m = 500;
@everywhere maxepoch = 10;

@everywhere using Iterators
@everywhere t=Iterators.product(2:4,4:6)
@everywhere myt=Array(Any,9);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
        it+=1;
        end

### as a function of training data points

@everywhere epsw=0.0001; @everywhere epsU=0.000001;@everywhere r=30;@everywhere I=samplenz(r,D,Q,seed);@everywhere @everywhere Nvec=[0.2,0.4,0.6,0.8,1.0]*80000
numbatches=int(ceil(maximum(Nvec)/m)); myn=size(Nvec,1);
testRMSE = SharedArray(Float64,maxepoch*numbatches,myn); trainRMSE= SharedArray(Float64,maxepoch*numbatches,myn);timer=SharedArray(Float64,myn);
@sync @parallel for k=1:myn
         Ntrain = convert(Int,Nvec[k])
         ytrain = Rating[1:Ntrain,3];
         ytest = Rating[end-Ntest+1:end,3];
         ytrainMean=mean(ytrain);
         ytrainStd=std(ytrain);
         ytrain=datawhitening(ytrain);
         ytest = (ytest-ytrainMean)/ytrainStd;

         phitrain=Array(Float64,size(phiUser,1),2,Ntrain)
         phitest=Array(Float64,size(phiUser,1),2,Ntest)
         for i=1:Ntrain
		phitrain[:,1,i]=phiUser[:,Rating[i,1]]
		phitrain[:,2,i]=phiMovie[:,Rating[i,2]]
	 end
         for i=1:Ntest
		phitest[:,1,i]=phiUser[:,Rating[end-Ntest+i,1]]
		phitest[:,2,i]=phiMovie[:,Rating[end-Ntest+i,2]]
	 end
        tic(); 
        w_store,U_store=GPTinf.GPTregression(phitrain, ytrain,signal_var, I, r, Q, m, epsw, epsU, burnin, maxepoch,param_seed);
	numbatches=int(ceil(Ntrain/m))
	for epoch=1:maxepoch*numbatches
		trainpred=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitrain)
		trainRMSE[epoch,k]=ytrainStd*norm(ytrain-trainpred)/sqrt(Ntrain)
		testpred=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitest)
		testRMSE[epoch,k]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
	end
        timer[k] = toq();  
println("Ntrain=",Ntrain,";minRMSE=",minimum(testRMSE[:,k]),";minepoch=",indmin(testRMSE[:,k]),";epsw=",epsw,";epsU=",epsU,";burnin=",burnin,";maxepoch=",maxepoch);
end
#=outfile=open("CF_Ntrain","a") #append to file
    println(outfile,"Ntrain=",Ntrain,"; testRMSE=",testRMSE,"; trainRMSE=",trainRMSE,
		    "; testRMSE=", testRMSE, "; timer=", timer)
    close(outfile)
=#



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
    w_store,U_store=GPTinf.GPTregression(phitrain, ytrain,signal_var, I, r, Q, m, epsw, epsU, burnin, maxepoch,param_seed);
   # testRMSE=Array(Float64,maxepoch*numbatches)
    numbatches=int(ceil(Ntrain/m))
    @sync @parallel for epoch=1:maxepoch*numbatches
        trainpred=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitrain)
	trainNMAE[epoch,k]=ytrainStd/(1.6*Ntrain) * sum(abs(ytrain-trainpred))
        testpred=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitest)
        testRMSE[epoch,k]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
	testNMAE[epoch,k]=ytrainStd/(1.6*Ntest) * sum(abs(ytest-testpred))
    end
    epsvec[k,:] = [epsw,epsU];
    timer[k] = toq();
println("r=",r,";minRMSE=",minimum(testRMSE[:,k]),";minepoch=",indmin(testRMSE[:,k]),";epsw=",epsw,";epsU=",epsU,";burnin=",burnin,";maxepoch=",maxepoch);
end
=#


##tuning ab=====
#=
@everywhere ab=Iterators.product([0.1,0.3,0.5,0.7],[0.1,0.3,0.7,0.5])
@everywhere myab=Array(Any,16);
@everywhere it=1;
@everywhere for prod in ab
	myab[it]=prod;
        it+=1;
        end
@everywhere epsw=0.001; @everywhere epsU=0.00001
@everywhere myn=size(myab,1);
testRMSE = SharedArray(Float64,maxepoch*numbatches,myn); testNMAE= SharedArray(Float64,maxepoch*numbatches,myn);trainNMAE= SharedArray(Float64,maxepoch*numbatches,myn);timer=SharedArray(Float64,myn);
@sync @parallel for k=1:size(myab,1)
    a,b = myab[k]
    tic()
 phiUser = hash_feature(UserData,n,M,a,b);
 phiMovie = hash_feature(MovieData,n+size(UserData,2)-size(MovieData,2),M,a,b);

 phitrain=Array(Float64,size(phiUser,1),2,Ntrain)
 phitest=Array(Float64,size(phiUser,1),2,Ntest)
  for i=1:Ntrain
			phitrain[:,1,i]=phiUser[:,Rating[i,1]]
			phitrain[:,2,i]=phiMovie[:,Rating[i,2]]
		end

   for i=1:Ntest
			phitest[:,1,i]=phiUser[:,Rating[Ntrain+i,1]]
			phitest[:,2,i]=phiMovie[:,Rating[Ntrain+i,2]]
	end
    w_store,U_store=GPTinf.GPTregression(phitrain, ytrain,signal_var, I, r, Q, m, epsw, epsU, burnin, maxepoch,param_seed);
   # testRMSE=Array(Float64,maxepoch*numbatches)
    numbatches=int(ceil(Ntrain/m))
    @sync @parallel for epoch=1:maxepoch*numbatches
        trainpred=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitrain)
	trainNMAE[epoch,k]=ytrainStd/(1.6*Ntrain) * sum(abs(ytrain-trainpred))
        testpred=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitest)
        testRMSE[epoch,k]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
	testNMAE[epoch,k]=ytrainStd/(1.6*Ntest) * sum(abs(ytest-testpred))
    end
    timer[k] = toq();
println("r=",r,";minRMSE=",minimum(testRMSE[:,k]),";minepoch=",indmin(testRMSE[:,k]),";a=",a,";b=",b,";burnin=",burnin,";maxepoch=",maxepoch);
end

###  find hyper parameters signal var
phifull=Array(Float64,size(phitrain,1)^2,size(phitrain,3))
for i=1:size(phitrain,3)
	phifull[:,i]=kron(phitrain[:,1,i],phitrain[:,2,i])
end
GPNT_hyperparameters_CF(phifull,ytrain,n,0.1,seed)
=#


##try 
#=
trainNMAEvec=SharedArray(Float64,maxepoch*numbatches);testRMSEvec=SharedArray(Float64,maxepoch*numbatches);testNMAEvec=SharedArray(Float64,maxepoch*numbatches);
@sync @parallel for epoch=1:maxepoch*numbatches
               trainpred=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitrain)
           trainNMAEvec[epoch]=ytrainStd/(1.6*Ntrain) * sum(abs(ytrain-trainpred))
               testpred=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitest)
               testRMSEvec[epoch]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
           testNMAEvec[epoch]=ytrainStd/(1.6*Ntest) * sum(abs(ytest-testpred))
           end

@everywhere function RMSEfunc(w_store::Array,U_store::Array,phitest::Array,ytest::Array)
	T = size(w_store,2);Ntest=size(ytest,1)
	tmp=zeros(Ntest)
	for epoch=1:T
		tmp += pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitest)
	end
	tmp = tmp/T
	return(ytrainStd*norm(ytest-tmp)/sqrt(Ntest))
end
RMSEfunc(w_store[:,end-50:end],U_store[:,:,:,end-50:end],phitest,ytest)
=#

### as a function of n
#=
@everywhere nvec=[50,100,200,300];@everywhere epsw=0.001; @everywhere epsU=0.00001;@everywhere r=30;@everywhere I=samplenz(r,D,Q,seed);
numbatches=int(ceil(Ntrain/m)); myn=size(nvec,1);
testRMSE = SharedArray(Float64,maxepoch*numbatches,myn); trainRMSE= SharedArray(Float64,maxepoch*numbatches,myn);timer=SharedArray(Float64,myn);
testRMSEgibbs = SharedArray(Float64,numiter,myn); trainRMSEgibbs = SharedArray(Float64,numiter,myn);timergibbs =SharedArray(Float64,myn)
@sync @parallel for k=1:myn
         n = convert(Int,nvec[k])

         phiUser = GPTinf.hash_feature(UserData,n,M,a,b);
         phiMovie = GPTinf.hash_feature(MovieData,n+size(UserData,2)-size(MovieData,2),M,a,b);
         phitrain=Array(Float64,size(phiUser,1),2,Ntrain)
         phitest=Array(Float64,size(phiUser,1),2,Ntest)
         for i=1:Ntrain
		phitrain[:,1,i]=phiUser[:,Rating[i,1]]
		phitrain[:,2,i]=phiMovie[:,Rating[i,2]]
	 end
         for i=1:Ntest
		phitest[:,1,i]=phiUser[:,Rating[Ntrain+i,1]]
		phitest[:,2,i]=phiMovie[:,Rating[Ntrain+i,2]]
	 end
        tic(); 
        w_store,U_store=GPTinf.GPTregression(phitrain, ytrain,signal_var, I, r, Q, m, epsw, epsU, burnin, maxepoch,param_seed);
	numbatches=int(ceil(Ntrain/m))
	for epoch=1:maxepoch*numbatches
		trainpred=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitrain)
		trainRMSE[epoch,k]=ytrainStd*norm(ytrain-trainpred)/sqrt(Ntrain)
		testpred=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitest)
		testRMSE[epoch,k]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
	end
        timer[k] = toq();
        tic();
        wGibbs,UGibbs=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter);
         for epoch=1:numiter
            trainpred=pred(wGibbs[:,epoch],UGibbs[:,:,:,epoch],I,phitrain)
	    trainRMSEgibbs[epoch,k]=ytrainStd*norm(ytrain-trainpred)/sqrt(Ntrain)
            testpred=pred(wGibbs[:,epoch],UGibbs[:,:,:,epoch],I,phitest)
            testRMSEgibbs[epoch,k]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
        end
        timergibbs[k]=toq();
println("n=",n,";minRMSE=",minimum(testRMSE[:,k]),";minepoch=",indmin(testRMSE[:,k]),";minRMSEgibbs=",minimum(testRMSEgibbs[:,k]),";epsw=",epsw,";epsU=",epsU,";burnin=",burnin,";maxepoch=",maxepoch);
end
=#
outfile=open("CF_n","a") #append to file
    println(outfile,"testRMSEgibbs=",testRMSEgibbs,"; timergibbs=",timergibbs,"; trainRMSEgibbs=",trainRMSEgibbs,"; testRMSE=",testRMSE,"; trainRMSE=",trainRMSE,"; timer=", timer)
    close(outfile)

### as a function of r
#=
@everywhere rvec=[15,20,30,50,100];@everywhere epsw=0.001; @everywhere epsU=0.00001;
numbatches=int(ceil(Ntrain/m)); myn=size(rvec,1);
testRMSE = SharedArray(Float64,maxepoch*numbatches,myn); testNMAE= SharedArray(Float64,maxepoch*numbatches,myn);trainNMAE= SharedArray(Float64,maxepoch*numbatches,myn)
timer=SharedArray(Float64,myn)
@sync @parallel for k=1:myn
         r = convert(Int,rvec[k])
   	 tic(); 
  	 I=samplenz(r,D,Q,seed);
         w_store,U_store=GPTinf.GPTregression(phitrain, ytrain,signal_var, I, r, Q, m, epsw, epsU, burnin, maxepoch,param_seed);
	numbatches=int(ceil(Ntrain/m))
	for epoch=1:maxepoch*numbatches
		trainpred=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitrain)
		trainNMAE[epoch,k]=ytrainStd/(1.6*Ntrain) * sum(abs(ytrain-trainpred))
		testpred=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitest)
		testRMSE[epoch,k]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
		testNMAE[epoch,k]=ytrainStd/(1.6*Ntest) * sum(abs(ytest-testpred))
	end
    timer[k] = toq();
println("r=",r,";minRMSE=",minimum(testRMSE[:,k]),";minepoch=",indmin(testRMSE[:,k]),";epsw=",epsw,";epsU=",epsU,";burnin=",burnin,";maxepoch=",maxepoch);
end



outfile=open("CF_SGLDr","a") #append to file
    println(outfile,"testNMAE=",testNMAE,"; trainNMAE=",trainNMAE,
		    "; testRMSE=", testRMSE, "; epsvec=", epsvec, "; timer=", timer)
    close(outfile)
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
