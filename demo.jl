@everywhere using DataFrames
@everywhere using GPTinf
@everywhere data=DataFrames.readtable("Folds5x2_pp.csv", header = true);

@everywhere data = convert(Array,data);
@everywhere N=size(data,1);
@everywhere D=4;
@everywhere Ntrain=5000;
@everywhere length_scale=1.4332;
@everywhere sigma=0.2299;
@everywhere Xtrain = data[1:Ntrain,1:D];
@everywhere ytrain = data[1:Ntrain,D+1];
@everywhere XtrainMean=mean(Xtrain,1);
@everywhere XtrainStd=zeros(1,D);
@everywhere for i=1:D
                XtrainStd[1,i]=std(Xtrain[:,i]);
            end
@everywhere ytrainMean=mean(ytrain);
@everywhere ytrainStd=std(ytrain);
@everywhere Xtrain = datawhitening(Xtrain);
@everywhere ytrain = datawhitening(ytrain);
@everywhere Xtest = (data[Ntrain+1:end,1:D]-repmat(XtrainMean,N-Ntrain,1))./repmat(XtrainStd,N-Ntrain,1);
@everywhere ytest = (data[Ntrain+1:end,D+1]-ytrainMean)/ytrainStd;
@everywhere burnin=250;
@everywhere numiter=50;
@everywhere r = 8;
@everywhere n = 50;


#### RMSE on test set #######
#=
restrain_Q=SharedArray(Float64,5,5);timertrain_Q=SharedArray(Float64,5,5);timer_Q = SharedArray(Float64,5,5)
restest_Q=SharedArray(Float64,5,5);timertest_Q=SharedArray(Float64,5,5);
Qvec = round(linspace(100,5000,5)); seed = 17;
phitrain=feature(Xtrain,n,length_scale,seed);
phitest=feature(Xtest,n,length_scale,seed);
@parallel for i in 1:5
    Q = convert(Int,Qvec[i])
    @parallel for j in 1:5
	tic();
   	seed = j;
    I=samplenz(r,D,Q,seed);
   	w_store,U_store=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter);
   	timer_Q[i,j] = toq();
  	tic();
  	restrain_Q[i,j] = ytrainStd*RMSE(w_store,U_store,I,phitrain,ytrain);
  	timertrain_Q[i,j] = toq() ;
        tic();
        restest_Q[i,j] = ytrainStd*RMSE(w_store,U_store,I,phitest,ytest);
   	timertest_Q[i,j] = toq() ;
    println(j)
    end
    println(Q)
end


outfile=open("RMSEgibbsQ","a") #append to file
    println(outfile,"restrain_Q=",restrain_Q,";timertrain_Q=",timertrain_Q,";timer_Q=",timer_Q,
            ";restest_Q=",restest_Q,";timertest_Q=",timertest_Q);
    close(outfile)


@everywhere r = 8; @everywhere Q = 100
restrain_n=SharedArray(Float64,5,5);timertrain_n=SharedArray(Float64,5,5);timer_n = SharedArray(Float64,5,5)
restest_n=SharedArray(Float64,5,5);timertest_n=SharedArray(Float64,5,5);
nvec = round(linspace(r,120,5))
@parallel for i in 1:5
    n = convert(Int,nvec[i])
    @parallel for j in 1:5
    	tic();
	seed = j;
	I=samplenz(r,D,Q,seed);
 	phitrain=feature(Xtrain,n,length_scale,seed);
 	phitest=feature(Xtest,n,length_scale,seed);
        w_store,U_store=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter);
        timer_n[i,j] = toq();
	tic();
	restrain_n[i,j] = ytrainStd*RMSE(w_store,U_store,I,phitrain,ytrain);
	timertrain_n[i,j] = toq();
	tic();
	restest_n[i,j] = ytrainStd*RMSE(w_store,U_store,I,phitest,ytest);
	timertest_n[i,j] = toq();
        println(j)
    end
    println(n)
end


outfile=open("RMSEgibbs_n","a") #append to file
    println(outfile,"restrain_n=",restrain_n,";timertrain_n=",timertrain_n,";timer_n=",timer_n,
            ";restest_n=",restest_n,";timertest_n=",timertest_n);
    close(outfile)
=#

@everywhere n = 50; @everywhere Q = 100; @everywhere seed = 17;
@everywhere phitrain=feature(Xtrain,n,length_scale,seed);
@everywhere phitest=feature(Xtest,n,length_scale,seed);
restrain_r=SharedArray(Float64,5,5);timertrain_r=SharedArray(Float64,5,5);;timer_r = SharedArray(Float64,5,5)
restest_r=SharedArray(Float64,5,5);timertest_r=SharedArray(Float64,5,5);
rvec = round(linspace(5,50,5));
@parallel for i in 1:5
    r = convert(Int,rvec[i])
    @parallel for j in 1:5
   	 tic();
	   seed = j;
  	 I=samplenz(r,D,Q,seed);
  	 w_store,U_store=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter);
  	 timer_r[i,j] = toq();
  	 tic();
  	 restrain_r[i,j] = ytrainStd*RMSE(w_store,U_store,I,phitrain,ytrain);
  	 timertrain_r[i,j] = toq();
  	 tic();
  	 restest_r[i,j] = ytrainStd*RMSE(w_store,U_store,I,phitest,ytest);
  	 timertest_r[i,j] = toq();
         println(j);
    end
    println(r)
end

outfile=open("RMSEgibbsr","a") #append to file
    println(outfile,"restrain_r=",restrain_r,";timertrain_r=",timertrain_r,";timer_r=",timer_r,
            ";restest_r=",restest_r,";timertest_r=",timertest_r,
            ";n=",n,";Q=",Q,";rvec=",rvec);
    close(outfile)
