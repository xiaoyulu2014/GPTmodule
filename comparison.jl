#compare tensor with no tensor case
using TGP,NoTensorGP
using PyPlot

N = 100; n = 10; sigmaRBF = 0.5; sigma = 0.1; generator = 123; r = 5; q = 100; num_iterations = 10; burnin = 5
cov = SECov(sigmaRBF); x = randn(N,2) ; f = GP(cov,x) ; y = f + sigma*randn(N)
xtrain, xtest = x[1:(N/2),:], x[(N/2+1):end,:]
ytrain, ytest = y[1:(N/2),:], y[(N/2+1):end,:]
ftrain, ftest = f[1:(N/2),:], f[(N/2+1):end,:]

RMSEnotensor = Float64[]; timernotensor = Float64[];
for n in round(linspace(100,50000,5))
    n = convert(Int64,n)
    RMSE, timer = FullTheta(xtrain,ytrain,featureNotensor,n,sigmaRBF,sigma,generator,xtest,ftest)
    RMSEnotensor = push!(RMSEnotensor,RMSE)
    timernotensor = push!(timernotensor,timer)
end

RMSE = Float64[]; timer = Float64[];
for n in round(linspace(20,150,5))
    n = convert(Int64,n)
    now = tic()
    tmp = TensorRes(xtrain,ytrain,sigma,n,r,sigmaRBF,q,generator,num_iterations,burnin,xtest,ftest)
    RMSE = push!(RMSE,tmp)
    timer = push!(timer,toc())
    println(RMSE)
end

