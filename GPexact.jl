module GPexact

export GP, GPmean, GPcov, SECov, Mean, Cov, GPrand, GPpost, plot1d, datawhitening

type GP
    mean::Union(Function,Real) #mean function (could also be const)
    cov::Function #covariance function
    dim::Integer #ndims of the domain of GP
end

function GPmean(gp::GP)
    #function to return mean function of GP
    return gp.mean
end

function GPcov(gp::GP)
    #function to return covariance function of GP
    return gp.cov
end

function datawhitening(X::Array)
    for i = 1:size(X,2)
        X[:,i] = (X[:,i] - mean(X[:,i]))/std(X[:,i])
    end
    return X
end

######### List of covariance functions ###############
function SECov(length::Real,sigma::Real)
    function f(x,y)
        return sigma^2*exp(-norm(x-y)^2/(2*length^2))
    end
    return f
end

######################################################

function Mean(gp::GP,x_values)
    #function to return gp.mean(x_values) and gp.cov(x_values,x_values)
    #mean as 1D float array and cov as float 2D array
    #Can assume input x_values is a 2D array
    n=size(x_values,1); #number of x_values

    #first evaluate mean
    meanfn=gp.mean;
    if typeof(meanfn)<:Real #deals with case where mean is const
        mean=[meanfn for i=1:n];
    else #if mean is a function
        mean=[meanfn(x_values[i,:]) for i=1:n];
    end
    m=float(map(scalar,mean));
    return m
end

function Cov(gp::GP,x_values)
    n=size(x_values,1);
    covfn=gp.cov;
    K=[covfn(x_values[i,:],x_values[j,:]) for i=1:n, j=1:n];
    return float(K)
end



function GPpost(gp::GP,xtrain,ytrain,xtest,sigma)
    #function to return the posterior mean and variance
    temp=convertTo2D(xtrain);
    xtrain=dimensionCheck(gp.dim,temp);
    temp=convertTo2D(ytrain);
    y_values=vec(temp); #want to keep as 1D array

    n=size(xtrain,1);
    ntest = size(xtest,1);
    K=Cov(gp,xtrain);
    covfn=gp.cov;

    Ktest = [covfn(xtest[i,:],xtrain[j,:]) for i = 1:ntest, j = 1:n]
    mean_post = Ktest * reshape(\(K + sigma^2*eye(n),ytrain),n,1)
    return mean_post
end

################### private functions #################
function convertTo2D(x)
    #function to convert objects of dimension 0,1 to 2D
    p=ndims(x);
    if p>2
        error ("dimension of x_values exceeds 3")
    end
    if p==0
        x_new=reshape([x],1,1);
        return x_new
    elseif p==1
        #when dim=1, default to row vector
        x_new=reshape(x,1,length(x));
        return x_new
    else
        return x
    end
end

function dimensionCheck(gpdim,x_values)
    #function to check that dimensions of x values=gpdim
    #Can Assume that x_values is a 2D array.
    #Say size(x_values)=(n,p).
    #We also need to distinguish between cases:
    #1. x_values is a single point in dim>1
    #2. x_values is multiple points in 1D
    (n,p)=size(x_values);
    if p==1 && gpdim==n && n>1 #case 1 but have col vector
        x_new=x_values';
    elseif n==1 && gpdim==1 && p>1 #case 2 but have row vector
        x_new=x_values';
    else
        x_new=x_values;
    end
    if size(x_new,2)==gpdim
        return x_new
    else
        error("dimensions of x_values do not match those of GP")
    end
end

function scalar(x)
    assert(length(x) == 1)
    return x[1]
end


end
