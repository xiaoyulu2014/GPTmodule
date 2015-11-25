module GPT_Gibbs

using Distributions

export datawhitening,featureNotensor,RMSE,FullTheta
  
function featureNotensor(X::Array,n::Integer,length_scale::Real,seed::Integer)    
    N,D=size(X)
    phi=Array(Float64,n,N)
    srand(seed)
    Z=randn(n,D)/length_scale
    b=2*pi*rand(n)
    for i=1:N
        phi[:,i]=cos(sum(repmat(X[i,:],n,1).*Z,2) + b)
    end
    return sqrt(2/n)*phi
end


function datawhitening(X::Array) 
    for i = 1:size(X,2)   
        X[:,i] = (X[:,i] - mean(X[:,i]))/std(X[:,i])   
    end
    return X
end


function RMSE(theta::Array,phitest::Array,ytest::Array)
    Ntest=length(ytest);
    ffit = phitest'*theta;
    return sqrt(sum((ffit - ytest).^2)/Ntest);
end


function FullTheta(phi::Array,y::Array,sigma::Real)
    n = size(phi,1);
    theta = Array(Float64,n,1);
    invSigma = float(1/(sigma^2) * phi * phi' + (1/sigma^2)*eye(n))
    theta = \(invSigma,float(1/(sigma^2) *(phi * y)))
    return theta
end

end
 