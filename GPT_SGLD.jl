module GPTinf

using Distributions,Optim,ForwardDiff

export datawhitening,feature,samplenz,RMSE,RMSESGLD,RMSESGLDvec, pred, GPT_SGLD

function datawhitening(X::Array)
    for i = 1:size(X,2)
        X[:,i] = (X[:,i] - mean(X[:,i]))/std(X[:,i])
    end
    return X
end

function data_simulator(X::Array,n::Integer,r::Integer,Q::Integer,sigma::Real,length_scale::Real,sigma_RBF::Real,seed::Integer)
  N,D = size(X)
  w = randn(Q)
  U=Array(Float64,n,r,D)
  for k=1:D
    Z=randn(r,n)
    U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) 
  end
  I=samplenz(r,D,Q,seed)
  phi = feature(X,r,length_scale,sigma_RBF,seed,1)
  temp=phidotU(U,phi)
  V=computeV(temp,I)
  return computefhat(V,w) + sigma*randn(N)
end

#extract features from tensor decomp of each row of X
function feature(X::Array,n::Integer,length_scale::Real,sigma_RBF::Real,seed::Integer,scale::Real)    
    N,D=size(X)
    phi=Array(Float64,n,D,N)
    srand(seed)
    Z=randn(n,D)/length_scale
    b=rand(n,D)*2*pi
    for i=1:N
	for k=1:D
	    for j=1:n
		phi[j,k,i]=cos(X[i,k]*Z[j,k]+b[j,k])
	    end
	end
    end
    return scale*(sigma_RBF)^(1/D)*sqrt(2/n)*phi
end

# sample the Q random non-zero locations of w
function samplenz(r::Integer,D::Integer,Q::Integer,seed::Integer)
    srand(seed)
    L=sample(0:(r^D-1),Q,replace=false)
    I=Array(Int32,Q,D)
    for q in 1:Q
        I[q,:]=digits(L[q],r,D)+1
    end
    # this way the locations are drawn uniformly from the lattice [r^D] without replacement
    # so I_qd=index of dth dim of qth non-zero
    return I
end


function proj(U::Array,V::Array)
    return V-U*(U'*V+V'*U)/2
end

# define geod for Stiefel manifold - just want endpt
function geod(U::Array,mom::Array,t::Real)
    n,r=size(U)
    A=U'*mom
    temp=[A -mom'*mom;eye(r) A]
    E=expm(t*temp) #can become NaN when temp too large. Return 0 in this case
    if sum(isnan(E))>0
        return zeros(n,r)
    else
        mexp=expm(-t*A)
        tmpU=[U mom]*E[:,1:r]*mexp;
        #ensure that tmpU has cols of unit norm
        normconst=Array(Float64,1,r);
        for l=1:r
    	    normconst[1,l]=norm(tmpU[:,l])
        end
        return tmpU./repmat(normconst,n,1)
    end
end
#compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch

function phidotU(U::Array,phi::Array)
    n,D,data_size=size(phi)
    r=size(U,2)
    temp=Array(Float64,D,r,data_size)
    for i=1:data_size
        for l=1:r
            for k=1:D
                temp[k,l,i]=dot(phi[:,k,i],U[:,l,k])
            end
        end
    end
    return temp
end

#compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])

function computeV(temp::Array,I::Array)
    Q,D=size(I);
    data_size=size(temp,3)
    V=ones(Q,data_size)
    for i=1:data_size
        for q=1:Q
            for k=1:D
                V[q,i]*=temp[k,I[q,k],i];
            end
        end
    end
    return V
end

#compute predictions fhat from V,w

function computefhat(V::Array,w::Array)
    data_size=size(V,2)
    fhat=Array(Float64,data_size)
    for i=1:data_size
        fhat[i]=dot(V[:,i],w)
    end
    return fhat
end


#compute predictions from w,U,I

function pred(w::Array,U::Array,I::Array,phitest::Array)

    # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,test and store in temp
    temp=phidotU(U,phitest)

    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
    V=computeV(temp,I)

    # compute fhat where fhat[i]=V[:,i]'w
    return computefhat(V,w)
end

function computeU_phi(V::Array,temp::Array,I::Array)
    Q,D=size(I)
    data_size=size(V,2)
    U_phi=Array(Float64,Q,data_size,D)
    for k=1:D
        for i=1:data_size
            for q=1:Q
	        U_phi[q,i,k]=V[q,i]/temp[k,I[q,k],i]
            end
	end
    end
    return U_phi
end


function computeA(U_phi::Array,w::Array,I::Array,r::Integer)
    Q,data_size,D=size(U_phi)
    A=zeros(r,D,data_size)
    for i=1:data_size
        for k=1:D
            for l in unique(I[:,k])
                index=findin(I[:,k],l) 
                A[l,k,i]=dot(U_phi[index,i,k],w[index]) 
            end
        end
    end
    return A
end


function computePsi(A,phi)
    r,D,data_size=size(A)
    n,D,data_size=size(phi)
    Psi=Array(Float64,n*r,data_size,D)
    for i=1:data_size
        for k=1:D
            Psi[:,i,k]=kron(A[:,k,i],phi[:,k,i])
        end
    end
    return Psi
end

