'''
'
' LIBRARY FOR ACTIVE INFERENCE
'
' by Sherin Grimbergen, TU Delft 03-2019
' 
' Note: Most likely bug: data types must be numpy matrices!
'''

import numpy as np
import scipy.linalg as la
# I do not want these to appear in files where I import ssai!!

def temporalPrec(k,gamma):
    """
    Construct the temporal variance matrix V(gamma)
    and return its inverse S, the  temporal precision
    
    INPUTS:
        k       - embedding order (>=1)
        gamma   - roughness parameter (>0)
        
    OUTPUT:
        S       - temporal precision matrix (k x k)
    """
    
    s = np.sqrt(2/gamma)
    p = np.arange(k)
    
    r = np.zeros(1+2*(k-1))
    r[2*p] = np.cumprod(1-2*p)/s**(2*p)    
    
    V = np.empty([0,k])

    for i in range(k):
        V = np.vstack([V,r[p+i]])
        r = -r
        
    S = la.inv(V)
    
    return S


def makeNoise(C,gamma,t):
    """
    Construct non-Markovian noise from white Gaussian noise
    
    INPUTS:
        C       - desired covariance matrix
        gamma   - roughness parameter (>0)
        
    OUTPUT:
        ws       - non-Markovian noise sequence
    """

    if np.size(C) == 1:
        n = 1
    elif np.size(np.shape(C)) == 1:
        n = np.size(C)  
        C = np.diag(C)        
    else:
        n = C.shape[0]
        
    # Generate white Gaussian noise sequences:
    # note: cholesky decomposition of C used to generate w with correct covariance
    N = np.size(t)      # number of elements
    L = la.cholesky(C).T   # square root of C
    w = L.dot(np.random.randn(n,N))
    
    if gamma > 1e4: # return white noise
        return w
    else: # smoothen noise
        # Set up convolution matrix:
        P = la.toeplitz(np.exp(-gamma/4*t**2))
        F = np.diag(np.sqrt(np.diag(P.T.dot(P))))
    
        # Make the smoothened noise:
        K  = P.dot(la.inv(F))
        ws = w.dot(K)
    
        return ws

def dynamics(s,t,dt,w,z,system,agent):
    """
    Convert current closed loop state 's' to its derivative 'ds'
    
    INPUTS:
        s       - current state (x,mu,u)
        t       - current time
        w       - full simulation state noise array
        z       - full simulation observation noise array
        system  - generative process state space model
        agent   - instance of activeInferenceAgent class
        
    OUTPUT:
        ds       - derivative of s at current time
    """
    # Retrieve current exogeneous inputs:
    i  = int(round(t/dt))
        
    w  = w[:,i:i+1]
    z  = z[:,i:i+1]   
    xi = agent.xi[:,i:i+1]

    # Unpack state:
    x  = np.matrix(s[0:agent.gm.n]).T
    mu = np.matrix(s[agent.gm.n:agent.gm.n*agent.gm.k+1]).T
    u  = np.matrix(s[agent.gm.n*agent.gm.k+1:]).T
        
    # Calculate derivatives using dynamics:
    dx  = system.A.dot(x) + system.B.dot(u) + w
    y   = system.C.dot(x) + z
    
    dmu = agent.perception(mu,y,xi)
    du  = agent.action(mu,y)

    # Save the full state derivative (compress dimensions for odeint function)
    ds = np.concatenate((dx,dmu,du),0)
    ds = np.squeeze(np.array(ds))
    
    return ds

class activeInferenceAgent():
    """
        Class that constructs an agent that can perform Active Inference
    """
    def __init__(self,k,gamma,Cw,Cz,a_mu,a_u,system):                
        # Scalars:
        self.k     = k          # embedding order (>1)
        self.gamma = gamma      # roughness parameter (>0)
        self.a_mu  = a_mu       # perception learning rate (>0)
        self.a_u   = a_u        # action learning rate (>0)
        
        if self.k == 1:
            #TODO: program what to do without generalised motions
            raise ValueError('k=1 not yet supported')
        
        # Precision matrices:
        Cw = np.matrix(Cw)
        Cz = np.matrix(Cz)
        self.S = temporalPrec(self.k,self.gamma)
        self.Piz   = la.inv(Cz)
        self.Piw   = np.kron(self.S,la.inv(Cw))
        
        # Internal models:
        self.gm = generativeModel(self.k,system)
        self.fm = forwardModel(system).getModel()
        
        # Other matrices:
        self.K = []             # pole placement
            
        # Initialize data vectors:
        self.mu = []   # generalised state belief
        self.xi = []   # prior variable
        self.F  = []   # free energy
        
        # ---- end ----
            
    ''' Perform perception '''
    def perception(self,mu,y,xi):
        # note: y and xi are input at current time
        if min(np.shape(mu))>1: #or min(np.shape(y))>1 or min(np.shape(xi))>1:
            raise ValueError('Cannot do perception for sequences of data.')
            
        Piw = self.Piw
        Piz = self.Piz 
        A   = self.gm.A
        C   = self.gm.C
        D   = self.gm.D
                
        # Free energy gradient:
        dFdmu = (D-A).T.dot(Piw).dot((D-A).dot(mu)-xi) \
                - C.T.dot(Piz).dot(y-C.dot(mu))
                
        # Perception dynamics:
        dmu   = D.dot(mu) - self.a_mu*dFdmu    
        
        return dmu
    
    ''' Perform action '''
    def action(self,mu,y):
        # Forward model:
        G   = self.fm
        C   = self.gm.C
        Piz = self.Piz
        #Free energy gradient:
        dFdy = Piz.dot(y-C.dot(mu))
        # Action dynamics:
        du = -self.a_u*G.T.dot(dFdy)
        return du
    
    ''' Set the pole placement term ''' 
    def setPolePlacement(self,K):
        self.K = K
        Ktilde = np.kron(np.identity(self.k),K)
        self.gm.A += Ktilde
        return
    
    ''' Set the prior variable '''
    def setPrior(self,x_ref,K):
        # x_ref is the reference (trajectory)
        n, N = x_ref.shape
        
        if N==1: # static reference
            muref = np.concatenate((x_ref,np.zeros([n*(self.gm.k-1),1])),0)
            self.setPolePlacement(K)
            xi   = (self.gm.D-self.gm.A).dot(muref)
            xi   = np.matlib.repmat(xi,1,N)
            self.xi = xi
            
        elif N>1: # dynamic reference
            raise ValueError('Dynamic reference tracking not yet implemented')
    
    ''' Evaluate the free energy '''
    def getFreeEnergy(self,mu,y,xi):
        # Extract length of input sequences:
        for x in mu.shape:
            N = x
            if x != self.k*self.gm.n:
                N = x
                break
        
        Piw = self.Piw
        Piz = self.Piz
        # Calculate free energy for each timepoint:   
        self.F = np.empty((1,N)) # initialize array
        for i in range(N):
            # Prediction errors:
            eps_mu = (self.gm.D-self.gm.A).dot(mu[:,i]) - xi[:,i]
            eps_y  =  y[:,i] - self.gm.C.dot(mu[:,i])
            
            # Free energy:
            self.F[0,i] = 0.5*( eps_mu.T.dot(Piw).dot(eps_mu) + eps_y.T.dot(Piz).dot(eps_y) )
                        
        return self.F

class generativeModel():
    """ Generative Model for Active Inference Agent """
    def __init__(self,k,system):
        self.k = k
        # Extract matrices and dimensions (for LTI state space model):
        try:
            A = system.A
            C = system.C
            self.n = A.shape[0]
            self.q = C.shape[0]
        except NameError:
            print('Your system is probably not defined as a linear state space model')
        
        Atilde = np.kron(np.identity(self.k),A)
        Ctilde = np.concatenate((C,np.zeros([self.q,self.n*(self.k-1)])),1)
        
        self.A = Atilde
        self.C = Ctilde
        
        self.setD()
     
        ''' Construct the shifting operator '''
    def setD(self):
        # Shifting operator:
        try:
            T  = la.toeplitz(np.zeros([1,self.k]),np.append(np.array([0,1]),np.zeros([1,self.k-2])))
            In = np.identity(self.n)
            D  = np.kron(T,In)
        except ValueError:
            print('Improper value for embedding order, k = ' + str(self.k))
        except:
            print('Something weird went wrong due to the embedding order...')
        
        self.D = D
        return

        # ---- end ----
        
class forwardModel():
    """ Forward Model for Active Inference Agent """
    def __init__(self,system):
        A = system.A
        B = system.B
        C = system.C
        
        self.n = A.shape[1]
        
        if system.dt == None:
            self.fm = -C.dot(la.inv(A).dot(B))
        else:
            self.fm = C.dot(la.inv(np.identity(self.n)-A)).dot(B)
        
    def getModel(self):
        return self.fm 

    # ---- end ----
    
# For testing purposes:
"""
if __name__== "__main__":

    k = 3
    gamma = 10
    S = temporalPrec(k,gamma)
    print(inv(S))
    
    Cw = 10
    Cz = 20
    a_mu = 1
    a_u = 1
    
    system = ss(1,1,1,0)
    
    t = np.arange(0,10.1,.1)
    w = makeNoise(Cw,gamma,t)
    
    plt.plot(t,w.T)
    
    agent = activeInferenceAgent(k,gamma,Cw,Cz,a_mu,a_u,system)
"""
