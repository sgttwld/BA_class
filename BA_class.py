import numpy as np
from matplotlib import pyplot as plt
import itertools
from scipy.special import lambertw
import copy

def marginalize(pagw,pw):
    return np.einsum('ik,i->k', pagw, pw)

def normalize(p):
    r = range(0,len(np.shape(p)))
    r0 = range(0,len(np.shape(p))-1)
    return np.einsum(p,r,1.0/np.einsum(p,r,r0),r0,r)

def DKL(post,prior):
    if len(np.shape(post)) == 1:
        return np.log(post/prior).dot(post)
    else:
        r = range(0,len(np.shape(post)))
        return np.einsum(post,r,np.log(post),r,r[:len(r)-1])-np.einsum(post,r,np.log(prior),r[1:],r[:len(r)-1])

def boltzmann(prior,beta,U):    # for len(shape(beta))>1, U has to be in the shape (N,M1,...,X) where (M1,...) = shape(beta)
    r = range(0,len(np.shape(U)))
    r_prior = r[len(r)-len(np.shape(prior)):]       #slice r from behind by the length of prior
    if len(np.shape(U)) > 2:
        betatimesU = np.einsum(beta,r[1:-1], U,r,r)
    else:
        betatimesU = beta[0]*U
    t = np.max(betatimesU)
    return normalize(np.einsum(prior,r_prior, np.exp(betatimesU-t),r, r))

def show(prob,axis=0,cols=4,vmi=0,vma=1,yrange=[],labels=[]):
    if not(type(prob) is np.ndarray):
        prob = np.array(prob)
    if len(prob.shape) == 1:
        plt.bar(range(0,len(prob)),prob)
        if len(yrange)>0:
            plt.ylim(yrange)
        if len(labels)>0:
            plt.xticks(range(0,len(prob)),labels)
    elif len(prob.shape) == 2:
        plt.pcolor(prob,vmin=vmi,vmax=vma)
    elif len(prob.shape) == 3:
        n = prob.shape[axis]
        X = int(n/cols)
        f, ax = plt.subplots(X,cols)
        for i in range(0,X):
            for j in range(0,cols):
                index = cols*i+j
                if axis == 0:
                    ax[i,j].pcolor(prob[index,:,:],vmin=vmi,vmax=vma)
                elif axis == 1:
                    ax[i,j].pcolor(prob[:,index,:],vmin=vmi,vmax=vma)
                elif axis == 2:
                    ax[i,j].pcolor(prob[:,:,index],vmin=vmi,vmax=vma)
    plt.show()

def shape_to_indices(dims):
    indices = []
    for index in itertools.product(*map(range,dims)):
        indices.append(index)
    return indices

def log(p):
    if len(np.shape(p)) == 1:
        lg = np.zeros(len(p))
        for i in range(0,len(p)):
            if p[i]<1e-50:
                lg[i] = 0
            else:
                lg[i] = np.log(p[i])
        return lg

class Solver(object):

    def __init__(self,dims,beta=[],BAtype=None,RD=True,alpha=[],genes=[],mult=False):
        self.steps = len(dims)-1                            #steps = 1,2,3
        self.BAtype = BAtype                                #BAtype = par,serial,None
        self.RD = RD                                        #RD = True, False
        self.dims = dims                                    #dims=[N,M1,M2,...,K]
        self.dimsvar = [(1,)]
        for i in range(1,self.steps):
            self.dimsvar.append(tuple(self.dims[1:i+1]))
        self.numnodes = self.calc_numnodes()
        self.post = []
        self.prior = []
        self.p0 = None
        self.F = []
        self.DKL = []
        self.DKLpr = []
        self.iterations = 0
        self.isDone = False
        self.test = None
        self.mult = mult
        # check for bounded priors:
        if len(alpha)>0:
            self.bpr = 1
        else:
            self.bpr = 0
            alpha = [1 for j in range(0,self.steps)]
        # betas:
        if self.steps >1:
            self.beta = [np.array([beta[0]])]
            # check for mult:
            if type(beta[1]) == list:
                self.mult = True
                for j in range(1,self.steps):
                    self.beta.append(np.array(self.b[j]))
            else:
                for j in range(1,self.steps):
                    self.beta.append(beta[j]*np.ones(self.dims[1:j+1]))
        else:
            self.beta = [np.array([beta])]
        # alphas:
        if self.steps > 1:
            alpha[0] = [alpha[0]]
            self.alpha = [np.array(alpha[0])]
            # check for mult:
            if type(alpha[1]) == list:
                self.mult = True
                for j in range(1,self.steps):
                    self.alpha.append(np.array(alpha[j]))
            else:
                for j in range(1,self.steps):
                    self.alpha.append(alpha[j]*np.ones(self.dims[1:j+1]))
        else:
            self.alpha = [np.array([alpha])]
            
        self.Finit = []

    def initialize(self,U,p0=np.array([]),pw=np.array([])):
        self.post = []
        self.prior = []
        self.F = []
        self.DKL = []
        self.DKLpr = []                                                              #DKL for priors
        self.iterations = 0
        self.isDone = False
        self.post.append(np.ones((self.dims[0],self.dims[1]))/self.dims[1])         # post[0] = pxgw (2step), pagw (1step)
        if len(pw) == 0:
            self.prior.append(np.ones(self.dims[0])/self.dims[0])                       # prior[0] = pw
        else:
            self.prior.append(pw)
        self.prior.append(np.ones(self.dims[1])/self.dims[1])                       # prior[1] = px (2step), pa (1step)
        if self.BAtype == 'ser':
            self.post.append(np.ones((self.dims[1],self.dims[2]))/self.dims[2])     # post[1] = pagx (ser)
            self.prior.append(np.ones(self.dims[2])/self.dims[2])                   # prior[2] = pa (ser)
        elif self.BAtype == 'par':
            for stp in range(2,self.steps+1):
                self.post.append(np.ones(self.dims[:stp+1])/self.dims[stp])         # post[1] = pagxw (par)
                self.prior.append(np.ones(self.dims[1:stp+1])/self.dims[stp])       # prior[2] = pagx (par)
        self.test = self.post[-1]
        if len(p0) == 0:
            self.p0 = [np.copy(prior) for prior in self.prior]
        else:
            self.p0 = [np.copy(prior) for prior in p0]
        if len(self.Finit) == 0:
            for stp in range(2,self.steps+1):
                self.Finit.append(np.random.rand(*self.dims[:stp]))                          # F[0] = Fwx (2step), F[1] = Fwx1x2 (3step)
            r = range(0,self.steps+1)
            self.Finit.append(np.einsum(np.ones(self.dims[1:self.steps]),r[1:-1],U,[0,self.steps],r)) # F[max] = U(a,w)
        self.F = copy.deepcopy(self.Finit)

    def initU(self,U):
        self.F[-1] = np.einsum('...,ik->i...k',np.ones(self.dims[1:self.steps]),U) # F[max] = U(a,w)

    def calc_numnodes(self):
        s = 0
        for d in self.dimsvar:
            s += np.prod(d)
        return s

    def evaluate(self,precision):
        self.iterations += 1
        if self.RD == False:
            self.isDone = True
        if np.linalg.norm(self.post[-1]-self.test) < precision:
            self.isDone = True

    def calc_joint(self):
        current = self.prior[0]
        for i in range(0,self.steps):
            r_new = range(0,i+2)
            r_post = r_new[len(r_new)-len(np.shape(self.post[i])):]
            r_current = range(0,i+1)
            current = np.einsum(current,r_current, self.post[i],r_post, r_new)
        self.joint = current

    def calc_pagw(self):
        self.pagw = np.einsum(self.joint,range(0,self.steps+1),1.0/self.prior[0],[0], [0,self.steps])

    def calc_EU(self,U):
        self.EU = np.einsum(self.joint,range(0,len(np.shape(self.joint))), U, [0,len(np.shape(self.joint))-1],[])

    def calc_DKL(self):
        self.DKL = []
        for j in range(0,self.steps):
            r_joint = range(0,self.steps+1)
            r = range(0,j+2)
            r_post = r[len(r)-len(np.shape(self.post[j])):]
            relsurpr = np.log(np.einsum(self.post[j],r_post,1.0/self.prior[j+1],r_post[1:],r_post))
            if j > 0 and self.mult == True and self.BAtype == 'par':
                pwx = np.einsum(self.joint,r_joint,r_post[:-1])
                px = np.einsum(pwx,r_post[:-1],r_post[1:-1])
                pwgx = np.einsum(pwx,r_post[:-1],1.0/px,r_post[1:-1],r_post[:-1])
                DKL_loc = np.einsum(self.post[j],r_post,relsurpr,r_post,r_post[:-1])
                self.DKL.append(np.einsum(pwgx,r_post[:-1],DKL_loc,r_post[:-1],r_post[1:-1]))
                # DKL_loc = np.einsum(self.post[j],r_post,relsurpr,r_post,r_post[:-1])
                # self.DKL.append(np.einsum(self.prior[0],[0],DKL_loc,r_post[:-1],r_post[1:-1]))
            else:
                self.DKL.append(np.einsum(self.joint,r_joint,relsurpr,r_post,[]))

    def calc_DKL_priors(self):
        self.DKLpr = []
        for j in range(0,self.steps):
            r_joint = range(0,self.steps+1)
            r = range(0,j+2)
            r_post = r[len(r)-len(np.shape(self.post[j])):]
            r_prior = r_post[1:]
            relsurpr = np.log(np.einsum(self.prior[j+1],r_prior,1.0/self.p0[j+1],r_prior,r_prior))
            DKL_loc = np.einsum(self.prior[j+1],r_prior,relsurpr,r_prior,r_prior[:-1])
            if j > 0 and self.mult == True:
                self.DKLpr.append(DKL_loc)
            else:
                self.DKLpr.append(np.einsum(self.joint,r_joint,DKL_loc,r_prior[:-1],[]))

    def calc_FE(self):
        FE = self.EU - self.DKL[0]/self.beta[0][0] - self.bpr*self.DKLpr[0]/self.alpha[0][0]
        if self.mult == True:
            for j in range(1,self.steps):
                r = range(1,j+1)
                r_joint = range(0,self.steps+1)
                r = range(0,j+2)
                r_post = r[len(r)-len(np.shape(self.post[j])):]
                r_prior = r_post[1:]
                r_DKL = r_prior[:-1]
                FE += (-np.einsum(self.joint,r_joint,1.0/self.beta[j],r_DKL,self.DKL[j],r_DKL,[])
                    - self.bpr*np.einsum(self.joint,r_joint,1.0/self.alpha[j],r_DKL,self.DKLpr[j],r_DKL,[]))
        elif self.mult == False:
            for j in range(1,self.steps):
                FE += -self.DKL[j]/self.beta[j].flatten()[0] - self.bpr*self.DKLpr[j]/self.alpha[j].flatten()[0]
        self.FE = FE

    def update_F(self,num,U):
        if self.BAtype == 'ser':
            if num == 99:   # update F[0] at the end
                r = [0,1,2]
                r0 = r[len(r)-len(np.shape(self.post[1])):]     # could do r[-len(np.shape(self.post[1])):]?
                self.F[0] = (np.einsum(U,[0,len(self.dims)-1], self.post[1],r0,[0,1])
                            - np.einsum(DKL(self.post[1],self.prior[2])/self.beta[1][0],[1],np.ones(self.dims[0]),[0],[0,1]))
            if num == 1:    # update F[1]
                mx = np.einsum('ij,i->j',self.post[0],self.prior[0])
                pwgx = np.einsum('i,ij,j->ij', self.prior[0], self.post[0], 1.0/mx)
                self.F[1] = np.einsum('ij,ik->jk', pwgx, U)
        else:  # RD and par: here we only update Fs at the end of each iteration step
            if num == 99:
                for j in range(0, self.steps):
                    r = range(0,self.steps+2-j)         #[0,1,2] in 2step case -> indices of post[steps-j]
                    r_F = r[:len(r)-1]
                    index = self.steps-1-j
                    if j == 0:
                        r = range(0,self.steps+1)
                        Ueff_exp = np.einsum(np.ones(self.dims[1:self.steps]),r[1:-1],U,[0,self.steps],r)
                        #Ueff_exp = np.einsum('...,ik->i...k',np.ones(self.dims[1:self.steps]),U)
                    else:
                        Ueff = (self.F[index+1] - np.einsum(np.log(np.einsum(self.post[index+1],
                        r,1.0/self.prior[index+2],r[1:],r)),r,1.0/self.beta[index+1],r[1:-1],r))
                        lg = np.log(np.einsum(self.prior[index+2],r[1:],1.0/self.p0[index+2],r[1:],r[1:]))
                        Ueff_exp = (np.einsum(self.post[index+1],r,Ueff,r,r_F)
                            - self.bpr*np.einsum(1.0/self.alpha[index+1],r[1:-1],self.prior[index+2],r[1:],lg,r[1:],np.ones(self.dims),range(0,len(self.dims)),r_F))
                    self.F[index] = Ueff_exp

    def update_prior(self,j):
        eps2 = 1e-55
        if self.RD == True:
            self.calc_joint()
            r = range(0,len(self.dims))
            l = len(np.shape(self.post[j]))-1
            r_pr = range(j+2-l,j+2)
            p = normalize(np.einsum(self.joint,r,r_pr))
            if np.shape(self.beta[j]) == (1,):
                r_beta = [0]
            else:
                r_beta = r_pr[1:]
            if self.bpr == 1 and self.BAtype == 'par':
                pr = np.copy(self.prior[j+1])
                indices = shape_to_indices(self.dims[j+2-l:j+1])
                for index in indices:
                    if self.alpha[j][index]/self.beta[j][index]>100:
                        pr[index] = normalize(p[index]/self.beta[j][index]-self.prior[j+1][index]*np.log(self.prior[j+1][index]/self.p0[j+1][index])/self.alpha[j][index]) + eps2
                    else:
                        DKL = np.log(self.prior[j+1][index]/self.p0[j+1][index]).dot(self.prior[j+1][index])
                        C = self.alpha[j][index]/self.beta[j][index] - DKL
                        denom = np.real(lambertw(np.exp(C)*(self.alpha[j][index]/self.beta[j][index])*p[index]/self.p0[j+1][index]))
                        pr[index] = (self.alpha[j][index]/self.beta[j][index])*p[index]/denom + eps2
                self.prior[j+1] = normalize(pr)
            elif self.bpr == 1:
                if self.alpha[j][0]/self.beta[j][0] > 100:
                    pr = p/self.beta[j][0] - self.prior[j+1]*np.log(self.prior[j+1]/self.p0[j+1])/self.alpha[j][0] + eps2
                else:
                    DKL = np.log(self.prior[j+1]/self.p0[j+1]).dot(self.prior[j+1])
                    C = self.alpha[j][0]/self.beta[j][0] - DKL
                    denom = np.real(lambertw(np.exp(C)*(self.alpha[j][0]/self.beta[j][0])*p/self.p0[j+1]))
                    pr = (self.alpha[j][0]/self.beta[j][0])*p/denom + eps2
                self.prior[j+1] = normalize(pr)
            else:
                self.prior[j+1] = p + eps2

    def init_search(self,U,sigma=2.0,initSearch=1,p0=np.array([])):
        Finit_sav = []
        FE_sav = 0
        self.initialize(U)
        for init in range(0,initSearch):
            self.iterate(U)
            if self.FE > FE_sav:
                FE_sav = self.FE
                Finit_sav = copy.deepcopy(self.Finit)
            for stp in range(0,self.steps-1):
                self.Finit[stp] = (Finit_sav[stp] + sigma*np.random.randn(*self.dims[:stp])).clip(0)
        self.Finit = Finit_sav

    def iterate(self,U,max_iterations=10000,precision=1e-10,p0=np.array([]),pw=np.array([])):
        eps2 = 1e-55
        self.initialize(U,p0,pw)
        for i in range(0,max_iterations):
            for j in range(0,self.steps):
                self.update_F(j,U)
                self.post[j] = boltzmann(self.prior[j+1],self.beta[j],self.F[j]) + eps2
                self.update_prior(j)
            self.update_F(99,U)
            self.evaluate(precision)
            if self.isDone == True:
                break
            self.test = self.post[-1]
        self.calc_joint()
        self.calc_pagw()
        self.calc_EU(U)
        self.calc_DKL()
        self.calc_DKL_priors()
        self.calc_FE()
