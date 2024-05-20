import os
import numpy as np
from tqdm import tqdm

class analyze_msm_clusters:
    '''
    This class can be used to analyze the clustering (with dtraj file) in terms of its efficiency.
    Given the clustering performed on non-intuitive datasets which can be any of large dimensional data,
    features from tica, autoencoder, pca etc and etc...; an intuitive data (ddata) can be used here to 
    evaluate the clusters.
    This class can characterized the clusters based on provided intuitive data or their location on fes.
    
    #INPUTS:
        dtrj  - [nd-a,str] dtrj data or npy file
        
    #OPERATIONS:
        take_data                  - read the intuitive data
        mean                       - calculate mean and std of intuitive data for each cluster
        get_microstates            - return microstates or cluster satisfying constraints with some criteria (ctype)
        take_fes_data              - process the 2dim data to be later used for get_fes_loc. if more than 2-dim, top-2 r taken.
        get_fes_loc                - return the location of microstate(s) on a 2-D fes.
        get_metastable_composition - calculate the intuitive state composition of pcca derived metastable states
    
    '''
    
    def __init__(self, dtrj):
        if any(type(dtrj) == i for i in [list,np.ndarray]):
            self.dtrj = dtrj
        elif type(dtrj) == str and os.path.exists(dtrj):
            self.dtrj = np.load(dtrj, allow_pickle=True)
        else:
            raise ValueError('No dtrj file')
            
        self.nclus = np.max(np.concatenate((self.dtrj))) + 1
        
        
    def take_data(self, ddata, output=False):
        '''
        read intuitive data (same type/shape as dtrj data, can be data or npy file)
        '''
        if any(type(ddata) == i for i in [list,np.ndarray]):
            self.ddata = ddata
        elif type(ddata) == str and os.path.exists(ddata):
            self.ddata = np.load(ddata, allow_pickle=True)
        else:
            raise ValueError('No ddata')
            
        for i in range(len(self.dtrj)):
            if self.dtrj[i].shape != self.ddata[i].shape:
                raise ValueError('Mismatch in shapes of dtraj and ddata')
                
        if output == True: return 'YES YES, your data is taken'
            

    def get_mean(self, output=True):
        '''
        calculate mean, standard deviation and probabilities of each microstate in dtrj based on intuitive data
        
        #OUTPUT:
            mean        - [arr]   same as number of 
            std         - [arr]     microstates in 
            probability - [arr]     dtrj, obj.nclus
        '''
        self.mean = np.zeros((self.nclus))
        self.std = np.zeros((self.nclus))
        counts = np.zeros((self.nclus))
        for i in tqdm(range(len(self.dtrj))):
            for j in range(len(self.dtrj[i])):
                counts[self.dtrj[i][j]] += 1
                self.mean[self.dtrj[i][j]] += self.ddata[i][j]
                self.std[self.dtrj[i][j]] += np.square(self.ddata[i][j])

        self.mean = self.mean/counts
        self.std = np.sqrt(self.std/counts - np.square(self.mean))
        self.probability = counts / np.sum(counts)

        if output == True:
            return self.mean, self.std, self.probability
        
    def get_microstates(self, constraints, ctype='loose', output=True):
        '''
        extract the microstates satisfying the given criteria (constraints as per ctype) of intuitive data
        
        #INPUTS:
            constraints - [list/arr] [a,b] a < b, a-lower bound; b-upper bound
            ctype       - [str] loose - without std, strict - with std
        #OUTPUTS:
            states      - [arr] array of microstates satisfying the constraints
            
        '''
        self.constraints = constraints
        
        try:
            self.mean = self.mean
        except:
            raise AttributeError('try obj.mean first')
            
#         if not any(type(constraints) == i for i in [list,np.array]) and len(constraints) == 2:
        
        if ctype == 'loose':
            self.cstates = np.where((self.mean >= self.constraints[0]) & (self.mean <= self.constraints[1]))
        elif ctype == 'strict':
            dplus = self.mean + self.std
            dminus= self.mean - self.std
            self.cstates = np.where((dminus >= self.constraints[0]) & (dplus <= self.constraints[1]))
        else:
            raise ValueError('ctype {} not identified'.format(ctype))
        
        if output == True:
            return self.cstates
        
    
    def take_fes_data(self, fdata, fbins=100, fextra=0.1, output=False):
        
        try:
            self.fdata = np.concatenate((fdata))[:,[0,1]]
            if len(np.concatenate((self.dtrj))) != len(self.fdata):raise ValueError('problem with fdata')
        except:
            raise ValueError('problem with fdata')
        
        if type(fbins) == int:
            fbins = [fbins, fbins]
            self.frange0 = np.linspace(np.min(self.fdata[:,0]), np.max(self.fdata[:,0])+fextra, fbins[0]+1)
            self.frange1 = np.linspace(np.min(self.fdata[:,1]), np.max(self.fdata[:,1])+fextra, fbins[1]+1)
        elif any([type(fbins) == i for i in [list,np.ndarray]]) and len(fbins) == 2:
            if all(type(i)==int for i in fbins):
                self.frange0 = np.linspace(np.min(self.fdata[:,0]), np.max(self.fdata[:,0])+fextra, fbins[0]+1)
                self.frange1 = np.linspace(np.min(self.fdata[:,1]), np.max(self.fdata[:,1])+fextra, fbins[1]+1)
            elif any(type(fbins[0]) == [list, np.ndarray]) and any(type(fbins[1]) == [list, np.ndarray]):
                if np.min(self.fdata[:,0]) >= fbins[0][0] and np.max(self.fdata[:,0] < fbins[0][-1]) and np.min(self.fdata[:,1]) >= fbins[1][0] and np.max(self.fdata[:,1] < fbins[1][-1]):
                    self.frange0 = fbins[0]
                    self.frange1 = fbins[1]
                else: raise ValueError('problem with fbins')
            else: raise ValueError('problem with fbins')
        else: raise ValueError('problem with fbins')
            
        self.dx0 = (self.frange0[-1] - self.frange0[0]) / (len(self.frange0) - 1)
        self.dx1 = (self.frange1[-1] - self.frange1[0]) / (len(self.frange1) - 1)
        
        if output == True:
            return 'Taken'
        
        
    def get_fes_loc(self, mstate, output=True):
        if type(mstate) != int and mstate > self.nclus: raise ValueError('problem with mstate')
        try: self.dx0 = self.dx0
        except: raise AttributeError('try self.take_fes_data()')
        
        self.hist = np.zeros(( len(self.frange0)-1, len(self.frange1)-1 ))
        counter   = 0
        for i in range(len(self.dtrj)):
            for j in range(len(self.dtrj[i])):
                if self.dtrj[i][j] == mstate:
                    position = self.fdata[get_index(self.dtrj, i, j)]
                    bin0 = int(np.floor( (position[0] - self.frange0[0])/self.dx0 ))
                    bin1 = int(np.floor( (position[1] - self.frange0[1])/self.dx1 ))
                    self.hist[bin0,bin1] += 1
                    counter += 1
        
        self.hist /= counter
        self.hist[np.where(self.hist == 0)] = -np.inf
        if output == True: return self.frange0, self.frange1, self.hist
        
        
    
    def get_metastable_composition(self, psets, states, 
                                   means=None, stds=None, weights=None,
                                   calc_strength=1000,
                                   output_format='dict', output_type='pset'):
        
        if any(type(psets) == i for i in [list,np.ndarray]) and any(type(psets[np.random.randint(0,len(psets))]) == i for i in [list,np.ndarray]):
            if any(type(psets[np.random.randint(0,len(psets))][0]) == i for i in [int,np.int0,np.int16,np.int32,np.int64,np.int8]):
                self.psets = psets
                self.nstates = len(self.psets)
            else: raise ValueError('psets should be int')
        else: raise ValueError('psets should be list/array or list/array')
            
        if type(states) != dict: raise TypeError('states E [dict]')
        try:
            if all(len(i) == 2 and i[1] > i[0]  for i in [states[key] for key in states.keys()]): self.states = states
            else: raise ValueError('problem with states')
        except: raise ValueError('problem with states')
            
        if all(i==None for i in [means,stds,weights]):
            try:
                self.mean, self.std, self.weights = self.mean, self.std, self.probability
            except: raise AttributeError('try OBJECT.mean first:')
        elif not all(i==None for i in [means,stds,weights]):
            try:
                if all(len(i) == self.nclus for i in [means,stds,weights]): 
                    self.mean, self.std, self.weights = means, stds, weights
                else: raise ValueError('[means, stds, weights] == nclus({})'.format(self.nclus))
            except: raise TypeError('type([means, stds, weights]) E [list,np.ndarray]')
        else: raise ValueError('[means, stds, weights] all should be provided')
        
        
        self.composition = []
        for i in range(len(self.psets)):
            self.composition.append({key:np.zeros((len(self.psets[i]))) for key in self.states.keys()})
            self.composition[i]['others'] = np.zeros((len(self.psets[i])))
            
            probs = np.zeros((len(self.psets[i])))
            
            for j in range(len(self.psets[i])):
                micro = self.psets[i][j]
                probs[j] = self.weights[micro]
                sums = 0
                
                for k in self.states.keys():
                    area = get_area_under_gaussian(self.mean[micro], self.std[micro], 
                                                   self.states[k][0], self.states[k][1], calc_strength)
                    self.composition[i][k][j] = area 
                    sums += area
                
                self.composition[i]['others'][j] = 1-sums
                
            for k in self.composition[i].keys():
                self.composition[i][k] = ( self.composition[i][k] * probs ) / np.sum(probs)
                self.composition[i][k] = np.sum(self.composition[i][k])
                
        
        if output_type == 'pset':
            self.composition = self.composition
        
        elif output_type == 'state':
            
            probs = np.zeros((len(self.psets)))
            new_composition = {i:np.zeros((len(self.psets))) for i in self.composition[0].keys()}
            
            for i in range(len(self.psets)):
                probs[i] = np.sum(self.weights[self.psets[i]])
                
                for k in self.composition[i].keys():
                    new_composition[k][i] = self.composition[i][k]
            
            for k in new_composition.keys():
                new_composition[k] = (new_composition[k] * probs)/np.sum(probs)
                new_composition[k] = new_composition[k] / np.sum(new_composition[k])
                
            self.composition = new_composition
            
        else:
            raise ValueError('output_type E [pset, state]')
            
        
        if output_format == 'dict':
            return self.composition
        
        
        elif output_format == 'array':
            if type(self.composition) == dict:
                return np.array(list(self.composition.values()))
            
            else:
                return np.array([ list(i.values()) for i in self.composition ])
            
        else:
            raise ValueError('output_format E [dict,array]')
        
        

        
        
def get_index(ndim, i,j):
    index = 0
    for k in range(i):
        index += len(ndim[k])
    return index+j

def get_area_under_gaussian(mean,std,lower,upper,nbins):
    xr = np.linspace(lower, upper, nbins)
    dx = (xr[-1] - xr[0])/len(xr)
    fr = gaussian(xr, mean, std)
    mr = np.array([np.mean(fr[i:i+2]) for i in range(len(fr)-1)])
    sr = len(str(nbins)) - 1
    return np.round(np.sum(dx*mr),sr)
    
def gaussian(x,m,s):
    return (1/(s*np.sqrt(2*np.pi)) *  np.exp(-0.5 * np.square((x-m)/s)))
