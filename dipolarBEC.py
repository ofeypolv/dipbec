# class for dipolar BEC

import numpy as np
import matplotlib.pyplot as plt
import random as rdm
import math

from scipy import sparse #sparse matrices (efficient data storage)
from scipy import linalg #linear algebra routines for small matrices
from scipy.sparse import linalg as sparse_linalg #linear algebra for big sparse matrices

from tqdm import tqdm

# helper functions

def valvec(Hk):
	val, vec = linalg.eigh(Hk,eigvals_only=False)
	mask = np.logical_or(np.round(val.real,10)>0,np.round(val.imag,10)>0)
	val = val[mask]
	vec = vec[mask]
	return val, vec

def valvec_sparse(Hk,nbands,Ef):
	val, vec = sparse_linalg.eigsh(Hk,k=nbands,sigma=Ef,return_eigenvectors=True)
	mask = np.logical_or(np.round(val.real,10)>0,np.round(val.imag,10)>0)
	val = val[mask]
	vec = vec[mask]
	return val, vec

def ipr(v):
	v = np.asarray(v)
	return np.sum(np.abs(v**4))

def fold(v):
	Nt = int( np.shape(v)[0]/2 )
	v1 = v[0: Nt ]
	v2 = v[Nt : ]
	return v1**2 + v2**2

def wf(v):
	print('fold-v2', np.shape(fold(v)), np.shape(v))
	return fold(v)


####################################################################
#### 					THE CLASS BEGINS						####
####################################################################

class dipolarBEC():

	# input parameters:
	# Ntubes : number of tubes
	# kx : momentum along x direction
	# Uc : contact interaction
	# Ud : dipolar NN interaction
	# Ndisr : disorder realizations
	# sigma : width of densities distributed along the tubes

	# pre-set parameters:
	# sparseAlgo = [True, 80, 0.0] translates to sparse = True, number of states = 80, around E = 0
	# prestr, endstr : identifiers for save files

	# here we go

	def __init__(self, 
		Ntubes, 			# number of tubes
		kx, 			# momentum along x direction
		Uc,			# contact interaction
		Ud,	        # dipolar NN interaction
		Ndisr,      # disorder realizations to average over
		sigma,		# width of densities distributed along the tubes
		sparseAlgo = [False, 80, 0.0],	# sparse = False, number of states = 80, around E = 0
		prestr = '',				# prefix string for saving files
		endstr = '',				# suffix string for saving files
		):

		# save the global variables to self
		self.Ntubes = int(Ntubes)
		self.kx = kx
		self.Uc = Uc 
		self.Ud = Ud
		self.Ndisr = int(Ndisr)
		self.sigma = sigma
		self.sparseAlgo = sparseAlgo
		self.prestr = prestr
		self.endstr = endstr


		# determine if we are going to use sparse algoritm
		if sparseAlgo[0]:
			self._valvec = lambda H, nbands, Ef: valvec_sparse(H, nbands, Ef)
		else:
			self._valvec = lambda H, nbands, Ef: valvec(H)


	def makeBogoMat(self, nb):
		# given an array of Boson densities, make the Bogo Matrix
		# Copy Camilla's Code !!CCC!!
		h_1 = np.zeros( [self.Ntubes, self.Ntubes] )
		h_2 = np.zeros( [self.Ntubes, self.Ntubes] )

		for i in range(self.Ntubes):
			for j in range(self.Ntubes):
				if ( i==j ):
					h_1[i,j] = (self.kx**2)/(2.0) + self.Uc*nb[i]
					h_2[i,j] = self.Uc*nb[i]
				else:
					if (abs(i-j)<2): #Nearest neighbor dipolar interaction
						h_1[i,j] = self.Ud*(math.sqrt(nb[i]*nb[j]))
						h_2[i,j] = self.Ud*(math.sqrt(nb[i]*nb[j]))
		Haml = np.block([[h_1,h_2],[-h_2,-h_1]])

		return Haml

	def iprLowestState(self, nb):
		ham = self.makeBogoMat(nb)
		# Copy Camilla's Code !!CCC!!
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		return ipr( vec[-1] )

	def iprAlltStates(self, nb):
		ham = self.makeBogoMat(nb)
		# Copy Camilla's Code !!CCC!!
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		return [ipr( vec[i] ) for i in range(len(vec))]

	def wfLowestState(self):
		nb = np.random.uniform(1-self.sigma, 1+self.sigma, self.Ntubes)
		ham = self.makeBogoMat(nb)
		# Copy Camilla's Code !!CCC!!
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		return wf( vec[-1] )


	def IPRDisr(self):
		iprvec = []
		for i in range( self.Ndisr ):
			# create a disorder realization
			nb = np.random.uniform(1-self.sigma, 1+self.sigma, self.Ntubes)
			# force that sum is Ntubes
			offset = np.sum(nb) - self.Ntubes 
			nb = nb - offset/self.Ntubes

			# get the ipr
			iprvec.append( self.iprLowestState(nb)  )

		return np.mean(iprvec)

	def IPRAllDisr(self):
		iprvec = []
		for i in range( self.Ndisr ):
			# create a disorder realization
			nb = np.random.uniform(1-self.sigma, 1+self.sigma, self.Ntubes)
			# force that sum is Ntubes
			offset = np.sum(nb) - self.Ntubes 
			nb = nb - offset/self.Ntubes

			# get the ipr
			iprvec.append( self.iprAlltStates(nb)  )

		return np.mean(iprvec, axis=0)
	
