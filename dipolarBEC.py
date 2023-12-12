# class for dipolar BEC

import numpy as np
import matplotlib.pyplot as plt
import random as rdm
import math

from scipy import sparse #sparse matrices (efficient data storage)
from scipy import linalg #linear algebra routines for small matrices
from scipy.sparse import linalg as sparse_linalg #linear algebra for big sparse matrices
from scipy.special import kn #modified Bessel function of the second kind of integer order n

from tqdm import tqdm

# helper functions

def valvec(Hk):
	val, vec = linalg.eigh(Hk,eigvals_only=False)
	mask = np.logical_or(np.round(val.real,10)>0,np.round(val.imag,10)>0)
	vval = val[mask]
	vvec = vec[:, mask]
	#print('valvec', np.shape(val), np.shape(vec))
	#print('vvalvec', np.shape(vval), np.shape(vvec))
	return vval, vvec

def valvec_sparse(Hk,nbands,Ef):
	val, vec = sparse_linalg.eigsh(Hk,k=nbands,sigma=Ef,return_eigenvectors=True)
	mask = np.logical_or(np.round(val.real,10)>0,np.round(val.imag,10)>0)
	val = val[mask]
	vec = vec[:, mask]
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

def summ(a,b,N):
	if a+b <= N:
		return a+b
	else:
		return a+b-N #summation with periodic boundary conditions

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
		NN_int = False,		# binary variable for NN vs 1/x^3 interaction
		sparseAlgo = [False, 80, 0.0],	# sparse = False, number of states = 80, around E = 0
		prestr = '',				# prefix string for saving files
		endstr = '',				# suffix string for saving files
		):

		# save the global variables to self
		self.Ntubes = int(Ntubes)
		self.kx = kx
		self.Uc = Uc
		self.NN_int = NN_int 
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
					if self.NN_int:
						if (abs(i-j)<2): #Nearest neighbor dipolar interaction
							h_1[i,j] = self.Ud*(math.sqrt(nb[i]*nb[j]))
							h_2[i,j] = self.Ud*(math.sqrt(nb[i]*nb[j]))
					else: #1/x^3 dipolar interaction
						h_1[i,j] = self.Ud*(math.sqrt(nb[i]*nb[j]))*2*self.kx*kn(1,abs(i-j)*self.kx)/(abs(i-j))
						h_2[i,j] = self.Ud*(math.sqrt(nb[i]*nb[j]))*2*self.kx*kn(1,abs(i-j)*self.kx)/(abs(i-j))


		Haml = np.block([[h_1,h_2],[-h_2,-h_1]])

		return Haml

	def BogUV(self, nb):
	# given the Bogo matrix, we extract the U and V matrices from its eigenvectors
		identity_matrix_n = np.eye(self.Ntubes)
		zero_matrix_n = np.zeros((self.Ntubes, self.Ntubes))
		s3n = np.block([[identity_matrix_n, zero_matrix_n],[zero_matrix_n, identity_matrix_n]]) #s3n is the n-dim pauli matrix sigmaz
		pn = np.block([[zero_matrix_n, identity_matrix_n],[1*identity_matrix_n, zero_matrix_n]]) #pn is the n-dim parity matrix
		ham = self.makeBogoMat(nb)
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2])
		#print(val)
		#print(vec.T)
		sort_index = np.argsort(val)
		val_s = val[sort_index]
		vec_s = vec[:, sort_index]
		#print(val_s)
		#print(vec_s.T)
		pvec_s = np.empty_like(vec_s)
		for i in range(vec_s.shape[1]): #define bottom half of eigenvectors through the parity matrix pn
			pvec_s[:, i] = np.matmul(pn, np.conj(vec_s[:, i]))
		#print(pvec_s.T)
		vec_s = np.concatenate((vec_s, pvec_s), axis=1)
		for i in range(vec_s.shape[1]): #normalize the eigenvectors wrt matrix s3n
			vec_s[:, i] = vec_s[:, i] / np.sqrt(np.matmul(vec_s[:, i].T, np.matmul(s3n, vec_s[:, i])))
		#print(vec_s.T)
		U = vec_s[0:self.Ntubes, 0:self.Ntubes]
		V = vec_s[self.Ntubes:, 0:self.Ntubes]
		#T = np.block([[U, V], [np.conj(V), np.conj(U)]])
		#np.set_printoptions(precision=2, suppress=True)
		#print(U)
		#print(V)
		return val_s,U,V

	def iprLowestState(self, nb):
		ham = self.makeBogoMat(nb)
		# Copy Camilla's Code !!CCC!!
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		return ipr( vec[:, -1] )

	def iprAlltStates(self, nb):
		ham = self.makeBogoMat(nb)
		# Copy Camilla's Code !!CCC!!
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		return [ipr( vec[:, i] ) for i in range(vec.shape[1])]

	def wfLowestState(self):
		nb = np.random.uniform(1-self.sigma, 1+self.sigma, self.Ntubes)
		ham = self.makeBogoMat(nb)
		# Copy Camilla's Code !!CCC!!
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		return wf( vec[:, -1] )

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
	
	def visc_k(self,ny,t): 
		nb = np.random.uniform(1-self.sigma, 1+self.sigma, self.Ntubes)
		val_s,U,V = self.BogUV(nb)
		intd_k = 0

		for y in range(1,self.Ntubes + 1):
			for i in range(1,self.Ntubes + 1):
				for j in range(1,self.Ntubes + 1):
					if y+ny <= self.Ntubes:
						intd_k += 2*(self.kx**2)*np.imag((U[y-1,i-1]*np.conj(V[y+ny-1,i-1])*V[y-1,j-1]*np.conj(U[y+ny-1,j-1])-U[y-1,i-1]*np.conj(U[y+ny-1,i-1])*V[y-1,j-1]*np.conj(V[y+ny-1,j-1]))*np.exp(-1j*(val_s[i-1] + val_s[j-1])*t))
					else:
						intd_k += 0
		return intd_k