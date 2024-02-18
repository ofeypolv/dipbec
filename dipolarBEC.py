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
	val, vec = linalg.eig(Hk)
	mask = np.logical_or(np.round(val.real,10)>0,np.round(val.imag,10)>0)
	val = val[mask]
	vec = vec[:, mask]
	idx = val.argsort()
	val = val[idx]
	vec = vec[:, idx]
	return val, vec

def valvec_sparse(Hk,nbands,Ef):
	val, vec = sparse_linalg.eigs(Hk,k=nbands,sigma=Ef)
	mask = np.logical_or(np.round(val.real,10)>0,np.round(val.imag,10)>0)
	val = val[mask]
	vec = vec[:, mask]
	idx = val.argsort()
	val = val[idx]
	vec = vec[:, idx]
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
	#print('fold-v2', np.shape(fold(v)), np.shape(v))
	return fold(v)

def summ(a,b,N):
	if a+b <= N:
		return a+b
	else:
		return a+b-N #summation with periodic boundary conditions
	
def dirac_delta(x, a): #Dirac delta as a limit of a Lorentzian, a=1e-3
    return a / (x**2 + a**2) 

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
		NN_int = True,		# binary variable for NN vs 1/x^3 interaction
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
		#Normalize the eigenvectors wrt matrix s3n using broadcasting
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2])
		normalization = np.sqrt(np.einsum('ij,ij->j', vec, np.matmul(s3n, vec)))
		vec /= normalization
		#for i in range(vec.shape[1]): #normalize the eigenvectors wrt matrix s3n
			#vec[:, i] = vec[:, i] / np.sqrt(np.matmul(vec[:, i].T, np.matmul(s3n, vec_s[:, i])))
		pvec = np.matmul(pn, np.conj(vec)) #define bottom half of eigenvectors through the parity matrix pn
		vec_s = np.concatenate((vec, pvec), axis=1)		
		U = vec_s[0:self.Ntubes, 0:self.Ntubes]
		V = vec_s[self.Ntubes:, 0:self.Ntubes]
		#T = np.block([[U, V], [np.conj(V), np.conj(U)]])
		#np.set_printoptions(precision=2, suppress=True)
		return val,U,V

	def iprLowestState(self, nb):
		ham = self.makeBogoMat(nb)
		# Copy Camilla's Code !!CCC!!
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		return ipr( vec[:, -1] )

	def iprAlltStates(self, nb):
		ham = self.makeBogoMat(nb)
		# Copy Camilla's Code !!CCC!!
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		iprv = [ipr( vec[:, i] ) for i in range(vec.shape[1])]
		#print(f'val:{val}')
		#print(f'iprv:{iprv}')
		return [val, iprv]

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
		iprval = []
		for i in range( self.Ndisr ):
			# create a disorder realization
			nb = np.random.uniform(1-self.sigma, 1+self.sigma, self.Ntubes)
			# force that sum is Ntubes
			offset = np.sum(nb) - self.Ntubes 
			nb = nb - offset/self.Ntubes
			# get the ipr and the corresponding energy
			iprvec.append( self.iprAlltStates(nb)[1]  )
			iprval.append( self.iprAlltStates(nb)[0]  )
		return np.mean(iprval, axis=0),np.mean(iprvec, axis=0)
	
	def visc_k_ij(self, ny, i,j, nb): 
		val_s, U, V = self.BogUV(nb)
		int_k = 0

		for y in range(1,self.Ntubes + 1):
			if y+ny <= self.Ntubes:
				term_uuvv = U[y-1,i-1]*np.conj(U[y+ny-1,i-1])*V[y-1,j-1]*np.conj(V[y+ny-1,j-1])
				term_uvvu = U[y-1,i-1]*np.conj(V[y+ny-1,i-1])*V[y-1,j-1]*np.conj(U[y+ny-1,j-1])
				int_k += (self.kx**2)*(term_uuvv - term_uvvu)
			else:
				int_k += 0

		return int_k

	def visc_k_time(self,ny, time, nb): 
		val_s,U,V = self.BogUV(nb)
		intd_k = 0

		for i in range(1,self.Ntubes + 1):
			for j in range(1,self.Ntubes + 1):
				intd_k += -2*self.visc_k_ij(ny,i,j,nb)*np.sin( (val_s[i-1] + val_s[j-1] )*time)

		return intd_k


	def visc_k_om(self,ny,om, nb, gamma): 
		val_s,U,V = self.BogUV(nb)
		intd_ko = 0

		for i in range(1,self.Ntubes + 1):
			for j in range(1,self.Ntubes + 1):
				fc_plus = 1/(om + val_s[i-1] + val_s[j-1] + 1j*gamma ) #plemelj-sokhotski formula
				fc_mnus = 1/(om - val_s[i-1] - val_s[j-1] + 1j*gamma ) #plemelj-sokhotski formula
				intd_ko += self.visc_k_ij(ny,i,j,nb)*(fc_plus-fc_mnus)

		return intd_ko
