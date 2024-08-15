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

def csi(v):
	v = np.asarray(v)
	return 1/(np.sum(np.abs(v**4)))

def fold(v):
	Nt = int( np.shape(v)[0]/2 )
	v1 = v[0: Nt ]
	v2 = v[Nt : ]
	return v1**2 + v2**2

def iprfd(v):
	v = np.asarray(v)
	Nt = int( np.shape(v)[0]/2 )
	v1 = v[0: Nt ]
	v2 = v[Nt : ]
	#print('check state normalization:', np.sum(np.abs(v1**2 - v2**2)))
	return np.sum(np.abs((v1**2 - v2**2)**2))

def csifd(v):
	Nt = int( np.shape(v)[0]/2 )
	v1 = v[0: Nt ]
	v2 = v[Nt : ]
	return 1/(np.sum(np.abs((v1**2 - v2**2)**2)))

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

class dipolarBECnb():

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
		nb,		# densities along the tubes
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
		self.nb = nb
		self.sparseAlgo = sparseAlgo
		self.prestr = prestr
		self.endstr = endstr

		# determine if we are going to use sparse algoritm
		if sparseAlgo[0]:
			self._valvec = lambda H, nbands, Ef: valvec_sparse(H, nbands, Ef)
		else:
			self._valvec = lambda H, nbands, Ef: valvec(H)


	def makeBogoMat(self):
		# given an array of Boson densities, make the Bogo Matrix
		h_1 = np.zeros( [self.Ntubes, self.Ntubes] )
		h_2 = np.zeros( [self.Ntubes, self.Ntubes] )

		for i in range(self.Ntubes):
			for j in range(self.Ntubes):
				if ( i==j ):
					h_1[i,j] = (self.kx**2)/(2.0) + self.Uc*self.nb[i]
					h_2[i,j] = self.Uc*self.nb[i]
				else:
					if self.NN_int:
						if (abs(i-j)<2): #Nearest neighbor dipolar interaction
							h_1[i,j] = self.Ud*(math.sqrt(self.nb[i]*self.nb[j]))
							h_2[i,j] = self.Ud*(math.sqrt(self.nb[i]*self.nb[j]))
					else: #1/x^3 dipolar interaction
						h_1[i,j] = self.Ud*(math.sqrt(self.nb[i]*self.nb[j]))*2*self.kx*kn(1,abs(i-j)*self.kx)/(abs(i-j))
						h_2[i,j] = self.Ud*(math.sqrt(self.nb[i]*self.nb[j]))*2*self.kx*kn(1,abs(i-j)*self.kx)/(abs(i-j))


		Haml = np.block([[h_1,h_2],[-h_2,-h_1]])

		return Haml

	def norm(self,vec):
		identity_matrix_n = np.eye(self.Ntubes)
		zero_matrix_n = np.zeros((self.Ntubes, self.Ntubes))
		s3n = np.block([[identity_matrix_n, zero_matrix_n],[zero_matrix_n, -identity_matrix_n]])
		normalization = np.sqrt(np.einsum('ij,ij->j', vec, np.matmul(s3n, vec)))
		#print('check state normalization:', normalization)
		vec = vec / normalization
		#print('check state normalization:', np.sqrt(np.einsum('ij,ij->j', vec, np.matmul(s3n, vec))))
		return vec

	def BogUV(self):
	# given the Bogo matrix, we extract the U and V matrices from its eigenvectors
		identity_matrix_n = np.eye(self.Ntubes)
		zero_matrix_n = np.zeros((self.Ntubes, self.Ntubes))
		s3n = np.block([[identity_matrix_n, zero_matrix_n],[zero_matrix_n, -identity_matrix_n]]) #s3n is the n-dim pauli matrix sigmaz
		pn = np.block([[zero_matrix_n, identity_matrix_n],[1*identity_matrix_n, zero_matrix_n]]) #pn is the n-dim parity matrix
		ham = self.makeBogoMat()
		#Normalize the eigenvectors wrt matrix s3n using broadcasting
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2])
		vec = self.norm(vec)
		#for i in range(vec.shape[1]): #normalize the eigenvectors wrt matrix s3n
			#vec[:, i] = vec[:, i] / np.sqrt(np.matmul(vec[:, i].T, np.matmul(s3n, vec_s[:, i])))
		pvec = np.matmul(pn, np.conj(vec)) #define bottom half of eigenvectors through the parity matrix pn
		vec_s = np.concatenate((vec, pvec), axis=1)		
		U = vec_s[0:self.Ntubes, 0:self.Ntubes]
		V = vec_s[self.Ntubes:, 0:self.Ntubes]
		#T = np.block([[U, V], [np.conj(V), np.conj(U)]])
		#np.set_printoptions(precision=2, suppress=True)
		return val,U,V	
	
	def wfLowestState(self):
		ham = self.makeBogoMat()
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		vec = self.norm(vec)
		return wf( vec[:, 0] )
	
	### IPR FUNCTIONS

	def iprLowestState(self):
		ham = self.makeBogoMat()
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		return ipr( vec[:, 0] )
	
	def csiLowestState(self):
		ham = self.makeBogoMat()
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		return csi( vec[:, 0] )
	
	def iprLowestStatefd(self):
		ham = self.makeBogoMat()
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		vec = self.norm(vec)
		return iprfd( vec[:, 0] )
	
	def csiLowestStatefd(self):
		ham = self.makeBogoMat()
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		vec = self.norm(vec)
		return csifd( vec[:, 0] )
	
	def iprMidState(self):
		ham = self.makeBogoMat()
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		Lm = vec.shape[1] // 2
		return ipr( vec[:, Lm] )
	
	def csiMidState(self):
		ham = self.makeBogoMat()
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		Lm = vec.shape[1] // 2
		return csi( vec[:, Lm] )
	
	def iprMidStatefd(self):
		ham = self.makeBogoMat()
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		vec = self.norm(vec)
		Lm = vec.shape[1] // 2
		return iprfd( vec[:, Lm] )
	
	def csiMidStatefd(self):
		ham = self.makeBogoMat()
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		vec = self.norm(vec)
		Lm = vec.shape[1] // 2
		return csifd( vec[:, Lm] )
	
	def iprHighestState(self):
		ham = self.makeBogoMat()
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		return ipr(vec[:, -1])
	
	def csiHighestState(self):
		ham = self.makeBogoMat()
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		return csi(vec[:, -1])
	
	def iprHighestStatefd(self):
		ham = self.makeBogoMat()
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		vec = self.norm(vec)
		return iprfd(vec[:, -1])
	
	def csiHighestStatefd(self):
		ham = self.makeBogoMat()
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		vec = self.norm(vec)
		return csifd(vec[:, -1])
	
	def LIPRDisr(self):
		iprvec = []
		# Store the original value of self.nb
		original_nb = self.nb
		for i in range( self.Ndisr ):
			# force that sum is Ntubes
			offset = np.sum(self.nb) - self.Ntubes 
			self.nb = self.nb - offset/self.Ntubes
			# get the ipr
			iprvec.append( self.iprLowestState()  )
			# Reset self.nb to its original value
			#self.nb = original_nb
		
		return np.mean(iprvec)
	
	def LCSIDisr(self):
		csivec = []
		# Store the original value of self.nb
		original_nb = self.nb
		for i in range( self.Ndisr ):
			# force that sum is Ntubes
			offset = np.sum(self.nb) - self.Ntubes 
			self.nb = self.nb - offset/self.Ntubes
			# get the csi
			csivec.append( self.csiLowestState()  )
			# Reset self.nb to its original value
			#self.nb = original_nb
		
		return np.mean(csivec)
	
	def LIPRDisrfd(self):
		iprvec = []
		# Store the original value of self.nb
		original_nb = self.nb
		for i in range( self.Ndisr ):
			# force that sum is Ntubes
			offset = np.sum(self.nb) - self.Ntubes 
			self.nb = self.nb - offset/self.Ntubes
			# get the ipr
			iprvec.append( self.iprLowestStatefd()  )
			# Reset self.nb to its original value
			#self.nb = original_nb
		
		return np.mean(iprvec)
	
	def LCSIDisrfd(self):
		csivec = []
		# Store the original value of self.nb
		original_nb = self.nb
		for i in range( self.Ndisr ):
			# force that sum is Ntubes
			offset = np.sum(self.nb) - self.Ntubes 
			self.nb = self.nb - offset/self.Ntubes
			# get the csi
			csivec.append( self.csiLowestStatefd()  )
			# Reset self.nb to its original value
			#self.nb = original_nb
		
		return np.mean(csivec)
	
	def MIPRDisr(self):
		iprvec = []
		# Store the original value of self.nb
		original_nb = self.nb
		for i in range( self.Ndisr ):
			# force that sum is Ntubes
			offset = np.sum(self.nb) - self.Ntubes 
			self.nb = self.nb - offset/self.Ntubes
			# get the ipr
			iprvec.append( self.iprMidState()  )
			# Reset self.nb to its original value
			#self.nb = original_nb

		return np.mean(iprvec)
	
	def MCSIDisr(self):
		csivec = []
		# Store the original value of self.nb
		original_nb = self.nb
		for i in range( self.Ndisr ):
			# force that sum is Ntubes
			offset = np.sum(self.nb) - self.Ntubes 
			self.nb = self.nb - offset/self.Ntubes
			# get the csi
			csivec.append( self.csiMidState()  )
			# Reset self.nb to its original value
			#self.nb = original_nb

		return np.mean(csivec)
	
	def MIPRDisrfd(self):
		iprvec = []
		# Store the original value of self.nb
		original_nb = self.nb
		for i in range( self.Ndisr ):
			# force that sum is Ntubes
			offset = np.sum(self.nb) - self.Ntubes 
			self.nb = self.nb - offset/self.Ntubes
			# get the ipr
			iprvec.append( self.iprMidStatefd()  )
			# Reset self.nb to its original value
			#self.nb = original_nb

		return np.mean(iprvec)
	
	def MCSIDisrfd(self):
		csivec = []
		# Store the original value of self.nb
		original_nb = self.nb
		for i in range( self.Ndisr ):
			# force that sum is Ntubes
			offset = np.sum(self.nb) - self.Ntubes 
			self.nb = self.nb - offset/self.Ntubes
			# get the csi
			csivec.append( self.csiMidStatefd()  )
			# Reset self.nb to its original value
			#self.nb = original_nb

		return np.mean(csivec)

	def HIPRDisr(self):
		iprvec = []
		# Store the original value of self.nb
		original_nb = self.nb
		for i in range( self.Ndisr ):
			# force that sum is Ntubes
			offset = np.sum(self.nb) - self.Ntubes 
			self.nb = self.nb - offset/self.Ntubes
			# get the ipr
			iprvec.append( self.iprHighestState()  )
			# Reset self.nb to its original value
			#self.nb = original_nb

		return np.mean(iprvec)
	
	def HCSIDisr(self):
		csivec = []
		# Store the original value of self.nb
		original_nb = self.nb
		for i in range( self.Ndisr ):
			# force that sum is Ntubes
			offset = np.sum(self.nb) - self.Ntubes 
			self.nb = self.nb - offset/self.Ntubes
			# get the csi
			csivec.append( self.csiHighestState()  )
			# Reset self.nb to its original value
			#self.nb = original_nb

		return np.mean(csivec)
	
	def HIPRDisrfd(self):
		iprvec = []
		# Store the original value of self.nb
		original_nb = self.nb
		for i in range( self.Ndisr ):  
            # force that sum is Ntubes
			offset = np.sum(self.nb) - self.Ntubes
			self.nb = self.nb - offset/self.Ntubes
            # get the ipr
			iprvec.append( self.iprHighestStatefd()  )
            # Reset self.nb to its original value
			self.nb = original_nb
			
		return np.mean(iprvec)
            
	
	def HCSIDisrfd(self):
		csivec = []
		# Store the original value of self.nb
		original_nb = self.nb   
		for i in range( self.Ndisr ):
            # force that sum is Ntubes
			offset = np.sum(self.nb) - self.Ntubes
			self.nb = self.nb - offset/self.Ntubes
			# get the csi
			csivec.append( self.csiHighestStatefd()  )
            # Reset self.nb to its original value
			#self.nb = original_nb

		return np.mean(csivec)
	
	def iprAllStates(self):
		ham = self.makeBogoMat()
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		iprv = [ipr( vec[:, i] ) for i in range(vec.shape[1])]
		#print(f'val:{val}')
		#print(f'iprv:{iprv}')
		return [val, iprv]
	
	def iprAllStatesfd(self):
		ham = self.makeBogoMat()
		val, vec = self._valvec(ham, self.sparseAlgo[1], self.sparseAlgo[2] )
		vec = self.norm(vec)
		iprv = [iprfd( vec[:, i] ) for i in range(vec.shape[1])]
		#print(f'val:{val}')
		#print(f'iprv:{iprv}')
		return [val, iprv]
	
	def IPRAllDisr(self):
		iprvec = []
		iprval = []
		# Store the original value of self.nb
		original_nb = self.nb
		for i in range( self.Ndisr ):
			# force that sum is Ntubes
			offset = np.sum(self.nb) - self.Ntubes 
			self.nb = self.nb - offset/self.Ntubes
			# get the ipr and the corresponding energy
			iprvec.append( self.iprAllStates()[1]  )
			iprval.append( self.iprAllStates()[0]  )
			#self.nb = original_nb
			
		return np.mean(iprval, axis=0),np.mean(iprvec, axis=0)
	
	def IPRAllDisrfd(self):
		iprvec = []
		iprval = []
		# Store the original value of self.nb
		original_nb = self.nb
		for i in range( self.Ndisr ):
			# force that sum is Ntubes
			offset = np.sum(self.nb) - self.Ntubes 
			self.nb = self.nb - offset/self.Ntubes
			# get the ipr and the corresponding energy
			iprvec.append( self.iprAllStatesfd()[1]  )
			iprval.append( self.iprAllStatesfd()[0]  )
			#self.nb = original_nb
			
		return np.mean(iprval, axis=0),np.mean(iprvec, axis=0)
	
### VISCOSITY FUNCTIONS
	
	def eta(self, y, yp): 
		val_s, U, V = self.BogUV()
		int_k = 0

		for i in range(1,self.Ntubes + 1):
			for j in range(1,self.Ntubes + 1):
				term_uuvv = U[y-1,i-1]*np.conj(U[yp-1,i-1])*V[y-1,j-1]*np.conj(V[yp-1,j-1])
				term_uvvu = U[y-1,i-1]*np.conj(V[yp-1,i-1])*V[y-1,j-1]*np.conj(U[yp-1,j-1])
				int_k += -2*(self.kx**2)*(term_uuvv - term_uvvu)

		return int_k
	
	def eta0(self, y, yp): 
		val_s, U, V = self.BogUV()
		int_k = 0

		for i in range(1,self.Ntubes + 1):
			for j in range(1,self.Ntubes + 1):
				term_uuvv = U[y-1,i-1]*np.conj(U[yp-1,i-1])*V[y-1,j-1]*np.conj(V[yp-1,j-1])
				term_uvvu = U[y-1,i-1]*np.conj(V[yp-1,i-1])*V[y-1,j-1]*np.conj(U[yp-1,j-1])
				int_k += -2*(self.kx**2)*(term_uuvv - term_uvvu)*(val_s[i-1] + val_s[j-1] )

		return int_k
	
	def eta0_lin(self, y, yp): 
		val_s, U, V = self.BogUV()
		int_k = 0

		for i in range(1,self.Ntubes + 1):
			for j in range(1,self.Ntubes + 1):
				term_uu = U[y-1,i-1]*np.conj(U[yp-1,i-1])
				term_vv = V[y-1,j-1]*np.conj(U[yp-1,j-1])
				term_uv = U[y-1,i-1]*np.conj(V[yp-1,i-1])
				term_vu = V[y-1,j-1]*np.conj(U[yp-1,j-1])
				if self.NN_int:
					if (abs(i-j)<2):
						int_k += -2*self.Ud*(math.sqrt(self.nb[i]*self.nb[j]))*(self.kx**2)*(term_uu + term_vv + term_uv + term_vu)
				else: #1/x^3 dipolar interaction
					dipk = 2*self.kx*kn(1,abs(i-j)*self.kx)/(abs(i-j))
					int_k += -2*(self.kx**2)*self.Ud*(math.sqrt(self.nb[i]*self.nb[j]))*dipk*(term_uu + term_vv + term_uv + term_vu)
			else:
				int_k += 0

		return int_k
	
	def etat(self, y, yp, t): 
		val_s, U, V = self.BogUV()
		int_k = 0

		for i in range(1,self.Ntubes + 1):
			for j in range(1,self.Ntubes + 1):
				term_uuvv = U[y-1,i-1]*np.conj(U[yp-1,i-1])*V[y-1,j-1]*np.conj(V[yp-1,j-1])
				term_uvvu = U[y-1,i-1]*np.conj(V[yp-1,i-1])*V[y-1,j-1]*np.conj(U[yp-1,j-1])
				int_k += -2*(self.kx**2)*(term_uuvv - term_uvvu)*np.sin((val_s[i-1] + val_s[j-1] )*t)

		return int_k


	def etaom(self, y, yp, om, gamma): 
		val_s,U,V = self.BogUV()
		int_k = 0

		for i in range(1,self.Ntubes + 1):
			for j in range(1,self.Ntubes + 1):
				fc_plus = 1/(om + val_s[i-1] + val_s[j-1] + 1j*gamma ) #plemelj-sokhotski formula
				fc_mnus = 1/(om - val_s[i-1] - val_s[j-1] + 1j*gamma ) #plemelj-sokhotski formula
				int_k += self.visc_k_ij_all(y,yp)*(fc_plus-fc_mnus)

		return int_k