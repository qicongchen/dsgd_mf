from pyspark import SparkContext, SparkConf
from scipy import sparse
from numpy import random, matrix, linalg, savetxt
import time
import itertools
import sys
import matplotlib.pyplot as plt
def tokenize(line):
	tokens=line.strip().split(",")
	return int(tokens[0])-1,int(tokens[1])-1,int(tokens[2])
def get_shape(path):
	fread=open(path, 'r')
	for line in fread:
		user,movie,rate=tokenize(line)
		if user not in users:
			users[user]=len(users)
		if movie not in movies:
			movies[movie]=len(movies)
	fread.close()
def create_matrix(path):
	#user to movie
	V=sparse.lil_matrix((m,n)) 
	fread=open(path, 'r')
	for line in fread:
		user,movie,rate=tokenize(line)
		V[users[user],movies[movie]]=rate
	fread.close()
	return V
#calculate epsilon
def eps(clock):
	return pow(100+clock,-beta)
############ transformers #####
def get_next_strata(s):
	s[1]=(s[1]+1)%d
	return s
def get_rows_cols(s):
	return (range(s[0], m, d),range(s[1], n, d))
def sgd(mtx):
	Vn=mtx["V"]
	Wn=mtx["W"]
	Hn=mtx["H"]
	rows,cols = Vn.nonzero()
	c=0
	for i,j in random.permutation(zip(rows,cols)):
		tmp=Vn[i,j]-(Wn[i,:]*Hn[:,j])[0,0]
		Wn_i=Wn[i,:]-eps(c+clock)*(-2*tmp*Hn[:,j].transpose()+2*lmda*Wn[i,:]/Vn[i,:].nnz)
		Hn[:,j]=Hn[:,j]-eps(c+clock)*(-2*tmp*Wn[i,:].transpose()+2*lmda*Hn[:,j]/Vn[:,j].nnz)
		Wn[i,:]=Wn_i
		c+=1
	clocka.add(len(rows))
	mtx.pop("V",None)
	return mtx
##################################
#########calculate mse############
def recon_error():
	error=0.0
	V_=W*H
	rows,cols = V.nonzero()
	for i,j in zip(rows,cols):
		tmp=V[i,j]-V_[i,j]
		error+=tmp*tmp
	error/=len(rows)
	return error
if __name__ == "__main__":
	#########create matrix########
	f=int(sys.argv[1])
	d=int(sys.argv[2])
	iters=int(sys.argv[3])
	lmda=float(sys.argv[5])
	beta=float(sys.argv[4])
	input_path=sys.argv[6]
	output_W_path=sys.argv[7]
	output_H_path=sys.argv[8]
	users=dict()
	movies=dict()
	get_shape(input_path)
	m=len(users)
	n=len(movies)
	V=create_matrix(input_path)
	#release memory
	users=None
	movies=None
	#################################
	########run dsgd#################
	#init spark context
	conf = SparkConf().setAppName("dsgd").setMaster("local[%d]" % d)
	sc = SparkContext(conf=conf)
	#init strata
	S= sc.parallelize([[i,v] for i,v in enumerate(random.permutation(d))])
	#init clock
	clocka=sc.accumulator(0)
	#init matrices
	W=matrix(random.rand(m,f))
	H=matrix(random.rand(f,n))
	#iterations
	for i in xrange(iters):
		#get rows and cols
		rcs=S.map(get_rows_cols).collect()
		#get reference to matrices
		mtxs=[]
		for j,rc in enumerate(rcs):
			mtx={"rows":rc[0],"cols":rc[1],"V":V.tocsr()[rc[0],:].tocsc()[:,rc[1]],"W":W[rc[0],:],"H":H[:,rc[1]]}
			mtxs.append(mtx)
		#parallelize matrices
		matrices=sc.parallelize(mtxs)
		#set clock
		clock=clocka.value
		#run sgd
		mtxs=matrices.map(sgd).collect()
		#update W and H
		for mtx in mtxs:
			W[mtx["rows"],:]=mtx["W"]
			H[:,mtx["cols"]]=mtx["H"]
		#find next strata
		S=S.map(get_next_strata)
	####################################
	# output result
	savetxt(output_W_path,W,delimiter=',')
	savetxt(output_H_path,H,delimiter=',')
	# error
	print 'Reconstruction MSE: %g\n' % recon_error()
