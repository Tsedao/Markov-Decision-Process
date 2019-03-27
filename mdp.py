# a little script to aim calculate markov decision process,
# given probability matrix P, reward martix R and duration n

import numpy as np
def product(probability, reward):
    result = np.zeros(shape=(probability.shape[0],1))
    for i in range(0,probability.shape[0]):
        result[i,0]= probability[i,:].dot(reward[i,:].transpose())
    return result

def MDP(tensor1,tensor2,n,f=0):

	def findProfits(tensor1, tensor2):
		shape = tensor1.shape
		profits = np.zeros(shape=(shape[1],1)) #initiate by a zero vector
		for i in range(0,shape[0]):
			temp = product(tensor1[i],tensor2[i])
			profits = np.concatenate((profits,temp),axis=1)
		return np.delete(profits,0,axis=1) #delele the first zero column

	profits = findProfits(tensor1, tensor2)
	maxProfit =np.max(profits, axis = 1) #find the max among different actions
	indexMaxProfit = np.argmax(profits,axis=1)
	print(indexMaxProfit,n)

	for i in range(0,indexMaxProfit.shape[0]):
		print('It suggested that during the %s period, if this period is in state %s action %s should be taken.' %(n,i+1,indexMaxProfit[i]+1))

	delta = maxProfit-f #find the increment in each year

	if n == 1:
		print(maxProfit)
		return maxProfit
	else:
		for i in range(0,tensor2.shape[0]): #refresh the reward matrix 
			tensor2[i] += delta
		return MDP(tensor1,tensor2,n-1,maxProfit)


# example
# A is probablity matrix, B is reward matrix
# Biggest return in three years is calculated 
A1=np.array([[1/2,1/2],[1/5,4/5]])
A2=np.array([[2/3,1/3],[1/2,1/2]])
A3=np.array([[2/3,1/3],[1/3,2/3]])
B1=np.array([[1600,800],[1600,800]])
B2=np.array([[1400,600],[1400,600]])
B3=B1*0.71
test1=np.array((A1,A2,A3),dtype='float64')
test2=np.array((B1,B2,B3),dtype='float64')
MDP(test1,test2,3)

# another example

# P1=np.array([[0.2,0.5,0.3],[0,0.5,0.5], [0,0,1]])
# P2=np.array([[0.3,0.6,0.1],[0.1,0.6,0.3], [0.05,0.4,0.55]])
# R1=np.array([[7,6,3],[0,5,1],[0,0,-1]])
# R2=np.array([[6,5,-1],[7,4,0],[6,3,-2]])
# tensor1 = np.array((P1,P2),dtype='float64')
# tensor2 = np.array((R1,R2),dtype='float64')
# MDP(tensor1,tensor2,3)

# C1=np.array([[0.6,0.4],[0.3,0.7]])
# C2=np.array([[0.7,0.3],[0.6,0.4]])
# C3=np.array([[0.6,0.4],[0.7,0.3]])
# D1=np.array([[80000,40000],[80000,40000]])
# D2=np.array([[72000,36000],[72000,36000]])
# D3=np.array([[75000,35000],[75000,35000]])
# test3=np.array((C1,C2,C3),dtype='float64')
# test4=np.array((D1,D2,D3),dtype='float64')
# MDP(test3,test4,3)




