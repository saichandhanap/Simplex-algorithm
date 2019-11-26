import numpy as np

R = int(input("Enter the number of rows in matrix A:"))
C = int(input("Enter the number of columns in matrix A:"))

print("Enter the entries in a single line separated by space: ")

entries = list(map(int, input().split()))

matrix_A = np.array(entries).reshape(R,C) # matrix A

max_or_min = int(input("Enter maximize(1) or minimize(0):"))#1 for maximize and 0 for minimize

print("Enter cost coefficients for objective function:")

cost_coeff = []
dummy=[]

for i in range(0,R+1):
    dummy.append(0)

dummy = np.array(dummy)

for i in range(0,C):
    cost_coeff.append(int(input()))

cost_coeff_m = np.array(cost_coeff)
if(max_or_min==1):
    cost_coeff_m = (-1)*(cost_coeff_m)
cost_coeff_m = np.append(cost_coeff_m, dummy)
cost_coeff_m = np.matrix(cost_coeff_m)

vector_b = []

print("Enter b vector values:")

for i in range(0,R):
    vector_b.append(int(input()))
vector_b_m = np.matrix(vector_b)
vector_b_m = vector_b_m.transpose()

print("Enter L if constraints have < symbol or R if they have > symbol")

eq = input()

if(eq == "L" or eq == "l"):
    I = np.identity(R,dtype=int)
    matrix_A = np.concatenate((matrix_A,I),axis=1)
    matrix_A = np.concatenate((matrix_A, vector_b_m), axis = 1)
    matrix_A = np.concatenate((matrix_A, cost_coeff_m), axis = 0)
elif(eq == "R" or eq == 'r'):
    matrix_A = (-1)*(matrix_A)
    I = np.identity(R,dtype=int)
    vector_b_m = (-1)*vector_b_m
    matrix_A = np.concatenate((matrix_A,I),axis=1)
    matrix_A = np.concatenate((matrix_A, vector_b_m), axis = 1)
    matrix_A = np.concatenate((matrix_A, cost_coeff_m), axis = 0)
'''elif(eq == "E" or eq == "e"):
	matrix_A_copy = (-1)*matrix_A
	matrix_A = np.concatenate((matrix_A,matrix_A_copy),axis=0)
	I = np.identity(2*R,dtype=int)
	matrix_A = np.concatenate((matrix_A,I),axis=1)
	vector_b_m_copy = (-1)*vector_b_m
	vector_b_m = np.concatenate((vector_b_m,vector_b_m_copy),axis = 0)
	matrix_A = np.concatenate((matrix_A, vector_b_m), axis = 1)

	matrix_A = np.concatenate((matrix_A, cost_coeff_m), axis = 0)
	#print(matrix_A)'''





#print(cost_coeff)#print(vector_b)#print(vector_b_m)

print(matrix_A)
'''
if((matrix_A[-1,:]>=0).all()):
    p_prime = np.argmin(matrix_A[:,-1])
    p_prime = int(p_prime)
    temp = np.sum(matrix_A[:-1,:],axis=0)-1
    w = np.dot(matrix_A[-1,:],temp)
    q_prime_list = np.true_divide(w,matrix_A[p_prime,:])
    q_prime = np.argmin(q_prime_list)
    print("P Value:")
    print(p_prime)
    print("Q Value:")
    print(q_prime)'''

while ((matrix_A[-1,:]<0).any()):
    q = np.argmin(matrix_A[-1,:])
    q=int(q)
    print("Value of q is: ")
    print(q)

    p_list=[]
    for i in range(0,np.size(matrix_A,0)-1):
        if matrix_A[i,q]>0 :
            p_list.append(np.true_divide(matrix_A[i,-1],matrix_A[i,q]))

    #p_list = [np.true_divide(matrix_A[:-1,-1],matrix_A[:-1,q]) for i in matrix_A[:-1,q] if i>0] 
    if(len(p_list)==0):
    	print("Unbounded..")
    	break
    p = int(np.argmin(p_list))
    print("Value of p is: ")
    print(p)
    s = np.shape(matrix_A)
    new_matrix_A = np.zeros(s)
    #print(new_matrix_A)

    for i in range(0,s[0]):
        for j in range(0,s[1]):
            if(i!=p):
                temp1 = matrix_A[i,q]*matrix_A[p,j]
                temp2 = temp1/matrix_A[p,q]
                new_matrix_A[i,j] = matrix_A[i,j] - temp2
            elif(i==p):
                temp1 = matrix_A[p,j]/matrix_A[p,q]
                new_matrix_A[i,j] = temp1
    matrix_A = np.copy(new_matrix_A)

    '''if((matrix_A[-1,:]>=0).all()): #doubt..
        p_prime = np.argmin(matrix_A[:,-1])
        p_prime = int(p_prime)

        w = np.dot(matrix_A[-1,:],np.sum(matrix_A[:-1,:],axis=0)-1)
        q_prime_list = np.true_divide(w,matrix_A[p_prime,:])
        q_prime = np.argmin(q_prime_list)
        print("P Value:")
        print(p_prime)
        print("Q Value:")
        print(q_prime)'''

    
    print(new_matrix_A)
    print("---------")

#find solutions:x1,x2,x3,...
def check(i,matrix_A):
    for k in range(0,np.size(matrix_A,0)-1):
        if matrix_A[k,i]==1:
            return matrix_A[k,-1]
final =  np.zeros(np.size(matrix_A,1))
for i in range(0,np.size(matrix_A,1)):
    if matrix_A[-1,i]==0:
        final[i]=check(i,matrix_A)
for i in range(0,np.size(final)-1):
    if final[i]!=0:
        print("x"+str(i+1)+"="+str(final[i]))

if(max_or_min==1):
    x = (-1)*(matrix_A[-1,-1])
    print("Objective Function: " + str(x))
else:
    x = matrix_A[-1,-1]
    print("Objective Function: " + str(x)) 
