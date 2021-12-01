import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))

#X = np.matrix("0.05 0.01 ; 0.1 0.99")
X = np.matrix("0.05 ; 0.1")
W = np.matrix("0.15 0.2 ; 0.25 0.3")

W = np.matrix("0.1498 0.1996 ; 0.2498 0.2995")

y = np.matrix('0.01;0.99')
U = np.matrix("0.4 0.45 ; 0.5 0.55")

U = np.matrix("0.3589 0.4086 ; 0.5113 0.5615")


b1 = np.matrix("0.35 ; 0.35")

b2 = np.matrix("0.6 ; 0.6")


print('Calculating X*W1\n')
print(W@X)
print('\n')
print('Calculating X*W1+b1\n')
Z1 = W@X+b1
print(Z1)
print('\n')
print('Applying sigmoid\n')
print(sigmoid(Z1))
Z1 = sigmoid(Z1)
print('\n')
print('Calculating Z1*U\n')
print(U@Z1)
print('\n')
print('Calculating Z1*U+b2\n')
Z2 = U@Z1+b2
print(Z2)
print('\n')
print('Applying sigmoid\n')
print(sigmoid(Z2))
#print('Error')