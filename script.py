import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    unique_y = np.unique(y)
    # print(unique_y)
    # print(type(unique_y))

    means = np.zeros((len(unique_y), X.shape[1]))
    for i in unique_y:
        temp_x = X[np.where(y == i)[0]]
        means[int(i) - 1] = temp_x.mean(axis=0)
    # print (means)
    covmat = np.cov(X.T)
    # print (covmat)

    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    covmats = []
    labels = np.unique(y)
    means = np.zeros([labels.shape[0], X.shape[1]])

    for i in range(labels.shape[0]):
        m = np.mean(X[np.where(y == labels[i])[0],], axis=0)
        means[i,] = m
        covmats.append(np.cov(np.transpose(X[np.where(y == labels[i])[0],])))

    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    g = 1 / np.sqrt((2 * np.pi ** means.shape[1]) * det(covmat))
    ll = np.zeros((Xtest.shape[0], means.shape[0]))
    for i in range(Xtest.shape[0]):
        for h in range(means.shape[0]):
            b = Xtest[i, :] - means[int(h) - 1]
            t = (-1 / 2) * np.dot(np.dot(b.T, inv(covmat)), b)
            ll[i, int(h) - 1] = g * np.e ** t

    ypred = []
    for row in ll:
        ypred.append(list(row).index(max(list(row))) + 1)
    # ypred = np.argmax(ll, axis=1)+1

    acc = 0
    for k in range(len(ypred)):
        if ypred[k] == ytest[k]:
            acc += 1
    # print(acc,len(ypred))
    acc = acc / len(ypred)
    ytest = ytest.flatten()
    ypred = np.array(ypred)

    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    ll = np.zeros((Xtest.shape[0], means.shape[0]))
    for i in range(Xtest.shape[0]):
        for h in range(means.shape[0]):
            index = int(h) - 1
            b = Xtest[i, :] - means[index]
            t = (-1 / 2) * np.dot(np.dot(b.T, inv(covmats[index])), b)
            g = 1 / np.sqrt((2 * np.pi ** means.shape[1]) * det(covmats[index]))
            ll[i, index] = g * np.e ** t

    ypred = []
    for row in ll:
        ypred.append(list(row).index(max(list(row))) + 1)

    acc = 0
    for k in range(len(ypred)):
        if ypred[k] == ytest[k]:
            acc += 1
    acc = acc / len(ypred)
    ytest = ytest.flatten()
    ypred = np.array(ypred)

    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD
    inv_of_xtransx = np.linalg.inv(np.dot(np.transpose(X), X))
    xtransy = np.dot(np.transpose(X), y)
    w = np.dot(inv_of_xtransx, xtransy)
    # print("OLERidge regression wt ",w)

    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    w = np.dot(np.linalg.inv(np.multiply(lambd, np.identity(X.shape[1])) + np.dot(np.transpose(X), X)),
               (np.dot(np.transpose(X), y)))

    # print(w)
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    N = Xtest.shape[0]
    xdotw = np.dot(Xtest, w)
    mse = (np.dot(np.transpose(ytest - xdotw), (ytest - xdotw))) / N

    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD
    w = w.reshape(w.size, 1)
    temp_err1 = y - np.dot(X, w)
    temp_err2 = 0.5 * lambd * np.dot(w.transpose(), w)

    error = 0.5 * np.dot(temp_err1.transpose(), temp_err1) + temp_err2

    err_grad1 = np.dot(np.dot(X.transpose(), X), w)
    err_grad2 = np.dot(X.transpose(), y)
    err_grad3 = lambd * w

    error_grad = (err_grad1 - err_grad2) + err_grad3
    error_grad = error_grad.flatten()

    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD
    N = x.shape[0]
    Xp = np.ones(shape=(N, p + 1))

    for i in range(0, N):
        for j in range(1, p + 1):
            Xp[i][j] = np.power(x[i], j)
    return Xp

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')


# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('QDA')

plt.show()

# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')


# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

w = learnOLERegression(X,y)
mle_t = testOLERegression(w,X,y)

w_i = learnOLERegression(X_i,y)
mle_ti = testOLERegression(w_i,X_i,y)

#print('MSE for training data without intercept '+str(mle_t))
#print('MSE for training data with intercept '+str(mle_ti))
# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
optimisedLambda_Train = list()
optimisedLambda_Test = list()
lambdaVsMse_Train = np.zeros(shape=(1,2))

for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    optimisedLambda_Train.append([lambd,mses3_train[i]])
    optimisedLambda_Test.append([lambd,mses3[i]])
    i = i + 1
    #print(lambd)
#print("Ridge regression wt ",w_l)
#print("MSE Train: \n", mses3_train)
#print("MSE Test: \n", mses3)

#print("Optimised Lambda Test Values ",optimisedLambda_Test)
#print("Optimised Lambda Train Values ", optimisedLambda_Train)

#print("Optimised Lambda Test  ", min(map(lambda x: x[1], optimisedLambda_Test )))
#print("Optimised Lambda Train  ", min(map(lambda x: x[1], optimisedLambda_Train )))

#print(min(map(lambda x: x[0], optimisedLambda_Test )))

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.show()

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
optimisedLambda_Train4 = list()
optimisedLambda_Test4 = list()

for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    optimisedLambda_Train4.append([lambd,mses4_train[i]])
    optimisedLambda_Test4.append([lambd,mses4[i]])
    i = i + 1

#print("Optimised Lambda Test  ",min(map(lambda x: x[1], optimisedLambda_Test4 )))
#print("Optimised Lambda Train  ",min(map(lambda x: x[1], optimisedLambda_Train4 )))
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(mses3)] # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

#print("Optimised Lambda Test  ",min(map(lambda x: x[1], mses5 )))
#print("Optimised Lambda Train  ",min(map(lambda x: x[1], mses5_train )))
#print("MSE Train: \n", mses3_train)
#print("MSE Test: \n", mses3)
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()

