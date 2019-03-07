"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
#import numpy as np
import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import  PCA
#data_x=np.genfromtxt('mnist_train.csv',delimiter=',')

# Third-party libraries
class CrossEntropyCost( ):

    
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

   



def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((2)).tolist()
    
    e[int(j)] = 1.0
    return e

class ACT:
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    def relu(x):
        return np.maximum(0,x)
    def tanh(x):
        return 2*ACT.sigmoid(x)-1
        
    def swish(x):
        return x*ACT.sigmoid(x)
    def softmax(x):
        x=x-x.max(axis=1,keepdims=True)
        z=np.exp(x)
        return z/np.sum(z,axis=1,keepdims=True)
class ACTD:
    def sigmoid(x):
        a=ACT.sigmoid(x)
        return a*(1-a)
    def relu(x):
        
        return (np.sign(x)>=0)
    def tanh(x):
        return 1-(ACT.tanh(x))**2
        
    def swish(x):
        a=ACT.swish(x)
        b=ACT.sigmoid(x)
        return a+b*(1-a)
    def softmax(x):
       a=ACT.softmax(x)
       return a*(1-a)
def waitbias(ip,op,function='relu',mode='gaussian'):
    a=1/(ip+op)**0.5
    if function in ('sigmoid','softmax'):
        r=6**0.5
        s=2**0.5
    elif function=='tanh':
        r=4*6**0.5
        s=4*2**0.5
    else:
        r=12**0.5
        s=2
        #print("X")
    r=r*a
    s=r*s
    #print(r,s)
    if mode =='uniform':
        return 2*r*np.random.random((op,ip))-r,2*r*np.random.random((1,op))-r
    elif mode=='gaussian':
        return 2*r*np.random.randn(op,ip)*s,2*r*np.random.randn(1,op)*s
    else:
        raise Exception('Code should be unreachable')
        
        
        

class Network(object):

    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        #self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights=[]
        self.biases=[]
        for i in range(1,len(sizes)):
            a,b= (waitbias(sizes[i-1],sizes[i],'relu',mode='gaussian'))
            self.weights.append(a)
            self.biases.append(b)
            
      

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        inpt=[]
        outpt=[]
        inpt.append(a)
        j=0
        for b, w in zip(self.biases, self.weights):
            #print(b.shape,w.shape)
            zz=(np.dot(a,w.T)+b)
            inpt.append(zz)
            
            
            if j==len(self.sizes)-2: a=ACT.softmax(zz)
                       
            else:      a=ACT.relu(zz)
                      
            j=j+1
            
        #print(a)
        return inpt,a
    def predictor(self,a):
        j=0
        for b, w in zip(self.biases, self.weights):
            #print(b.shape,w.shape)
            zz=(np.dot(a,w.T)+b)
            if j==len(self.sizes)-2:
                      a=ACT.softmax(zz)
                       
            else:
                       a=ACT.relu(zz)
                       
            j=j+1
            
            
            #a=ACT.swish(zz)
            
        #print(a)
        return a

    def SGD (self,training_data,train_label,epochs, mini_batch_size, eta,test_data=None,test_label=None):
    
        if len(test_data)>0:
            n_test = len(test_data)
        
        
        n = len(training_data)
        #print(n)
        error=[]
        epo=[]
        for j in range(epochs):
            zzz=np.random.permutation(len(training_data))
            training_data=training_data[zzz]
            train_label=train_label[zzz]
            for k in range(0,len(training_data),mini_batch_size):
                X_=training_data[k:k+mini_batch_size]
                Y_=train_label[k:k+mini_batch_size]
                #print(Y_)
                #return
                z,a=self.feedforward(X_)
                #print(a)
                #return
                
                self.back_(z,a,Y_,eta)
                
                #self.update_mini_batch(z,a, eta)
            if len(test_data)>0:
                test,tt,t=self.evaluate(test_data)
                #x,y=test[0],test[1]
                #error.append(CrossEntropyCost.fn(x,y))
                #epo.append(j)
                
                #print(x,y)
                xxt=t*100.0
                xxt/=n_test
                epo.append(xxt)
                print ("Epoch {0}: {1} / {2},***accuracy___{3}%".format(
                    j, t, n_test,xxt))
                
            else:
                print ("Epoch {0} complete".format(j))
        return epo

    def back_(self,z,a,y,eta):
        delta=0
        n=len(self.sizes)
        xx=len(self.weights)
        #print(xx)
        for i in range(n-1,0,-1):
            if i==n-1:
                cost=self.cost_derivative(a,y)
                act=ACTD.softmax(z[i])
                delta=act*cost
                #print(delta.shape)
            else:
                #print(self.weights[i].shape)
                cost=np.dot(delta,self.weights[i])
                act=ACTD.relu(z[i])
                delta=act*cost
                #print(delta.shape)
            #print(len(y))
            dw=eta*np.dot(delta.T,ACT.relu(z[i-1]))
            db=eta*np.mean(delta,axis=0)
            #print(db.shape)
            self.weights[i-1]-=dw
            self.biases[i-1]-=db
        

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        x1=[]
        y1=[]
        for (x,y) in test_data:
            x1.append(np.argmax(self.predictor(x)));
            y1.append(y[0])
            
            
            
        test_results = [(np.argmax(self.predictor(x)), y)
                        for (x, y) in test_data]
        #print(test_results)
        #plt.plot(test_results,test_data[1])
        #plt.show()

        return x1,y1,sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


import glob
from PIL import Image
import numpy as np
#
#import io
import sklearn
import skimage
from skimage import io
#import skimage.io as io
filename='final_modelcatdog.sav'
fn='scalar'




#path=""
x=[]
y=[]
#if file=="
images=glob.glob("/home/clabuser/swyam/Cat-Dog/cat/*.jpg")

for image in images:
    img = skimage.io.imread(image, as_gray=True)
    img=skimage.transform.resize(img,(28,28))
    img=np.ravel(img)
    #img=np_r[0,img]
    #print(img)
    
    x.append(img)
    y.append(0)
images=glob.glob("/home/clabuser/swyam/Cat-Dog/dog/*.jpg")
for image in images:
    img = skimage.io.imread(image, as_gray=True)
    img=skimage.transform.resize(img,(28,28))
    img=np.ravel(img)
    y.append(1)
    #img=np_r[1,img]
    #print(img)
    
    x.append(img)
x=np.array(x)
x=np.c_[np.array(y),x]

data_x=x

#print(data_x)
   
#print(x)
    #img = Image.open(image)
    #img1 = img.resize(50,50)
    #img1.save("newfolder\\"+image) 


data_x=data_x
train_x=data_x[:,:1]
data_x=np.delete(data_x,0,1)
#pca=PCA(n_components=784)
scaler = StandardScaler()
data=scaler.fit_transform(data_x)
#size=[data_x.shape[1],data_x.shape[1]*2-100,
#size=[[784,300,2],[784,400,2],[784,500,2],[784,200,100,2],[784,300,100,2],[784,400,100,2],[784,1500,600,100,2],[784,1500,700,100,2],[784,1500,800,100,2]]
size=[784,1500,600,100,2]
if int((sklearn.__version__).split(".")[1]) < 18:
                from sklearn.cross_validation import train_test_split
                 
                
else:
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(data,
                                                                    train_x,
                                                                    test_size = 0.25,
                                                                    random_state=0)
                (X_train, valData, y_train, valLabels) = train_test_split(X_train, y_train,test_size=0.05, random_state=84)
train_label=[]
test_label=[]
for i in range(len(y_train)):
    train_label.append(vectorized_result(y_train[i]))
train_label=np.asarray(train_label)
#print(train_label)
for i in range(len(y_test)):
   test_label.append(vectorized_result(y_test[i]))
#X_train=list(zip(X_train,train_label))
V_test=list(zip(valData,valLabels))
X_test=list(zip(X_test,y_test))
accuracy=[]
#layer_count=[]
micro=[]
macro=[]
net=Network(size)
cd=net.SGD(X_train,train_label,100,50,0.00009,V_test,test_label)
ll=np.arange(100)
x1,y1,avr=Network.evaluate(net,X_test)
accuracy.append(accuracy_score(y1, x1))
plt.plot(ll,cd)
plt.savefig('task_3_4_2 uniform vs layer_count.png')
layer_count=[600,700,800]
"""
#for i in range(6,9):
        #layer_count.append(i+1)
        net=Network(size[i])
        #print(size[i])

        net.SGD(X_train,train_label,100,50,0.00009,V_test,test_label)


        x1,y1,avr=Network.evaluate(net,X_test)
        #print(test_[0])
        accuracy.append(accuracy_score(y1, x1))
        #print(accuracy[i])
        micro.append(f1_score(y1, x1, average='micro'))
        macro.append(f1_score(y1, x1, average='macro'))
"""
pickle.dump(net,open(filename,'wb'))
pickle.dump(scaler,open(fn,'wb'))
"""
plt.plot(layer_count,accuracy)
plt.savefig('task_2_1_3 accuracy vs layer_count.png')
plt.plot(layer_count,micro)
plt.savefig('task_2_1_3 micro vs layer_count.png')
plt.plot(layer_count,macro)
plt.savefig('task_2_1_3 macro vs layer_count.png')"""

