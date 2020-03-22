#!/usr/bin/env python
# coding: utf-8

# # Lab 3: Bayes Classifier and Boosting

# ## Jupyter notebooks
# 
# In this lab, you can use Jupyter <https://jupyter.org/> to get a nice layout of your code and plots in one document. However, you may also use Python as usual, without Jupyter.
# 
# If you have Python and pip, you can install Jupyter with `sudo pip install jupyter`. Otherwise you can follow the instruction on <http://jupyter.readthedocs.org/en/latest/install.html>.
# 
# And that is everything you need! Now use a terminal to go into the folder with the provided lab files. Then run `jupyter notebook` to start a session in that folder. Click `lab3.ipynb` in the browser window that appeared to start this very notebook. You should click on the cells in order and either press `ctrl+enter` or `run cell` in the toolbar above to evaluate all the expressions.
# 
# Be sure to put `%matplotlib inline` at the top of every code cell where you call plotting functions to get the resulting plots inside the document.

# ## Import the libraries
# 
# In Jupyter, select the cell below and press `ctrl + enter` to import the needed libraries.
# Check out `labfuns.py` if you are interested in the details.

# In[9]:


import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random
import math


# ## Bayes classifier functions to implement
# 
# The lab descriptions state what each function should do.

# In[10]:


# NOTE: you do not need to handle the W argument for this part!
# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    # TODO: compute the values of prior for each class!
    # ==========================
    Noccurences = np.zeros(Nclasses)
    #TotalOccur = len(labels)
    TotalOccur = 0.
    prior=[]
    # Iterate over both index and value
    for jdx,class_k in enumerate(classes):
        idx = labels==class_k # Returns a true or false with the length of y
        # Or more compactly extract the indices for which y==class is true,
        # analogous to MATLAB's find
        # idx = np.where(labels==class_k)[0]
        # xlc = X[idx,:] # Get the x for the class labels. Vectors are rows.
        Noccurences[jdx] = len(idx) * W[jdx]
        TotalOccur = TotalOccur + Noccurences[jdx]
    prior = Noccurences / TotalOccur 
    # ==========================

    return prior

# NOTE: you do not need to handle the W argument for this part!
# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))

    # TODO: fill in the code to compute mu and sigma!
    # ==========================
    # Assignment 1 and 4
    Nk=np.zeros(Nclasses)

    for k in classes:
        for i in range(0,Npts):
            if(labels[i]==k):
                mu[k] = mu[k]+ (X[i]*W[i])
                Nk[k] = Nk[k]+(1*W[i])

        mu[k] = mu[k] / Nk[k]

    for k in classes:
        for i in range(0,Npts):
            for j in range(0,Ndims):
                if(labels[i]==k):
                    sigma[k][j][j] = sigma[k][j][j] + (math.pow((X[i][j]-mu[k][j]),2))*W[i]

        sigma[k] = sigma[k] / Nk[k]
    # ==========================

    return mu, sigma

# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(X, prior, mu, sigma):

    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    # TODO: fill in the code to compute the log posterior logProb!
    # ==========================
    for k in range(Nclasses):
        for j in range(Npts):
            logProb[k][j] = - 0.5 * math.log(np.linalg.det(sigma[k])) - 0.5 * (np.dot((X[j] - mu[k]), np.dot(np.linalg.inv(sigma[k]), np.transpose((X[j] - mu[k]))))) + math.log(prior[k])

    # ==========================
    
    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb,axis=0)
    return h


# The implemented functions can now be summarized into the `BayesClassifier` class, which we will use later to test the classifier, no need to add anything else here:

# In[11]:


# NOTE: no need to touch this
class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


# ## Test the Maximum Likelihood estimates
# 
# Call `genBlobs` and `plotGaussian` to verify your estimates.

# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')

X, labels = genBlobs(centers=5)
mu, sigma = mlParams(X,labels)
plotGaussian(X,labels,mu,sigma)


# Call the `testClassifier` and `plotBoundary` functions for this part.

# In[13]:


testClassifier(BayesClassifier(), dataset='iris', split=0.7)


# In[14]:


testClassifier(BayesClassifier(), dataset='vowel', split=0.7)


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
plotBoundary(BayesClassifier(), dataset='iris',split=0.7)


# # Assignment 3
# Feature independence assumption is reasonable when we want to simplify our data in order to use Naive Bayes. 
# And also when the features of the dataset are not very correlated in between them => covariance is either 0 or 1
# In case we don't a big dataset then we can also work on non diagonal covariance matrices.
# 
# In iris dataset, there is a clear distinction between one class from the others. We may use boosting for the other two unstable / sensitive classifiers. Thus the weight of the unstable classifiers will be increased, the learning algorithm will focus on this, and will result in improving it

# ## Boosting functions to implement
# 
# The lab descriptions state what each function should do.

# In[16]:


# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)

        # TODO: Fill in the rest, construct the alphas etc.
        # ==========================
        
        delta = vote==labels
        error = np.inner(wCur.reshape(-1), 1.0 -delta)
        if error <0.001:
            error=0.001
        if error>0.5:
            error = 0.5

        alpha = 0.5 * (math.log(1-error) - math.log(error))
        alphas.append(alpha) # you will need to append the new alpha
        wCur = np.multiply(wCur.reshape(-1), np.exp(((1.0 - delta)-0.5)*2*alpha))
        wCur = wCur / float(np.sum(wCur))
        # ==========================
        
    return classifiers, alphas

# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))

        # TODO: implement classificiation when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================
        for t, classifier in enumerate(classifiers):
            h = classifier.classify(X)
            for ii in range(Npts):
                votes[ii,h[ii]] += alphas[t]
        # ==========================

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)


# The implemented functions can now be summarized another classifer, the `BoostClassifier` class. This class enables boosting different types of classifiers by initializing it with the `base_classifier` argument. No need to add anything here.

# In[17]:


# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


# ## Run some experiments
# 
# Call the `testClassifier` and `plotBoundary` functions for this part.

# In[18]:


testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)


# In[19]:


testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris',split=0.7)


# # Assignment 5
# 
# 1. Is there any improvement in classification accuracy? Why/why not?
# Yes, the accuracy in Iris dataset improved from 89% to 94.6% and for the vowel dataset from 64.7% to 71.6%.
# The improvement is because there was high uncertainty in one (or more) of the classes making the classifier weak. Using boosting, we focused to improve that classifier.
# 
# 2. Plot the decision boundary of the boosted classifier on iris and compare it with that of the basic. What differences do you notice? Is the boundary of the boosted version more complex?
# The boundary between the 0 and 1 class it's the same as before since it was quite distinct. However the boundary of 1-2 class is a bit more marginal. So after the boosting, the classifiers became more accurate and the boundary between the two more clear.
# 
# 3. Can we make up for not using a more advanced model in the basic classiffier (e.g. independent features) by using boosting?
# Boosting and advanced model are independent features. It may happen that the advanced model such as inependent features (Bayes) creates classifiers strong enough that the computational cost of using boosting is not worth it using it. However in our case, since we do have a weak classifier, boosting will make our classifiers stronger without important extra computational cost. 

# Now repeat the steps with a decision tree classifier.

# In[21]:


testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)


# In[22]:


testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)


# In[23]:


testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)


# In[24]:


testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)


# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')
plotBoundary(DecisionTreeClassifier(), dataset='iris',split=0.7)


# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')
plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)


# In[31]:


get_ipython().run_line_magic('matplotlib', 'inline')
plotBoundary(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)


# # Assignment 6
# 
# 1. Is there any improvement in classification accuracy? Why/why not?
# 
# 
# 2. Plot the decision boundary of the boosted classifier on iris and compare it with that of the basic. What differences do you notice? Is the boundary of the boosted version more complex?
# The boundaries in the decision tree case are less accurate than Bayes since the are not continuous (straight lines). When we boost the decision tree, the boundaries increase their resolution resulting in more accurate distinction, but still discrete.
# 
# 3. Can we make up for not using a more advanced model in the basic classifier (e.g. independent features) by using boosting?
# The same

# # Assignment 7
# 
# If you had to pick a classifier, naive Bayes or a decision tree or the boosted versions of these, which one would you pick?
# 
# * Outliers: Decision trees are high-variance models — that is, relatively modest changes in your data can yield large changes in the resulting tree. As a result they can be very sensitive to noise. => Better use Naive Bayes
# 
# * Irrelevant inputs: part of the feature space is irrelevant: The boosted version of any of the classifier methods will provide stronger classifiers, increasing the accuracy even with irrelevant input. A pruned DT is very tolerant to nonsense input.
# 
# * Predictive power: It highly depends on the dependence between the features of the data. For independent features, we can hardly find a better classifier than boosted Bayes. Decision trees can also give high accuracy, but because they have to be splitted in order to give higher accuracy, they will be more complex.
# 
# * Mixed types of data: binary, categorical or continuous features, etc.: NB can perform well in both categorical and continuous feature space.
# 
# * Scalability: the dimension of the data, D, is large or the number of instances, N, is large, or both.: The higher the dimensions, the higher the complexity of a NB, which can ultimately give poor performance, so in that case, the simpler DT is preferred.

# ## Bonus: Visualize faces classified using boosted decision trees
# 
# Note that this part of the assignment is completely voluntary! First, let's check how a boosted decision tree classifier performs on the olivetti data. Note that we need to reduce the dimension a bit using PCA, as the original dimension of the image vectors is `64 x 64 = 4096` elements.

# In[28]:


testClassifier(BayesClassifier(), dataset='olivetti',split=0.7, dim=20)


# In[29]:


testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='olivetti',split=0.7, dim=20)


# You should get an accuracy around 70%. If you wish, you can compare this with using pure decision trees or a boosted bayes classifier. Not too bad, now let's try and classify a face as belonging to one of 40 persons!

# In[30]:


get_ipython().run_line_magic('matplotlib', 'inline')
X,y,pcadim = fetchDataset('olivetti') # fetch the olivetti data
xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,0.7) # split into training and testing
pca = decomposition.PCA(n_components=20) # use PCA to reduce the dimension to 20
pca.fit(xTr) # use training data to fit the transform
xTrpca = pca.transform(xTr) # apply on training data
xTepca = pca.transform(xTe) # apply on test data
# use our pre-defined decision tree classifier together with the implemented
# boosting to classify data points in the training data
classifier = BoostClassifier(DecisionTreeClassifier(), T=10).trainClassifier(xTrpca, yTr)
yPr = classifier.classify(xTepca)
# choose a test point to visualize
testind = random.randint(0, xTe.shape[0]-1)
# visualize the test point together with the training points used to train
# the class that the test point was classified to belong to
visualizeOlivettiVectors(xTr[yTr == yPr[testind],:], xTe[testind,:])

