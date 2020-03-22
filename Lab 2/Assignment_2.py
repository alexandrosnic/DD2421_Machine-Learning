import numpy , random , math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as ptch


# Kernel function
def kernel(x1, y1, kernel_type):
    if kernel_type == "linear":
        x1 = numpy.transpose(x1)
        return numpy.dot(x1, y1)
    elif kernel_type == "polynomial":
        x1 = numpy.transpose(x1)
        return (numpy.dot(x1, y1)+1) ** 3
    elif kernel_type == "RBF":
        sigma = 3.3
        #return math.exp(-1*(numpy.linalg.norm(x1-y1))/(2*sigma**2))
        return math.exp(-1 * ( math.sqrt((x1[0] - y1[0]) ** 2 + (x1[1] - y1[1]) ** 2) / (2*sigma**2)) )

############ Basic code ##############

# zerofun is a function which calculates the value which should be constrained to zero
def zerofun(alpha):
    constr = [None] * 40
    for i in range(N):
        constr[i] = numpy.dot(alpha[i], targets[i])
    sum_constr = numpy.sum(constr)
    return sum_constr


# Objective is a function , which takes a vector  as argument and returns a scalar value
def objective(alpha):
    L = 0.5 * numpy.dot(alpha, numpy.dot(p_matrix, alpha)) - numpy.sum(alpha)
    return L

def indicator(x, y):
    sum_ind = 0
    new_point = numpy.array([x, y])
    for i in range(len(alpha)):
        sum_ind = sum_ind + alpha[i]*targets[i] * kernel(new_point, inputs[i], kernel_type)
    ind = sum_ind - b
    return ind


########### Tasks: ############

# 1. Try all the kernels, find a suitable one
# 2. Implement the objective function. Make use of global P matrix
# 3. Implement zerofun function
# 4. Call minimize
# 5. Pick non zero alpha with threshold = 10^-5 and save them with their corresponding xi, ti
# 6. Calculate b
# 7. Implement indicator function to classify new points



if __name__ == "__main__":
    kernel_type = "polynomial"

    ######### Generate data ############
    classA=numpy.concatenate((numpy.random.randn(10,2) * 0.2+[1.5,0.5],numpy.random.randn(10,2) * 0.2+[-1.5,0.5]))
    classB=numpy.random.randn(20,2) * 0.2+[0.0,-0.5]
    inputs=numpy.concatenate((classA,classB))
    targets=numpy.concatenate((numpy.ones(classA.shape[0]), - numpy.ones(classB.shape[0])))
    # N is here the number of training samples
    N=inputs.shape[0] # Number of rows (samples)

    permute=list(range(N))
    random.shuffle(permute)
    inputs=inputs[permute,:]
    targets=targets[permute]
    # Or just use numpy.random.seed(100)

    # Bound is a list of pairs of the same length as thevector, stating the lower and upper bounds
    C = None
    bound = [(0,C) for b in range(N)] # if upper and lower bounds
    #bound=[(0,None) for b in range(N)] #if only lower bound

    # Start is a vector with the initial guess of the vector
    start = numpy.zeros(N)

    # Initialize P matrix
    p_matrix = numpy.zeros([N,N])

    for i in range(N):
        for j in range(N):
            p_matrix[i][j] = (targets[i] * targets[j] * kernel(inputs[i], inputs[j], kernel_type))


    # Minimize function will find the vector which minimizes the function objective within the bounds B and the constraints XC.
    ret = minimize( objective , start , bounds = bound, constraints = {'type':'eq', 'fun':zerofun} )
    alpha = ret ["x"]
    threshold = 1*10**-5
    positives  = [[alpha[i], i] for i in range(len(alpha)) if numpy.abs(alpha[i]) >  threshold ]
    sv = list(zip(*positives))[0] # Support vectors alpha
    num = list(zip(*positives))[1] # Indeces of the support vectors
    targets_sv = [targets[num[i]] for i in range(len(num))] # Targets of the corresponding support vectors
    inputs_sv = [inputs[num[i]] for i in range(len(num))] # Inputs of the corresponding support vectors
    sum_b = 0
    for i in range(len(alpha)):
        sum_b = sum_b + alpha[i] * targets[i] * kernel(inputs_sv[1], inputs[i], kernel_type)
    b = sum_b - targets_sv[1]


    ########## Visualization ###########

    plt.plot([p[0] for p in classA], [p[1] for p in classA], "b+" )
    plt.plot([p[0] for p in classB], [p[1] for p in classB], "r." )


    plt.axis("equal") # For same scale on both axes

    xgrid = numpy.linspace(-5,5)
    ygrid = numpy.linspace(-4,4)
    grid = numpy.array([[indicator(x,y) for x in xgrid] for y in ygrid])
    plt.contour(xgrid, ygrid, grid, (-1.0, 0, 1.0), colors = ("red", "black", "blue"), linewidths = (1, 3, 1))
    
    blue_patch = ptch.Patch(color="blue", label="ClassA")
    red_patch = ptch.Patch(color="red", label="ClassB")

    plt.legend(handles=[blue_patch, red_patch])

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("svmplot.pdf") # Save a copy in a file
    plt.show() # Show the plot on the screen
    
    pass

