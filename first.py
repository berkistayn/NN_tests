import numpy as np
import matplotlib.pyplot as plt
import time


class fundamentals(object):
    def __init__(self):
        # n: number of features
        # m: number of training examples

        # data:
        self.trainIn  = None   # m*n+1 matrix
        self.trainOut = None   # m*1 vector
        self.testIn  = None
        self.testOut = None

        # computational parameters
        self.start_time = time.time()
        self.iterations = 0
        self.error = np.array([])

        self.features = None   # n+1*1 vector

        self.IsTrained = False
        self.learning_rate = None
        self.regularization_parameter = None

    def setTrainData(self, train_in, train_out):
        self.trainOut = train_out
        self.trainIn = np.vstack((np.ones([1, train_in.shape[1]]), train_in))  # add the X_0 = [1; 1; ... ; 1]

    def setTestData(self, test_in, test_out):
        self.testIn  = np.array(test_in)
        self.testOut = np.array(test_out)

    def countLastIteration(self):
        self.iterations = self.iterations + 1

    def hypothesis(self, i):
        #result = np.transpose(self.features[i]) * np.transpose([self.trainIn[:, i]])  # same with below
        #print('features vector: ', np.transpose(self.features))
        #print('training vector: ', self.trainIn[:, i])
        res =  np.matmul(self.features, self.trainIn[:, i])
        #print(res)
        return res


class Compute(fundamentals):

    def setLearningRate(self, new_LR):
        self.learning_rate = new_LR

    def timePassed(self):
        return time.time() - self.start_time

    def sigmoid(self, x):
        y = 1/(1 + np.exp(-x))
        return y

    def GD_regularized(self):
        RP = self.regularization_parameter

        if isinstance(RP, float) or isinstance(RP, int) is not True:
            raise TypeError('Tried to use regularization before assigning a regularization parameter.')

        pass

    def scaleFeatures(self):
        pass

    def solveNormalEqn(self):
        pass


class Visualize(fundamentals):

    # computational plots:
    def __init__(self):
        super().__init__()
        self.fig, self.ax = plt.subplots()

    def plotError_vs_Iteration(self):
        pass

    def plotCostFunction(self):
        pass

    # analysis plots:

    def ONTOP_plotTrain(self):
        print(self.trainIn[1, :])
        print(self.trainOut)
        self.ax.scatter(self.trainIn[1, :], self.trainOut)

    def ONTOP_plotFit(self, delay=0.01):
        lims = [np.min(self.trainIn[1, :]), np.max(self.trainIn[1, :])]
        plot_x = np.linspace(lims[0], lims[1], num=100)
        plot_y = self.fit(plot_x)
        self.ax.plot(plot_x, plot_y)
        plt.pause(delay)

    def fit(self, X):
        rang = len(X)
        fit = np.zeros(rang)
        X = np.vstack((np.ones([1, len(X)]), X))
        for i in range(0, rang):
            fit[i] = np.matmul(self.features, X[:, i])

        return fit

    def ONTOP_plotTest(self):
        pass


class LinearRegression(Compute, Visualize):

    def GD_simple(self, max_iteration = 1000, err_tresh = 0.01, IsVisual=True):
        self.iterations = 0
        self.IsTrained = False
        del_prev = 0  # relative change in descents are recorded.

        LR = self.learning_rate
        Y = self.trainOut
        X = self.trainIn

        m = len(self.trainOut)
        n = len(self.features)

        HouseModel.ONTOP_plotTrain()
        while self.IsTrained is False and self.iterations < max_iteration:
            features_old = self.features.copy()
            features_new = np.zeros(n)
            for j in range(0, n):
                del_cost = 0
                for i in range(0, m):
                    '''
                    print('hypo: ', self.hypothesis(i))
                    print('y: ', Y)
                    print('y[i]: ', Y[i])
                    print('X[j, i]: ', X[j][i])
                    '''
                    del_cost = del_cost + (self.hypothesis(i) - Y[i]) * X[j][i]
                features_new[j] = features_old[j] - LR * 1/m * del_cost
                if abs((del_prev - del_cost)/del_cost) < err_tresh:
                    self.IsTrained = True
                del_prev = del_cost.copy()
            self.features = features_new
            self.countLastIteration()
            if IsVisual:
                self.ONTOP_plotFit()
                #time.sleep(0.002)
        plt.show()

    def createFeatures(self):
        #self.features = np.random.rand(self.trainIn.shape[1], 1)
        self.features = np.random.rand(self.trainIn.shape[0])

    def costFunction_SEF(self):  # squared error function
        pass


class LogisticRegression(Compute, Visualize):

    def costFunction_log(self):
        return 'yo'


# multi variable
'''
train_IN = np.array([
                     [60, 65, 70, 85, 90, 110, 130, 150, 180, 190],
                     [25,  0, 30, 10, 12,   0,  80,   25, 10,  10]
                    ]).reshape(2, 10)
train_IN = np.array([60, 65, 70, 85, 90, 110, 130, 150, 180, 190]).reshape(1, 10)

# notscaled data - hard to manage.
train_IN = np.array([60, 65, 70, 85, 90, 110, 130, 150, 180, 190]).reshape(1, 10)
house_price = np.array([1400, 1500, 1500, 1800, 1600, 2400, 2700, 3100, 2800, 3000])
'''

### PLOT DIFFERENT FITS ON 1-D DATA
# ---------------------------------

# Using randomly scaled data
train_IN = np.array([0.1, 0.15, 0.22, 0.33, 0.44, 0.77, 0.78, 0.83, 0.95, 0.99,
                     1.44, 1.5, 1.66, 1.68, 1.71, 1.77, 1.78, 1.83, 1.95, 1.99]).reshape(1, 20)

#house_price = np.array([1400, 1500, 1500, 1800, 1600, 2400, 2700, 3100, 2800, 3000])  # trainingIn_Y
house_price = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.71, 0.72, 0.73, 0.74, 0.75,
                        0.66, 0.65, 0.68, 0.64, 0.65, 0.671, 0.672, 0.673, 0.674, 0.675])


# create the model
HouseModel = LinearRegression()

# set the model
HouseModel.setTrainData(train_IN, house_price)

start_time = time.time()
i = 0
while True:
    # train
    HouseModel.start_time = time.time()
    HouseModel.setLearningRate(0.03)
    HouseModel.createFeatures()
    HouseModel.GD_simple(max_iteration=5000, err_tresh=0.0001)

    # the results
    print('It took ' + str(HouseModel.timePassed()) + ' seconds to finish.')
    print('It took ' + str(HouseModel.iterations) + ' iterations to finish.')
    #HouseModel.ONTOP_plotTrain()
    #HouseModel.ONTOP_plotFit()
    #HouseModel.plotError_vs_Iteration()
    i = i + 1

#plt.show()

# ---------------------------------
###



