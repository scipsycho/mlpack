import numpy as np
class linear_regression:

    def __init__(self,batch_size=0,epochs=100,learning_rate=0.001,tolerance=0.01,show_progress=True):
        """
        The function initiaizes the class

        Parameters
        ----------
        batch_size: int
                    It defines the number of data sets the alogrithm takes at ones to optimise the Parameters.
                    It may be the factor of the number of examples given to the algorithm in the fit function
                    Default Value is 0 which means it will compute all the data sets together.
        epochs:     int
                    It is the maximum number of times the algorithm is going to compute the whole data set available
                    for training.
                    Default Value is 100

        learning_rate: float
                    It is the learning rate of the machine learning algorithm.
                    Default Value is 0.001
        tolerance:  float
                    It defines the minimum improvement that the algorithm will tolerate i.e. if the parameters show a change
                    less than the value of tolerance, it assumes that the alogrithm is optimised to the maximum
                    Default Value is 0.01
        show_progress: Boolean
                    It controls whether the object will show the progress as output or not.
                    Default Value: True

        Returns
        -------
        Nothing

        """
        self.batch=batch_size
        self.epochs=epochs
        self.l_rate=learning_rate
        self.show_progress=show_progress
        self.tol=tolerance

    def fit(self,X,Y):
        """
        The function fits the training data set to the algorithm.

        Detailed Description
        --------------------
        The function takes on the input and actual ouput of the data set and optimises the parameters accordingly.

        Parameters
        ----------
        X:  numpy.ndarray
            It is the input data set. The number of columns define the number of dimensions in the input data.
            The number of rows defines the number of data sets avaiable for training.
            If there is only one dimension, it can also be a linear numpy.ndarray.

        Y:  numpy.ndarray
            It is the proposed output corresponding to the input given in any row of the input data set X.
            The number of rows defines the number of data sets avaiable for training.
            It can also be a linear numpy.ndarray.

        Returns
        -------

        Nothing

        Notes
        -----
        X.shape[0] must be equal to Y.shape[0] which is also the number of data sets avaiable for training.
        """
        self.note=X.shape[0]
        if self.batch is 0:
            self.batch=self.note
        if len(X.shape) is 1:
            X=X.reshape([X.shape[0],1])
        self.nod=X.shape[1]+1
        self.train_i=np.ones([self.note,self.nod])
        self.train_i[:,1:]=X
        self.train_o=Y
        self.parameters=np.random.random([self.nod,1])
        self.__start_gradient_descent__()

    def __GradDescent__(self,initial,final):
        """
        The function optimises the paramters according a specific subset of the data set available

        Parameters
        ----------

        initial:  int
                  It is the inital index of block of the data set being used.
        final:    int
                  It is the final index of block of the data set being used.

        Returns
        -------

        Nothing

        Notes
        -----
        initial should always be less than or equal to final. Also, final should always be less than the
        number of data sets avaiable
        """
        diff=(self.train_i[initial:final,:].dot(self.parameters).T-self.train_o[initial:final]).T
        diff=(diff*self.train_i[initial:final,:]).sum(axis=0)
        diff=diff*(self.l_rate/(final-initial+1))
        self.parameters=(self.parameters.T - diff).T

    def __start_gradient_descent__(self):

        """
        This function optimises the parameters for the whole data set.

        Detailed Description
        --------------------
        This function uses the number of batches, epochs, tolerance to find the optimised value of the parameters
        according to the need of the user. The function also shows progress in terms of the epochs covered. This does
        not take into account the tolerance value.
        Parameters
        ----------

        None

        Returns
        -------

        None
        """
        times=int(self.note/self.batch)
        percent=1
        for i in range(self.epochs):
            self.initial_parameters=self.parameters
            for j in range(times):
                initial=j*self.batch
                final=(j+1)*self.batch
                self.__GradDescent__(initial,final)

            if (np.abs(self.initial_parameters-self.parameters)).sum()/self.note < self.tol:
                print('Optimised to the maxium')
                break
            if  self.show_progress and(i*100/self.epochs >= percent):
                print('|',end='')
                percent+=1
        while percent<=101 and self.show_progress:
            print('|',end='')
            percent+=1
        if self.show_progress:
            print(" 100%")

    def predict(self,Y):
        """
        This function gives the predicted value of the data set given for testing.

        Parameters
        ----------

        Y:   numpy.ndarray
             This is the input of the linear regression model whose number of columns represent
             the number of dimensions of the input. The rows represent the number of data sets given
             for prediction.
        Returns
        ------

        numpy.ndarray
             This the predicted output of the input given in Y. It's number of rows represent
             the number of data sets given for prediction

        Notes
        -----

        Y.shape[0] should be equal to the number of dimensions given in the fit function.
        """
        self.test_i=np.ones([Y.shape[0],self.nod])
        if len(Y.shape) is 1:
            Y=Y.reshape([Y.shape[0],1])
        self.test_i[:,1:]=Y
        self.test_o=self.test_i.dot(self.parameters)

        return self.test_o
