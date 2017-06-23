import numpy as np
class linear_regression1:

    def __init__(self,batch_size=0,epochs=100,learning_rate=0.001,tolerance=0.00001,show_progress=True):

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
                    Default Value is 0.00001
        show_progress: Boolean
                    It controls whether the object will show the progress as output or not.
                    Default Value: True

        Returns
        -------
        Nothing

        """


        #Batch Size
        self.batch=batch_size

        #Maximum number of iterations that the object will perfom
        self.epochs=epochs

        #Learning Rate of the linear regression algo
        self.l_rate=learning_rate

        #Bool Value of whtether to show progress or not
        self.show_progress=show_progress

        #Maximum change in parameters or weights that can be assumed negligible
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

        #Number of Training Examples
        self.note=X.shape[0]

        #If Batch value is zero, it is assumed the whole dataset is the batch
        if self.batch is 0:
            self.batch=self.note

        #Changing Vector To Mat
        if len(X.shape) is 1:
            X=X.reshape([X.shape[0],1])

        #Number of Dimensions plus one bias introducted
        self.nod=X.shape[1]+1

        #Training data initialized
        self.train_i=np.ones([self.note,self.nod])

        #Leaving Bias values as 1
        self.train_i[:,1:]=X

        #Training data output stored and changing Vector To Matrix
        if len(Y.shape) is 1:
            Y=Y.reshape([Y.shape[0],1])
        self.train_o=Y

        #Parameters or weights randomly generated
        self.parameters=np.random.random([self.nod,1])

        #Starting Gradient Descent
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



        #Difference between expected and actual values
        diff=(self.train_i[initial:final].dot(self.parameters)-self.train_o[initial:final])

        #Multiplying with respected values to get differentiation
        product=diff*self.train_i[initial:final]

        #Adding column-wise to get differentitation w.r.t. parameters
        delta=(product.sum(axis=0))*self.l_rate/(final-initial+1)

        #Changing the Value Of parameters
        self.parameters=self.parameters-delta.reshape([delta.shape[0],1])

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

        #Number of times the whole set of Parameters be optimized in one epochs
        times=int(self.note/self.batch)

        #Value used to show percentage
        percent=1

        #Loss Curve is initialzed every time this function is being called
        self.loss_curve=[]

        #Gradient Desecent Started
        for i in range(self.epochs):

            #Initial Parameters Stored
            self.initial_parameters=self.parameters
            for j in range(times):
                initial=j*self.batch
                final=(j+1)*self.batch
                self.__GradDescent__(initial,final)
            #One Iteration of Gradient Descent Complete


            #Finding and adding loss to the loss curve
            diff=(self.train_i.dot(self.parameters)-self.train_o)
            loss=(np.abs(diff)).sum()
            self.loss_curve.append(loss)

            #Checking for tolerance
            if (np.abs(self.initial_parameters-self.parameters)).sum()/self.note < self.tol:
                print('Optimised to the maxium')
                break

            #For showing percentage
            if  self.show_progress and(i*100/self.epochs >= percent):
                print('|',end='')
                percent+=1

        #Completing the Percentage if the loops is broken in between
        while percent<=101 and self.show_progress:
            print('|',end='')
            percent+=1

        #Displaying 100% Complete
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

        Y.shape[1] should be equal to the number of dimensions given in the fit function.
        """

        #Converting the testing data into data with bias
        self.test_i=np.ones([Y.shape[0],self.nod])
        if len(Y.shape) is 1:
            Y=Y.reshape([Y.shape[0],1])
        self.test_i[:,1:]=Y

        #Storing Output
        self.test_o=self.test_i.dot(self.parameters)

        return self.test_o
