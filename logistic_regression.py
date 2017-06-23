import numpy as np
class logistic_regression:

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
                    Default Value is 0.00001

        show_progress: Boolean
                    It controls whether the object will show the progress as output or not.
                    Default Value: True

        Returns
        -------
        Nothing

        """

        #Batches Size
        self.batch=batch_size

        #Maximum number of iterations that the object will perfom
        self.epochs=epochs

        #The learning rate of the logistic regression algorithm
        self.l_rate=learning_rate

        #Tolerance i.e. the maximum change in parameters that is assumed negligible
        self.tolerance=tolerance

        #Bool to whether to show progress or not
        self.show_progress=show_progress

        #Stores score history which is the difference between the real and expected values
        self.score_curve=[]

    def fit(self,X,Y,number_of_categories=0):

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
            The number of rows defines the number of data sets avaiable for training. It should be preferrably a
            vector.

        number_of_categories:  int

            It is the number of categories identified by the logistic regression algorithm.
            Default Value is 0 which means the algorithm will find the maximum value in Y and find the number of categories
            on its own.



        Returns
        -------

        Nothing

        Notes
        -----
        X.shape[0] must be equal to Y.shape[0] which is also the number of data sets avaiable for training.
        The Vector Y should contain integer values from 0 to n-1 (both inclusive) where n is the number of categories.
        """

        #The matrix X is arranged such that it contains examples row-wise,
        #So, number of rows = Number of Training Examples(note)
        self.note=X.shape[0]

        #if X is a vector converting it to matrix so as to get the shape
        if len(X.shape) is 1:
            X=X.reshape([X.shape[0],1])

        #Number of dimensions in one example
        #Adding one for bias
        self.nod=X.shape[1]+1

        #Creating training set
        self.train_i=np.ones([self.note,self.nod])

        #Leaving the first column as it is (bias for every example), the rest is equal to X
        self.train_i[:,1:]=X

        #if the self.batch size is left zero then the batch size is assumed to be the whole data set
        if self.batch is 0:
            self.batch=self.note

        #if number of categories are not given, then it is assumed that maximum value +1 is the number of categories
        #There is an underlying assumption that the categories are numbered between 0 to n-1 where n is the number of
        #categories
        if number_of_categories is 0:
            self.noc=int(Y.max()+1)
        else:
            self.noc=int(number_of_categories)

        #if the labels are in matrix they are flattened in a vector
        if len(Y.shape)>1:
            Y=Y.flatten()

        #if the number of labels not equal to the number of examples in X, a ShapeError is raised
        if Y.shape[0]!=self.note:
            raise ShapeError('Number of Examples do not match in input and output data')

        #Label matrix is initialized to zero matrix
        self.train_o=np.zeros([self.note,self.noc])

        #Creation of One-Hot-Vector using the labels
        for i in range(self.note):
            x=np.zeros([self.noc,])
            try:
                x[int(Y[i])]=1
            except IndexError:
                print("Number of Categories not consistent with values in Training Labels")
            self.train_o[i,:]=x

        #Parameters or Weights are initialized randomly
        self.parameters=np.random.random([self.nod,self.noc])

        #Temporary Matrix used to store change in parameters is also initialized
        self.temp=np.zeros(self.parameters.shape)

        #Function is called for start fitting the parameters for the dataset
        self.__Start_Gradient_Descent__()

    #Function for getting outputs for a specific bath
    def __Expected_Output__(self,initial,final):

        """
        The function finds the expected output.

        Detailed Description
        --------------------
        The function takes on the initial and final values of the batch of which it gives the expected probabilities
        w.r.t to each category.

        Parameters
        ----------
        initial:  int
            It is the starting index of the batch in the training input data set.

        final:  int
            It is the end index of the batch + 1 in the training input data set.

        Returns
        -------

        numpy.ndarray
        It gives the expected probabilties for the examples in the respective batch for every category.

        Notes
        -----
        It is an internal function used by the class.
        inital is greater than equal to final.
        final should be less than equal to number of training examples
        """

        product=self.train_i[initial:final].dot(self.parameters)
        #Sigmoid function
        output=1/(1+np.exp(-product))
        return output

    def __Gradient_Descent__(self,initial,final):

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
        initial should always be less than or equal to final.
        final should always be less than equal to the number of training examples.

        """


        output=self.__Expected_Output__(initial,final)
        #Difference between actual output and expected output
        diff=(self.train_o[initial:final]-output)

        #Now for every category a new parameter is generated
        shape=diff.shape[0]
        for i in range(self.noc):
            #Normal Logistic regression is performed
            product=(diff[:,i].reshape([shape,1]))*self.train_i[initial:final]

            #Sum along the column is done so as to get change for every specific weight or Parameter
            temp=product.sum(axis=0)

            #Parameter for every category is stored
            self.temp[:,i]=temp

        #Change in real parameters after 1 iteration of Gradient descent on one specific batch
        self.parameters=self.parameters + self.l_rate*self.temp/(final-initial+1)


    def __Start_Gradient_Descent__(self):

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


        #Number of times the loop should be exectuted to do Gradient Descent on every batch
        times=int(self.note/self.batch)

        #Percent of epochs executed , initially  1
        percent=1;

        #Initialized every time when this funtion is called
        self.score_curve=[]
        self.log_likelihood_history=[]

        #Intial Output stored
        previous_output=self.__Expected_Output__(0,self.note)
        initial_parameters=self.parameters
        for i in range(self.epochs):
            for j in range(times):
                initial=j*self.batch
                final=(j+1)*self.batch
                self.__Gradient_Descent__(initial,final)

            #After this loop Gradient Descent has been executed over the whole dataset
            output=self.__Expected_Output__(0,self.note)

            #Finding the change in values expected and actual
            score=(np.abs(output-previous_output).sum())

            #Storing new output in previous_output
            previous_output=output

            #Adding score to the score_curve
            self.score_curve.append(score)

            #Calculating change in parameters
            diff=np.abs(initial_parameters-self.parameters).sum()/(self.nod*self.noc)

            #if the difference is less than tolerance then the loop is broken
            if diff<=self.tolerance:
                print("Optimized to the Maximum")
                break

            #Storing new parameters in initial_parameters
            initial_parameters=self.parameters
            #Progress is show in form of ||||||||||
            if self.show_progress and (i*100/self.epochs > percent):
                print('|',end='')
                percent+=1
        while self.show_progress and percent<101:
            print('|',end='')
            percent+=1

        #Once completed 100% is shown at the end
        if self.show_progress:
            print("100%")

    #Used to get the predicted value
    def predict(self,X):

        """
        This function gives the predicted value of the data set given for testing.

        Parameters
        ----------

        X:   numpy.ndarray
             This is the input of the linear regression model whose number of columns represent
             the number of dimensions of the input. The rows represent the number of data sets given
             for prediction.
        Returns
        ------

        numpy.ndarray
             This the predicted output of the input given in Y. It's number of rows represent
             the number of data sets given for prediction. It will be a flattened array with data type
             of int with values ranging from 0 to n-1 where n is the number of categories.

        Notes
        -----

        Y.shape[1] should be equal to the number of dimensions given in the fit function.

        """

        #Procedure for adding one bias dimension to every example in X
        #----Start--------#
        self.test_i=np.ones([X.shape[0],self.nod])
        if len(X.shape) is 1:
            X=X.reshape([X.shape[0],1])
        self.test_i[:,1:]=X
        #-----End-------#

        #Procedure for getting output in form of probabilities
        #---------Start---------#
        product=self.test_i.dot(self.parameters)
        one_hot_output_vector=1/(1+np.exp(-product))
        #---------End-----------#

        #Procedure for converting one hot vector back to normal form
        #---------Start---------#
        self.train_o=np.zeros([X.shape[0],])
        for i in range(X.shape[0]):
            maxx=-1
            index=0
            for j in range(self.noc):
                if(one_hot_output_vector[i][j]>maxx):
                    maxx=one_hot_output_vector[i][j]
                    index=j
                self.train_o[i]=int(index)
        #--------End------------#

        return self.train_o
