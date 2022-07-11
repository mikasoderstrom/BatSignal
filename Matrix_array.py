import numpy as np

class Matrix_array:
    def __init__(self,r_a,element_distance,row_elements,column_elements):
        #
        # Generates coordinates for a 2D array. Uni_distance is the uniform distance between the 
        # array-elements.
        #
        # r_a is the vector pointing to the middle of the array
        #
        #
        #
        # r_prime is a vector on the form 
        #            x_1    x_2 ... x_n
        # r_prime =  y_1    y_2 ... y_n, where n = rows*columns
        #            z_1    z_2 ... z_n
        #

        self.__row_elements = row_elements
        self.__column_elements = column_elements
        self.__uni_distance = element_distance
        self.__elements = row_elements * column_elements
        self.__r_prime = np.zeros((3,self.__elements))

        element_index = 0
        for i in range(row_elements):
            for j in range(column_elements):
                self.__r_prime[0][element_index] = i*self.__uni_distance + r_a[0]
                self.__r_prime[1][element_index] = j*self.__uni_distance + r_a[1]
                element_index += 1
    

    def get_r_prime(self):
        return self.__r_prime

    def get_uni_distance(self):
        return self.__uni_distance

    def get_elements(self):
        return self.__elements

    def get_row_elements(self):
        return self.__row_elements

    def get_column_elements(self):
        return self.__column_elements
      
      