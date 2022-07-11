from cmath import sin
import numpy as np
from Audio_data import Audio_data
from Matrix_array import Matrix_array
import math

def antenna_setup():
    r_a1 = [0.1,0,0]
    uni_distance = math.pow(10,-3) * 20
    row_elements = 8
    column_elements = 8

    array_matrix_1 = Matrix_array(r_a1,uni_distance,row_elements,column_elements)

    array_matrices = [array_matrix_1]
    #print(array_matrices[0].get_r_prime())s
    print(array_matrices[0].get_r_prime()[1,47])
    test1 = [[32,32,32],[33,33,33],[34,34,34]]
    test = Audio_data(test1)
    print(test.get_audio_signals())

def r_vec(theta,phi):
    r = [(math.sin(theta)*math.cos(phi)), math.sin(theta)*math.sin(phi), math.cos(theta)]
    r = np.array(r) # convert list to vector
    return r

def beam_forming_algorithm(matrix_array,direction,weight,audio_signal,frequency,sampling_frequency,c):
    #
    #   IMPORTANT! The beamforming algorithm assumes the array matrices lies in the xy-plane
    #
    #   The beamforming algorithm calculates the necessary phase to introduce to the narrowband signal
    #   in order to have the maximum directivity in the direction r(theta,phi)
    #
    #   This phase-shift is introduced by a phase shifting function, which acts as a filter in time domain.
    #
    #   To improve performance, all elements of the array matrices are not in use. The user decides which 
    #   element to use by sending in a weight vectir as a argument. The output signal is then normalized after
    #   how many elements where in use.

    #   Get the amount of samples of the audio tracks 
    samples = len(audio_signal[:,0])

    #   The listening-direction vector contains two scalar values, theta and phi
    theta = direction[0]
    phi = direction[1]

    #   The r_prime vector of the matrix array to know the location of every element, as well as how many 
    #   elements exists.
    r_prime = matrix_array.get_r_prime()
    elements = matrix_array.get_elements()

    #   The narrowband wavevnumber
    k = 2*math.pi*frequency/c

    #   The normalized frequency
    ny = frequency/sampling_frequency

    #   Initialize output vector
    mic_data = np.zeros((samples,1))

    #   The compensation factors to obtain uniform phase in the direction r_hat(theta,phi)
    x_factor = math.sin(theta)*math.cos(phi)
    y_factor = math.sin(theta)*math.sin(phi)

    for mic_ind in range(elements):
        #   calculate the narrowband phase-shift
        phase_shift_value = -k*(r_prime[0,mic_ind] * x_factor + r_prime[1,mic_ind]*y_factor)

        #   Sum the individually shifted data from the atnenna elements as well as weight them with
        #   appropriate weight.
        mic_data = mic_data + weight[mic_ind] * phase_shift(audio_signal[:,mic_ind],ny,phase_shift_value)

    norm_coeff = 1/sum(weight)
    mic_data = mic_data * norm_coeff

def phase_shift(x,ny,phase):
    #   Input signal x
    #
    #   Output signal y
    #
    #   if x = cos(n*2*pi*ny), then y = cos(n*2*pi*ny + phase)
    #
    x_length = len(x)
    y = np.zeros((x_length,1))

    for i in range(x_length-1):
        y[i] = math.cos(phase) * x[i] + math.sin(phase)/(2*math.pi*ny)*(x[i+1]/2 - x[i-1]/2)
    
    
antenna_setup()
