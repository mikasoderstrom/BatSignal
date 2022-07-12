from cmath import sin
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal
from Audio_data import Audio_data
from Matrix_array import Matrix_array
from Audio_source import Audio_source


# GLOBAL VARIABLES
c = 340                     # propagation speed of sound

def antenna_setup():
    r_a1 = [0,0,0]
    uni_distance = math.pow(10,-3) * 20
    row_elements = 8
    column_elements = 8

    array_matrix_1 = Matrix_array(r_a1,uni_distance,row_elements,column_elements)
    array_matrices = np.array([array_matrix_1], dtype=object)

    sub_arrays = len(array_matrices)

    for array in range(sub_arrays):
        plt.title('Array setup')
        plt.scatter(array_matrices[array].get_r_prime()[0,:], array_matrices[array].get_r_prime()[1,:])

    #print(array_matrices[0].get_r_prime())
    #print(array_matrices[0].get_r_prime()[1,47])
    #test1 = np.array([[32,32,32],[33,33,33],[34,34,34]])
    #test = Audio_data(test1)
    #print(test.get_audio_signals()[1,1])
    return array_matrices


def generate_array_signals(matrix_array, sources, t):
    r_prime = matrix_array.get_r_prime()
    Audio_signal = np.zeros((len(t), len(r_prime[0,:])))

    for sample in range(len(t)):
        print(sample)
        for mic in range(len(r_prime[0,:])):
            x_i = r_prime[0,mic]
            y_i = r_prime[0,mic]
            temp_signal_sample = 0
            for source in range(len(sources)):
                if (sources[source].get_t_start() < t[sample]) and (t[sample] < sources[source].get_t_end()):
                    frequencies_ps = sources[source].get_frequency()
                    theta_source = sources[source].get_theta()
                    phi_source = sources[source].get_phi()
                    rho_soruce = sources[source].get_rho()
                    for freq_ind in range(len(frequencies_ps)):
                        k = 2*math.pi*frequencies_ps[freq_ind]/c
                        r_1 = np.array([x_i,y_i,0])
                        r_2 = rho_soruce * r_vec(theta_source,phi_source)
                        phase_offset = -k*np.linalg.norm(r_2-r_1)
                        element_amplitude = 1/np.linalg.norm(r_2-r_1)
                        temp_signal_sample += element_amplitude * math.sin(2*math.pi* frequencies_ps[freq_ind] * t[sample] + phase_offset)
            Audio_signal[sample,mic] = temp_signal_sample
    return Audio_signal


def r_vec(theta,phi):
    r = np.array([(math.sin(theta)*math.cos(phi)), math.sin(theta)*math.sin(phi), math.cos(theta)])
    return r

def filtering(array_audio_signals, sub_arrays, frequency_bands, f_sampling, elements):
    for array in range(sub_arrays):
        Audio_signal = array_audio_signals[array].get_audio_signals()
        #elements = array_matrices[array].get_elements()

        audio_filtered_complete = np.zeros((sub_arrays, len(frequency_bands)), dtype=object)

        for freq_ind in range(len(frequency_bands)):
            # filter design for each band
            filter_order = 200
            nu_0 = 2*frequency_bands[freq_ind]/f_sampling   # normalized frequency
            scale_factor = 10000                            # scale factor, making filter bandwidth more narrow
            cut_off = [nu_0 - nu_0/scale_factor, nu_0 + nu_0/scale_factor]

            b = signal.firwin(filter_order, cut_off, window="hamming", pass_zero=False) # filter coefficients
            
            audio_temp = np.zeros((len(Audio_signal[:,0]),elements))
            for mic_ind in range(elements):
                # apply filter on evert signal recorded from the elements
                audio_temp[:,mic_ind] = signal.lfilter(b, 1.0, Audio_signal[:,mic_ind])

            audio_filtered_complete[array,freq_ind] = Audio_data(audio_temp)

            plt.figure(2)
            w, h = signal.freqz(b, worN=8000)
            H = 20*np.log10(abs(h))
            plt.plot((w/math.pi)*f_sampling/2, 20*np.log10(abs(h)), linewidth=2)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Gain (dB)')
            plt.title('Frequency Response of all filters')
            plt.ylim(-5, 0.5)
            plt.xlim(70, f_sampling/2)
            plt.grid(True)
        plt.show()
    return audio_filtered_complete

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

    return mic_data

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
    
    return y
    
    

def main():
    # Initialization
    f_sampling = 1000           # sampling frequency in Hz
    t_start = 0                 # start time of simulation 
    t_end = 1                  # end time of simulation
    t_total = t_end - t_start   # total simulation time
    t = np.linspace(t_start, t_end, t_total*f_sampling) # time vector

    # Point audio source
    away_distance = 700         # distance between the array and sources, in cm (?)

    # vector holding the two sources of class Audio_source
    #sources = np.array([Audio_source(1600, 2500, 40, 20, 140, away_distance, 0, 1.5), Audio_source(1600, 1900, 40, 30, 0, away_distance, 0.5, 2)], dtype=object)
    sources = np.array([Audio_source(300, 400, 20, 20, 140, away_distance, 0, 1.5)])

    array_matrices = antenna_setup()

    sub_arrays = len(array_matrices)

    # GENERATE AUDIO SIGNALS
    array_audio_signals = np.zeros((sub_arrays), dtype=object)
    for array in range(sub_arrays):
        # generate the audio signals on each array-element for each sub-array
        temp_signal = generate_array_signals(array_matrices[array],sources,t)
        array_audio_signals[array] = Audio_data(temp_signal)
        print('Audio signal for array '+str(array+1)+' generated')

    # BEAMFORMING
    x_res = 10                          # resolution in x
    y_res = 10                          # resolution in y
    x_listen = np.linspace(-1,1,x_res)  # scanning window, x coordinates
    y_listen = np.linspace(-1,1,y_res)  # scanning window, y coordinates
    r_scan = 2                          # radius of our scanning window, r_scan² = x²+y²+z²

    f_bands_N = 45                      # number of frequency bands
    bandwidth = [100, 3900]             # bandwidth of incoming audio signal
    frequency_bands = np.linspace(bandwidth[0],bandwidth[1],f_bands_N) # vector holding center frequencies of all frequency bands
    samples = len(t)
    #filter_coefficients = np.zeros((f_bands_N, filter_order+1)) # might only be used for plots

    # CREATE COLORMAPS

    # FILTERING
    audio_filtered_complete = filtering(array_audio_signals, sub_arrays, frequency_bands, f_sampling)

    plt.show() # show all plots






main()