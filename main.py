import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.signal import find_peaks, peak_prominences
from scipy.integrate import simps
import pandas as pd

@st.experimental_memo
def fits_io(path_to_file):
    """
    I/O for fits files
    Input: ASCII/FITS/LC/CSV/XLS file path
    Output: Time and Rate arrays
    """
    if path_to_file.name.endswith('.txt') or path_to_file.name.endswith('.ascii'):
      data = Table.read(path_to_file, format='ascii')
      time=data.field("TIME")
      rate=data.field("RATE")
    elif path_to_file.name.endswith('.lc') or path_to_file.name.endswith('.fits'):
      data = Table.read(path_to_file, format='fits')
      time=data.field("TIME")
      rate=data.field("RATE")
    elif path_to_file.name.endswith('.csv'):
      data = Table.read(path_to_file, format='csv')
      time=data.field("TIME")
      rate=data.field("RATE")
    elif path_to_file.name.endswith('.xlsx') or path_to_file.name.endswith('.xls') or path_to_file.name.endswith('.xlsm') or path_to_file.endswith('.xlsb') or path_to_file.name.endswith('.odt') or path_to_file.name.endswith('.odf'):
      data = pd.read_excel(path_to_file)
      time=data["TIME"]
      rate=data["RATE"]
    background_count = np.mean(rate)
    return time, rate, background_count

def noise_reduction(time, rate):
    """
    Noise reduction of rate using time averaging
    Input: time, rate arrays
    Output: Filtered time and rate arrays
    """
    w = 500
    n = len(time)
    time_filtered = time[:n-w+1]
    rate_filtered = np.convolve(rate, np.ones(w), 'valid') / w
    return time_filtered, rate_filtered

def get_and_segregate_peaks(time_filtered, rate_filtered):
    """
    Recognizes peaks in the time vs rate data and segregates out the peaks
    Input: time_filtered, rate_filtered
    Output: Number of bursts==Number of peaks, 2 dictionaries(bursts_rate containing the rate of bursts and bursts_time containing time of the bursts)
    Dict format:
    {'key is the index of burst': [rate array for bursts_rate, time array for bursts_time]}
    """
    # Peaks recognition
    peaks, _ = find_peaks(rate_filtered)
    prominences, _, _ = peak_prominences(rate_filtered, peaks)
    selected = prominences > 0.25 * (np.min(prominences) + np.max(prominences))
    top = peaks[selected]   # Contains indexes of all peak values in rate array

    num_bursts = len(top)   # Assuming each peak for corresponds to a burst

    # Peaks segregation
    rise_pt_idx=[]
    decay_pt_idx=[]
    for i in range(num_bursts):
        peak_idx = top[i]
        if i==0:
            rise_pt_idx.append(np.where(rate_filtered==min(rate_filtered[:peak_idx])))
        else:
            rise_pt_idx.append(np.where(rate_filtered==min(rate_filtered[top[i-1]:peak_idx])))

        if i==num_bursts-1:
            decay_pt_idx.append(np.where(rate_filtered==min(rate_filtered[peak_idx:])))
        else:
            decay_pt_idx.append(np.where(rate_filtered==min(rate_filtered[peak_idx:top[i+1]])))

    bursts_rate={}
    bursts_time={}
    for i in range(len(rise_pt_idx)):
        start=rise_pt_idx[i][0][0]
        end=decay_pt_idx[i][0][0]
        
        bursts_rate[i] = rate_filtered[start:end]
        bursts_time[i] = time_filtered[start:end]

    return num_bursts, bursts_rate, bursts_time

def analyse_wavelets(time, rate):
    """
    Run this function for each wavelet in the dictionaries in get_and_segregate_peaks()
    Returns desired parameters(rise time, decay time, peak flux etc)
    Input: filtered time and rate arrays
    Output: desired Parameters
    """
    peak_count=max(rate)
    peak_idx = np.where(rate==max(rate))[0][0]
    # print(peak_idx)
    y_values = (rate-np.mean(rate)) / np.std(rate)
    x_values = (time-np.mean(time)) / np.std(time)
    mean = np.mean(rate)
    stdev = np.std(rate)
    arr_prev_rev = list(rate[:peak_idx])
    arr_prev_rev = arr_prev_rev[::-1]
    for i in arr_prev_rev:
      rise_start_idx=arr_prev_rev[::-1].index(i)
      if i<=0.1*peak_count:
        break
  
    arr_fwd = list(rate[peak_idx:])
    for i in arr_fwd:
      decay_end_idx = arr_fwd.index(i)+peak_idx
      if i<=0.1*peak_count:
        break
    rise_time = time[peak_idx]-time[rise_start_idx]
    decay_time = time[decay_end_idx]-time[peak_idx]
    flare_duration = rise_time+decay_time
    peak_count = int(max(rate))

    # Calculating total count for each wavelet
    total_count = simps(rate, dx = time[decay_end_idx]-time[rise_start_idx])

    return rise_time, decay_time, flare_duration, peak_count, total_count

@st.experimental_memo
def classify_flare(peak_count):
    """
    CLassifies the flare based on peak flux
    Input: peak flux of the flare
    Output: Class 
    """
    if(peak_count > 1e5):
        return "X"
    if(peak_count > 1e4):
        return "M"
    if(peak_count > 1e3):
        return "C"
    if(peak_count > 1e2):
        return "B"
    else:
        return "A"

def visualize_flux(time, rate):
    fig, ax = plt.subplots()
    ax.plot(time/1e8, rate)
    ax.set_ylabel('Rate (counts/sec)')
    ax.set_xlabel('Time (sec)')
    return fig

def st_ui():
    st.write("# Welcome to the Chandrayaan-2 X-Ray Monitor Remote Sensing Daisi! 👋")
    st.markdown(
        """
        Cosmic sources in the sky, including our star, the Sun, burst intermittently in the X-ray energy
        band of the electromagnetic spectrum. The burst amplitude and the duration vary depending on
        the source and the cause of the burst. The observed burst has a fast rise and slow decay
        pattern/shape with variable rise and decay times. An automatic identification system for such
        data will further simplify the analysis process and the associated scientific investigation of
        these bursts.
        """
    )

    demo_type = st.sidebar.radio("Select the Demonstration ✨", ["Sample File", "File Upload"])

    if demo_type == "File Upload":
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=['lc','csv','ascii', 'nc', 'txt', 'xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods' and 'odt'])
    else:
        selector_dict = {
            "Sample_1": "./data/ch2_xsm_20211013_v1_level2.lc",
            "Sample_2": "./data/ch2_xsm_20210919_v1_level2.lc",
            "Sample_3": "./data/ch2_xsm_20211022_v1_level2.lc"
        }

        document_selector = st.sidebar.selectbox("Select the Sample File 📄", selector_dict.keys())
        uploaded_file = open(selector_dict[document_selector], "rb")

    if uploaded_file is not None:

        if st.button("Analyze"):
            time, rate, background_count = fits_io(uploaded_file)

            st.header("Raw Data Input")
            st.write("The light curve is parsed and the ‘Time’ and ‘Rate’ values are plotted. ")
            df = pd.DataFrame({'Time (in sec)':time/1e8,'Rate (counts/sec)':rate})
            df = df.set_index(time/1e8)
            raw_fig = visualize_flux(time/1e8, rate)
            st.pyplot(raw_fig)

            st.write("Background Count is %d " %background_count)

            st.header("Data after Noise Reduction")
            st.write("Now we perform noise reduction of rate using time averaging. This is the filtered data that can be seen in the app.")
            filtered_time, filtered_rate = noise_reduction(time, rate)
            clean_fig = visualize_flux(filtered_time/1e8, filtered_rate)
            st.pyplot(clean_fig)
            
            st.header("Peaks observed from Input Data")
            st.subheader("Peak Detection")
            st.write("Since there can be multiple peaks, meaning multiple flares, in a single light curve, we need to detect all of them. We use the find_peaks() and peak_prominences() already implemented in SciPy to find the peaks and their index values in the ‘Time’ and ‘Rate’ arrays.")
            num_bursts, bursts_rate, bursts_time = get_and_segregate_peaks(filtered_time, filtered_rate)
            
            st.subheader("Peaks Segregation")
            st.write("Once all the peaks are detected, they are separated using a devised algorithm. For each peak, we get the minimum value of the rate between that particular peak and the peak before it(or the start of data, whichever is available first), and assign it as the start time for that wavelet. Similarly, we take the minimum value of the rate between this peak and the peak after it(or the end of data, whichever is available first), and assign it as the end time for that wavelet. The Rate and Time values for each of the wavelets are stored for further analysis. These are the multiple peaks that are plotted in the app (if the number of peaks is more than 1).")
            cols = st.columns(2)
            for i in range(num_bursts):      
                peak_seg_fig = visualize_flux(bursts_time[i]/1e8, bursts_rate[i])
                cols[i%2].pyplot(peak_seg_fig)
        
            st.caption("There were %s peaks observed in the LC data." % num_bursts)

            st.header("Wavelet Analysis")
            st.write("Once we get the wavelets, we can run an analysis on each of them and extract all of the necessary parameters from the information available. We first get the Mean and Standard Deviation of the data using regular methods. Then we find the Peak Flux for the data in counts per second and use it further to find the Rise and Decay Time. For Rise Time and Decay Time, we use the general definition of Rise Time (The time taken by the signal to rise from 10% of peak value to 90% of peak value) and Decay Time (The time taken by the signal to decay from 90% of peak value to 10% of peak value). We find the start and endpoint for the wavelet (i.e. Where the rate value is 10% of the peak value) and calculate Rise Time as T(90% of peak) - T(10% of peak) and Decay Time as T(10% of peak) - T(90% of peak).")
            st.write("The Total Flux of the wavelet is calculated as the area under the curve between the starting and ending points of the wavelet. This is calculated by integrating between these two points using simps() method implemented in SciPy.")
            df_analysis = {"rise_time":[], "decay_time":[], "flare_duration":[], "peak_count":[], "total_count":[]}
            

            for key in bursts_rate.keys():
                rate = bursts_rate[key]
                time = bursts_time[key]
                params=analyse_wavelets(time, rate)
                df_analysis['rise_time'].append(params[0])
                df_analysis["decay_time"].append(params[1])
                df_analysis["flare_duration"].append(params[2])
                df_analysis["peak_count"].append(params[3])
                df_analysis["total_count"].append(params[4])
            
            st.dataframe(pd.DataFrame(df_analysis).astype(str))
            st.caption("**Rise and Decay time in sec") 

            st.header("LC Solar Flare Classification")
            st.write("Solar flares can be classified using the Soft X-ray classification or the H-alpha classification. The Soft X-ray classification is the modern classification system. Here 5 letters, A, B, C, M, or X are used according to the peak flux (Watt/m2). This classification system divides solar flares according to their strength. The smallest ones are ‘A-class’, followed by ‘B’, ‘C’, ‘M’ and ‘X’. Each letter here represents a 10-fold increase in energy output. And within each scale, there exists a finer scale from 1 to 9. And then comes the X-class flares, these are the most powerful flares of all. These flares can go higher than 9.")
            df_classify = {"Classification":[]}
            for i in range(num_bursts):
                df_classify["Classification"].append(classify_flare(df_analysis["peak_count"][i]))
            st.dataframe(pd.DataFrame(df_classify))

if __name__ == '__main__':
    st_ui()