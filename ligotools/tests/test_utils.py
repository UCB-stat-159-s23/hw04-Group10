from ligotools import utils as ul
from ligotools import readligo as rl
from scipy.io import wavfile
import numpy as np
import json
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pytest
fnjson = 'data/'+"BBH_events_v3.json"

#load the data for whiten()
eventname = 'GW150914' 
events = json.load(open(fnjson,"r"))
event = events[eventname]
fs = event['fs']
tevent = event['tevent']  
h1_name = 'data/H-H1_LOSC_4_V2-1126259446-32.hdf5'
l1_name = 'data/L-L1_LOSC_4_V2-1126259446-32.hdf5'
strain_H1, time_H1, chan_dict_H1 = rl.loaddata(h1_name, 'H1')
strain_L1, time_L1, chan_dict_L1 = rl.loaddata(l1_name, 'L1')
time = time_H1
dt = time[1] - time[0]
NFFT = 4*fs
Pxx_H1, freqs = mlab.psd(strain_H1, Fs = fs, NFFT = NFFT)
Pxx_L1, freqs = mlab.psd(strain_L1, Fs = fs, NFFT = NFFT)
psd_H1 = interp1d(freqs, Pxx_H1)
psd_L1 = interp1d(freqs, Pxx_L1)

#Use whiten()
strain_H1_whiten = ul.whiten(strain_H1,psd_H1,dt)
strain_L1_whiten = ul.whiten(strain_L1,psd_L1,dt)

#Load data for write_wavfile and reqshift()
fs = 4096
fshift = 400.
fband = event['fband']
bb, ab = butter(4, [fband[0]*2./fs, fband[1]*2./fs], btype='band')
normalization = np.sqrt((fband[1]-fband[0])/(fs/2))
strain_H1_whitenbp = filtfilt(bb, ab, strain_H1_whiten) / normalization
strain_L1_whitenbp = filtfilt(bb, ab, strain_L1_whiten) / normalization
deltat_sound = 2.                     
indxd = np.where((time >= tevent-deltat_sound) & (time < tevent+deltat_sound))

#Use write_wavfile 
ul.write_wavfile('ligotools/tests/write_wavfile_test'+eventname+"_H1_whitenbp.wav",int(fs), strain_H1_whitenbp[indxd])
ul.write_wavfile('ligotools/tests/write_wavfile_test'+eventname+"_L1_whitenbp.wav",int(fs), strain_L1_whitenbp[indxd])

#Use reqshift()
strain_H1_shifted = ul.reqshift(strain_H1_whitenbp,fshift=fshift,sample_rate=fs)
strain_L1_shifted = ul.reqshift(strain_L1_whitenbp,fshift=fshift,sample_rate=fs)

def test_whiten_empty():
    assert strain_H1_whiten is not None
    assert strain_L1_whiten is not None
    
def test_write_wavfile(tmpdir):
    # Generate some test data
    fs = 44100  # Sample rate
    duration = 5  # Duration in seconds
    f = 440  # Frequency in Hz
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # Time vector
    data = np.sin(2 * np.pi * f * t)  # Sinusoidal waveform

    # Write data to a temporary file using write_wavfile()
    file = tmpdir.join('output.wav')
    ul.write_wavfile(file.strpath, fs, data)

    # Read data from the temporary file
    fs_read, data_read = wavfile.read(file.strpath)

    # Check that the sample rate is the same as the original input
    assert fs_read == fs
    # Check that the data has been scaled and converted correctly
    data_scaled = np.int16(data / np.max(np.abs(data)) * 32767 * 0.9)
    assert np.allclose(data_read, data_scaled)
    
def test_reqshift_empty():
    assert strain_H1_shifted is not None
    assert strain_L1_shifted is not None
    

def test_sigplot():
    # setup test data
    time = np.linspace(0, 1, 100)
    timemax = 0.5
    SNR = np.sin(2*np.pi*time*10) + np.random.normal(0, 0.5, len(time))
    pcolor = 'b'
    det = 'H1'
    eventname = 'test_event'
    fs = 100
    plottype = 'png'
    tevent = 0.3
    strain_whitenbp = np.sin(2*np.pi*time*10) + np.random.normal(0, 0.1, len(time))
    template_match = np.sin(2*np.pi*time*10) * 0.5
    template_fft = np.fft.fft(template_match)
    datafreq = np.fft.fftfreq(len(time), 1/fs)
    d_eff = 1.0
    freqs = np.logspace(1, np.log10(fs/2), 100)
    data_psd = np.interp(freqs, datafreq[:len(time)//2+1], np.abs(np.fft.fft(strain_whitenbp)[:len(time)//2+1])**2)

    # call function to create plot
    ul.sigplot(time, timemax, SNR, pcolor, det, eventname, fs, plottype, tevent,
            strain_whitenbp, template_match, template_fft, datafreq, d_eff,
            freqs, data_psd)
    
    fig = plt.gcf()
    plt.close(fig)

    assert fig is not None

