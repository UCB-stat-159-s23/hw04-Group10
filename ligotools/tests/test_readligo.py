from ligotools import readligo as rl
import pytest

h1_name = 'data/H-H1_LOSC_4_V2-1126259446-32.hdf5'
l1_name = 'data/L-L1_LOSC_4_V2-1126259446-32.hdf5'

#load the data
strain_H1, time_H1, chan_dict_H1 = rl.loaddata(h1_name, 'H1')
strain_L1, time_L1, chan_dict_L1 = rl.loaddata(l1_name, 'L1')

def test_h1_empty():
    assert strain_H1.any()
    assert time_H1.any()
    assert chan_dict_H1 is not None
    
def test_l1_empty():
    assert strain_L1.any()
    assert time_L1.any()
    assert chan_dict_L1 is not None
    
def test_h1_vals():
    assert strain_H1[0] == 2.177040281449375e-19
    assert time_H1[1] == 1126259446.0002441
    assert chan_dict_H1['DATA'][2] == 1
    
def test_l1_vals():
    assert strain_L1[0] == -1.0428999418774637e-18
    assert time_L1[1] == 1126259446.0002441
    assert chan_dict_L1['DATA'][2] == 1