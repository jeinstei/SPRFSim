#!/usr/bin/python


import numpy

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import time 
import math

import scipy.optimize as opt
from scipy.optimize import curve_fit

from elements import *

 
cell_dict = {'QWR-2012-02': 1, 'HWRDonut': 2, 'SSR13D_#72': 3, 'SSR2_beta047_v26': 4, 'beta063D_#43': 5, 'beta092': 6}
type_dict = {'QWR-2012-02': 'buncher', 'HWRDonut': 'hwr', 'SSR13D_#72': 'ssr1', 'SSR2_beta047_v26': 'ssr2', 'beta063D_#43': 'lb650', 'beta092': 'hb650'}
QL_dict = {'buncher':4850, 'hwr':2.7e6, 'ssr1':3.7e6, 'ssr2':5.8e6, 'lb650':11.3e6, 'hb650':11.5e6}
Q0_dict = {'buncher':9700, 'hwr':0.5e10, 'ssr1':0.6e10, 'ssr2':0.8e10, 'lb650':1.5e10, 'hb650':2.0e10}
R_dict = {'buncher':530.0, 'hwr':275.0, 'ssr1':242.0, 'ssr2':296.0, 'lb650':375.0, 'hb650':609.0}
kp_dict = {'buncher':1.5, 'hwr':664.0, 'ssr1':455.0, 'ssr2':713.0, 'lb650':695.0, 'hb650':707.0}
ki_dict = {'buncher':3.0e5, 'hwr':5.0e7, 'ssr1':5.0e7, 'ssr2':5.0e7, 'lb650':5.0e7, 'hb650':5.0e7}
f_dict = {'buncher':162.5e6, 'hwr':162.5e6, 'ssr1':325.0e6, 'ssr2':325.0e6, 'lb650':650.0e6, 'hb650':650.0e6}
name_list = ['DRIFT', 'QUAD', 'FIELD_MAP']

def PIP_II_cdr(beam_loading_fraction,timing_error,feed_forward_width,error_flag,error_flag_ff,single_error_flag,gain_error_flag):

	f0 = 162.5e6
	f1 = 325.0e6
	f2 = 650.0e6
	f = 162.5e6
	#feed_forward_width = 500.0e-6
	
	T = 1.0
	lattice = []

	cell_parameters = numpy.loadtxt('cell_parameters_cdr.csv',delimiter=',')

	file = open('PIP_II_CDR_v_1_2.dat','r')
	table=[row.strip().split() for row in file]	
	tableLength=len(table)
	file.close()
	
	count = 0
	k = 0
	z = 0
	m = 0
	ele = 1
	name1 = 'buncher'
	c_name = 'buncher'
	for i in range(0,len(table)):
		
		line = table[i]

		if not line:
			continue
	
		else:
			name = line[0]
			
			if name in name_list:
				if name != 'FIELD_MAP':
					L = float(line[1]) / 1000.0
					z = z + L
					element = drift(f,L,z,ele)	
					lattice.append(element)
					ele = ele + 1

				else: 
					V = cell_parameters[k,1]*1.0e6
					Phi = cell_parameters[k,2]
					L = float(line[2]) / 1000.0
					z = z + L
					
					if line[1] == '10':
						element = drift(f,L,z,ele)	
						lattice.append(element)
						ele = ele + 1
						
					else:
						type = cell_dict[line[-1]]
						name = type_dict[line[-1]]
						c_name = type_dict[line[-1]]
						f = f_dict[name]
						if name == name1:
							m = m + 1
						else:
							m = 1
						
						ff_magnitude = 0.002*R_dict[name]*QL_dict[name]*beam_loading_fraction / 2. * ( 1. + error_flag_ff * numpy.random.uniform(-0.1,0.1))
						
  						timing_errors = timing_error + error_flag_ff * numpy.random.uniform(0.0e-6,1.0e-6)
						
						if (single_error_flag == 1) and (ele == 606): 
							print m
							ff_magnitude = 0.
							timing_errors = 0.
						
						ff_phase = Phi * (1. + error_flag_ff * numpy.random.uniform(-5.0,5.0))
						
						kp = kp_dict[name] + gain_error_flag * kp_dict[name] * numpy.random.uniform(-0.5,0.5)
						ki = ki_dict[name] + gain_error_flag * ki_dict[name] * numpy.random.uniform(-0.5,0.5)
						
						
						element = rf_cavity(f0 = f,f = f,
							L = L, V = V, phase = Phi, cavity_index = k,
							ele = ele, name = name+'.'+str(m),
							QL = QL_dict[name], RoverQ = R_dict[name], kp = kp, ki = ki,
							phase_error = numpy.random.uniform(-0.5,0.5)*error_flag, voltage_error = numpy.random.uniform(-0.01,0.01)*V*error_flag,
							feed_forward_magnitude = ff_magnitude, feed_forward_phase = ff_phase, 
							feed_forward_time_error = timing_errors, feed_forward_width = feed_forward_width, ki_ff = 0.001)
						
						lattice.append(element)
												
						if lattice[-1].name == 'buncher.1':
							lattice[-1].feed_forward_magnitude = 0.005*R_dict[name]*QL_dict[name]*beam_loading_fraction / 2.
							lattice[-1].update_parameters()

						if lattice[-1].name == 'buncher.2':
							lattice[-1].feed_forward_magnitude = 0.005*R_dict[name]*QL_dict[name]*beam_loading_fraction / 2.
							lattice[-1].update_parameters()
							ele = ele + 1
							lattice.append(chopper(0.4,ele,z))
							
						name1 = name
						k = k + 1
						
	return lattice

