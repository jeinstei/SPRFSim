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


def PIP_II(beam_loading_fraction,timing_errors,error_flag):

	file = open('PIP-II_MEBT-HB650.txt','r')
	table=[row.strip().split() for row in file]	
	tableLength=len(table)
	file.close()


	f0 = 162.5e6
	f1 = 325.0e6
	f2 = 650.0e6
	f = f0
	feed_forward_width = 511.0e-6
	
	QL_B = 5210.
	RoverQ_B = 700.
	Q0_B = 10000.
	kp_B = 1.5
	ki_B = 3.0e5
	feed_forward_magnitude_B = -0.002*RoverQ_B*QL_B*beam_loading_fraction
	feed_forward_time_error_B = 0 + timing_errors
				
	QL_HWR = 2.7e6
	G_HWR = 48.
	RoverQ_HWR = 275
	Q0_HWR = 0.5e10
	kp_HWR = 664.
	ki_HWR = 5.0e7
	feed_forward_magnitude_HWR = -0.002*RoverQ_HWR*QL_HWR*beam_loading_fraction
	feed_forward_time_error_HWR = 0 + timing_errors

	QL_SSR1 = 3.7e6
	G_SSR1 = 84.
	RoverQ_SSR1 = 242
	Q0_SSR1 = 0.6e10
	kp_SSR1 = 455.0
	ki_SSR1 = 5.0e7
	feed_forward_magnitude_SSR1 = -0.002*RoverQ_SSR1*QL_SSR1*beam_loading_fraction
	feed_forward_time_error_SSR1 = 0 + timing_errors
	
	QL_SSR2 = 5.8e6
	G_SSR2 = 115.
	RoverQ_SSR2 = 296
	Q0_SSR2 = 0.8e10
	kp_SSR2 = 713.0
	ki_SSR2 = 5.0e7
	feed_forward_magnitude_SSR2 = -0.002*RoverQ_SSR2*QL_SSR2*beam_loading_fraction
	feed_forward_time_error_SSR2 = 0 + timing_errors
	
	QL_LB650 = 11.3e6
	G_LB650 = 191.
	RoverQ_LB650 = 375
	Q0_LB650 = 1.5e10
	kp_LB650 = 695.0
	ki_LB650 = 5.0e7
	feed_forward_magnitude_LB650 = -0.002*RoverQ_LB650*QL_LB650*beam_loading_fraction
	feed_forward_time_error_LB650 = 0 + timing_errors
	
	QL_HB650 = 11.5e6
	G_HB650 = 260.
	RoverQ_HB650 = 609
	Q0_HB650 = 2.e10
	kp_HB650 = 707.0
	ki_HB650 = 5.0e7
	feed_forward_magnitude_HB650 = -0.002*RoverQ_HB650*QL_HB650*beam_loading_fraction
	feed_forward_time_error_HB650 = 0 + timing_errors
	
	T = 1.0

	lattice = []

	cell_parameters = numpy.loadtxt('cell_parameters.csv',delimiter=',',skiprows=0)
	k = 0
	z = 0
	last_cavity = 0
	ele = 1
	m = 1
	for i in range(0,len(table)):

		time_error = 0

		line = table[i]
		if not line:
			continue
		
		else:
			
			if (line[0] == 'cell'):
				V = cell_parameters[k,1]*1.0e6
				Phi = cell_parameters[k,2]
				dz = float(line[1])/100.0
				z = z + dz
				L = dz
				

				if line[6] == '1':
					f = 162.5e6
					element = rf_cavity(f,f,L,V,Phi,z-L/2,k,ele,'buncher.'+str(m),
							QL = QL_B, RoverQ = RoverQ_B, kp = kp_B, ki = ki_B,
							phase_error = numpy.random.uniform(-0.1,0.1)*error_flag, voltage_error = numpy.random.uniform(-0.001,0.001)*V*error_flag,
							feed_forward_magnitude = feed_forward_magnitude_B, feed_forward_phase = Phi, 
							feed_forward_time_error = timing_errors, feed_forward_width = feed_forward_width)
					m = m + 1
					lattice.append(element)

					if lattice[-1].type == 'buncher.1':
						lattice[-1].feed_forward_magnitude = -0.005*RoverQ_B*QL_B*beam_loading_fraction
						ele = ele + 1
						lattice.append(chopper(0.4,ele,z))
						
				if line[6] == '2':
					f = 162.5e6
					element = rf_cavity(f,f,L,V,Phi,z-L/2,k,ele,'hwr',
							QL = QL_HWR, RoverQ = RoverQ_HWR, kp = kp_HWR, ki = ki_HWR,
							phase_error = numpy.random.uniform(-0.1,0.1)*error_flag, voltage_error = numpy.random.uniform(-0.001,0.001)*V*error_flag,
							feed_forward_magnitude = feed_forward_magnitude_HWR, feed_forward_phase = Phi, 
							feed_forward_time_error = timing_errors, feed_forward_width = feed_forward_width)
						
					lattice.append(element)

					
				if line[6] == '3':
					f = 325.0e6
					element = rf_cavity(f,f,L,V,Phi,z-L/2,k,ele,'ssr1',
							QL = QL_SSR1, RoverQ = RoverQ_SSR1, kp = kp_SSR1, ki = ki_SSR1,
							phase_error = numpy.random.uniform(-0.1,0.1)*error_flag , voltage_error = numpy.random.uniform(-0.001,0.001)*V*error_flag,
							feed_forward_magnitude = feed_forward_magnitude_SSR1, feed_forward_phase = Phi, 
							feed_forward_time_error = timing_errors, feed_forward_width = feed_forward_width)
								
					lattice.append(element)
										
				if line[6] == '4':
					f = 325.0e6
					element = rf_cavity(f,f,L,V,Phi,z-L/2,k,ele,'ssr2',
							QL = QL_SSR2, RoverQ = RoverQ_SSR2, kp = kp_SSR2, ki = ki_SSR2,
							phase_error = numpy.random.uniform(-0.1,0.1)*error_flag, voltage_error = numpy.random.uniform(-0.001,0.001)*V*error_flag,
							feed_forward_magnitude = feed_forward_magnitude_SSR2, feed_forward_phase = Phi, 
							feed_forward_time_error = timing_errors, feed_forward_width = feed_forward_width)

					lattice.append(element)

				if line[6] == '5':
					f = 650.0e6
					element = rf_cavity(f,f,L,V,Phi,z-L/2,k,ele,'lb650',
							QL = QL_LB650, RoverQ = RoverQ_LB650, kp = kp_LB650, ki = ki_LB650,
							phase_error = numpy.random.uniform(-0.1,0.1)*error_flag, voltage_error = numpy.random.uniform(-0.001,0.001)*V*error_flag,
							feed_forward_magnitude = feed_forward_magnitude_LB650, feed_forward_phase = Phi, 
							feed_forward_time_error = timing_errors, feed_forward_width = feed_forward_width)			

					lattice.append(element)
									
				if line[6] == '6':
					f = 650.0e6
					element = rf_cavity(f,f,L,V,Phi,z-L/2,k,ele,'hb650',
							QL = QL_HB650, RoverQ = RoverQ_HB650, kp = kp_HB650, ki = ki_HB650,
							phase_error = numpy.random.uniform(-0.1,0.1)*error_flag, voltage_error = numpy.random.uniform(-0.001,0.001)*V*error_flag,
							feed_forward_magnitude = feed_forward_magnitude_HB650, feed_forward_phase = Phi, 
							feed_forward_time_error = timing_errors, feed_forward_width = feed_forward_width)
						
					lattice.append(element)
					
				k=k+1
				ele = ele + 1

				
			elif (line[0] == 'drift') or (line[0] == 'quad'):
				

					
				L = float(line[1])/100.0
				z = z + L
				element = drift(f,L,z,ele)		
				lattice.append(element)
				ele = ele + 1
				


	return lattice
	



 
cell_dict = {'QWR-2012-02': 1, 'HWRDonut': 2, 'SSR13D_#72': 3, 'SSR2_beta047_v26': 4, 'beta063D_#43': 5, 'beta092': 6}
type_dict = {'QWR-2012-02': 'buncher', 'HWRDonut': 'hwr', 'SSR13D_#72': 'ssr1', 'SSR2_beta047_v26': 'ssr2', 'beta063D_#43': 'lb650', 'beta092': 'hb650'}
QL_dict = {'buncher':4850, 'hwr':2.7e6, 'ssr1':3.7e6, 'ssr2':5.8e6, 'lb650':11.3e6, 'hb650':11.5e6}
Q0_dict = {'buncher':9700, 'hwr':0.5e10, 'ssr1':0.6e10, 'ssr2':0.8e10, 'lb650':1.5e10, 'hb650':2.0e10}
R_dict = {'buncher':530.0, 'hwr':275.0, 'ssr1':242.0, 'ssr2':296.0, 'lb650':375.0, 'hb650':609.0}
kp_dict = {'buncher':1.5, 'hwr':664.0, 'ssr1':455.0, 'ssr2':713.0, 'lb650':695.0, 'hb650':707.0}
#kp_dict = {'buncher':1.0, 'hwr':25.0, 'ssr1':25.0, 'ssr2':25.0, 'lb650':25.0, 'hb650':25.0}
ki_dict = {'buncher':3.0e5, 'hwr':5.0e7, 'ssr1':5.0e7, 'ssr2':5.0e7, 'lb650':5.0e7, 'hb650':5.0e7}
#ki_dict = {'buncher':0, 'hwr':0.0e7, 'ssr1':0.0e7, 'ssr2':0.0e7, 'lb650':0.0e7, 'hb650':0.0e7}

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
						
						
						element = rf_cavity(f,f,L,V,Phi,z-L/2,k,ele,name+'.'+str(m),
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

