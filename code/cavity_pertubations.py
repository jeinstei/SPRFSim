#!/usr/bin/env python3

import math
import numpy
import os
import scipy.integrate as sint
import time

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams.update({'font.size': 16})

parmela_data = numpy.loadtxt('energy_parmela.csv',delimiter=',')



#from simulation_elements import *
from elements import *

import make_lattice

## Constants and parameters             
pi = numpy.pi           
m0 = 1.67444377e-27
c0 = 299792458 
q0 = 1.60217662e-19
mc2 = m0*c0**2/q0





E0 = 2.1e6
m0 = 1.67444377e-27
c0 = 299792458 
q0 = 1.60217662e-19
mc2 = m0*c0**2/q0


def monte_carlo_current_errors(n,current_profile = 'top_hat',beam_loading_compensation = 1)

        k = 0
        
        while k < n:
        
                phi0 = 0.0 
                current = numpy.random.uniform(0.0015,0.0025)
                        
                if current_profile == 'import'
                        
                        if beam_loading_compensation == 1:
                                lattice0 = make_lattice.PIP_II(1.0,3.0e-6,0)

                        else: 
                                lattice0 = make_lattice.PIP_II(0.0,0.0e-6,0)
                        
                        beam0 = beam(0,2.1e6,2.1e6,current,500,0.0e-6,0.5e-3,0.0e-6,mc2,0,profile='import',filename='current_profile.csv')

                else: 
                        if beam_loading_compensation == 1:
                                lattice0 = make_lattice.PIP_II(0.0,0.0e-6,0)

                        else: 
                                lattice0 = make_lattice.PIP_II(0.0,0.0e-6,0)
                
                        beam0 = beam(0,2.1e6,2.1e6,current,500,0.0e-6,0.5e-3,0.0e-6,mc2,0,profile='top_hat')

                
                sim0 = simulation(lattice0)             
                sim0.run(beam0)

                phi = (beam0.phase_envelope[-1,:])*180./numpy.pi
                E = (beam0.energy_envelope[-1,:] - beam0.reference_energy[-1])/1.0e3
                t = beam0.time
                
                data = numpy.column_stack([t,phi,E])
        
                filename = 'output_current_errors.'+str(k)+'.csv'

                numpy.savetxt(filename,data,delimiter=',')
                k = k + 1
                print(k)



def monte_carlo_cavity_errors(n,current_profile = 'top_hat',beam_loading_compensation = 1)

        k = 0
        while k < n:
        
                phi0 = 0.0 
                current = 0.002 
                        
                if current_profile == 'import'
                        
                        if beam_loading_compensation == 1:
                                lattice0 = make_lattice.PIP_II(1.0,3.0e-6,1)

                        else: 
                                lattice0 = make_lattice.PIP_II(0.0,0.0e-6,1)
                        
                        beam0 = beam(0,2.1e6,2.1e6,current,500,0.0e-6,0.5e-3,0.0e-6,mc2,0,profile='import',filename='current_profile.csv')

                else: 
                        if beam_loading_compensation == 1:
                                lattice0 = make_lattice.PIP_II(0.0,0.0e-6,1)

                        else: 
                                lattice0 = make_lattice.PIP_II(0.0,0.0e-6,1)
                
                        beam0 = beam(0,2.1e6,2.1e6,current,500,0.0e-6,0.5e-3,0.0e-6,mc2,0,profile='top_hat')

                
                sim0 = simulation(lattice0)             
                sim0.run(beam0)

                phi = (beam0.phase_envelope[-1,:])*180./numpy.pi
                E = (beam0.energy_envelope[-1,:] - beam0.reference_energy[-1])/1.0e3
                t = beam0.time
                
                data = numpy.column_stack([t,phi,E])
        
                filename = 'output_cavity_errors.'+str(k)+'.csv'

                numpy.savetxt(filename,data,delimiter=',')
                k = k + 1
                print(k)



def cavity_scan(parameter,output_file):
        
        dE_vec = []
        dPhi_vec = []

        lattice0 = make_lattice.PIP_II_cdr(0.0,0.0e-6,0)

        for i in range(0,len(lattice0)):

                if lattice0[i].type == 'rf_cavity':
                        print(lattice0[i].name)
                        if parameter == 'voltage'
                                lattice0[i].voltage_error = 0.01 * lattice0[i].V
                        
                        elif parameter == 'phase'
                                lattice0[i].phase_error = 1.0 * pi / 180.
                                
                        sim0 = simulation(lattice0)
                        beam0 = beam(0,2.1e6,2.1e6,0.002,515,0.0e-6,0.515e-3,0.0e-6,mc2,0,profile='top_hat')
                        sim0.run(beam0)
                
                        dE = beam0.energy[-1] - beam0.reference_energy[-1]
                        dPhi = beam0.phase[-1] - beam0.reference_phase[-1]
                
                        dE_vec.append(dE)
                        dPhi_vec.append(dPhi)
                        
                        lattice0[i].voltage_error = 0.0
                        lattice0[i].phase_error = 0.0

        output = numpy.column_stack([numpy.asarray(dE_vec), numpy.asarray(dPhi_vec)])
        numpy.savetxt(output_file,output,delimiter=',')






