#!/usr/bin/env python3

import math
import numpy

import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams.update({'font.size': 16})

import elements 
import make_lattice

## Constants and parameters
pi = numpy.pi
m0 = 1.67444377e-27
c0 = 299792458
q0 = 1.60217662e-19
mc2 = m0*c0**2/q0

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Linac RF simulation with parameters from within linac.py")
    parser.add_argument("-p", "--plot", action="store_true", default=False, help="Show plots")

    args = parser.parse_args()

    ## This makes the lattice using the parser
    lattice0 = make_lattice.PIP_II_cdr(0.0,0.0e-6,515e-6,0,0,0,0)

    ## This creates the simulation
    sim0 = elements.simulation(lattice0)

    ## This creates a beam that is
    initial_energy = 2.1e6  #eV
    initial_reference_energy = 2.1e6  #eV
    beam_current = -0.005 #A
    number_of_slices = 500 #This is the number of beam slices to simulate along the beam pulse
    start_time = 0 #Beam-pulse start time
    stop_time = 500.0e-6 #Beam-pulse stop time
    beam_delay = 0 #This is for feed-forward compensation, is not being used in this run
    beam_mc2 = mc2

    beam1 = elements.beam(0,initial_energy,initial_reference_energy,beam_current,
        number_of_slices,start_time,stop_time,beam_delay,beam_mc2,
        profile='top-hat',filename = 'current_profile.csv')


    ## This runs the simulation
    print("Running Simulation...")
    sim0.run(beam1)
    print("Finished running simulation.")

    ## Plotting of the simulation results
    if args.plot:

        plt.figure()
        plt.plot(beam1.z,(beam1.energy_envelope[:,5] - beam1.reference_energy)/1.0e3,linewidth = 2.0,label = r'5$\mu$s Slice')
        plt.plot(beam1.z,(beam1.energy_envelope[:,15] - beam1.reference_energy)/1.0e3,linewidth = 2.0,label = r'15$\mu$s Slice')
        plt.plot(beam1.z,(beam1.energy_envelope[:,40] - beam1.reference_energy)/1.0e3,linewidth = 2.0,label = r'40$\mu$s Slice')
        plt.plot(beam1.z,(beam1.energy_envelope[:,60] - beam1.reference_energy)/1.0e3,linewidth = 2.0,label = r'60$\mu$s Slice')
        plt.xlabel('Position [m]')
        plt.ylabel(r'$\Delta$ E [keV]')
        plt.title('Beam Energy along the linac relative to reference phase')
        plt.grid()
        plt.xlim([0,beam1.z[-1]])
        plt.legend(loc=0)

        plt.figure()
        plt.plot(beam1.z,(beam1.phase_envelope[:,5])*180./numpy.pi,linewidth = 2.0,label = r'5$\mu$s Slice')
        plt.plot(beam1.z,(beam1.phase_envelope[:,15])*180./numpy.pi,linewidth = 2.0,label = r'15$\mu$s Slice')
        plt.plot(beam1.z,(beam1.phase_envelope[:,40])*180./numpy.pi,linewidth = 2.0,label = r'40$\mu$s Slice')
        plt.plot(beam1.z,(beam1.phase_envelope[:,60])*180./numpy.pi,linewidth = 2.0,label = r'60$\mu$s Slice')
        plt.xlabel('Position [m]')
        plt.ylabel(r'$\Delta\phi$ [deg]')
        plt.title('Beam phase along the linac relative to reference phase')
        plt.grid()
        plt.xlim([0,beam1.z[-1]])
        plt.legend(loc=0)

        plt.figure()
        plt.plot(beam1.time*1.0e6,(beam1.energy_envelope[-1,:] - beam1.reference_energy[-1])/1.0e3, linewidth = 2.0)
        plt.xlim([0,100])
        plt.xlabel(r'Time [$\mu$s]')
        plt.ylabel(r'$\Delta$ E [keV]')
        plt.title('Energy modulation along the beam-pulse')
        plt.grid()

        plt.figure()
        plt.plot(beam1.time*1.0e6,beam1.phase_envelope[-1,:]*180./numpy.pi, linewidth = 2.0)
        plt.xlim([0,100])
        plt.xlabel(r'Time [$\mu$s]')
        plt.ylabel(r'$\Delta \phi$ [deg]')
        plt.title('Phase modulation along the beam-pulse')
        plt.grid()

        plt.figure()
        plt.plot((beam1.phase_envelope[-1,:])*180./numpy.pi,(beam1.energy_envelope[-1,:] - beam1.reference_energy[-1])/1.0e3,'o', linewidth = 2.0)
        plt.xlabel(r'$\Delta \phi$ [deg]')
        plt.ylabel(r'$\Delta$ E [keV]')
        plt.title('Effective phase space of the beam pulse')
        plt.grid()


        plt.show()

