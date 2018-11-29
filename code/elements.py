#!/usr/bin/python

import math
import numpy
import os
import scipy.integrate as sint
import time

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

import scipy.interpolate as interp
from numpy import linalg 

## physical constants (assuming H- beam)
m0 = 1.67444377e-27
c0 = 299792458 
q0 = 1.60217662e-19
mc2 = m0*c0**2/q0
pi = numpy.pi


class rf_cavity:
	
	## The RF cavity class defines the parameters for each cavity and computes the RF feedback response.
	## The feedback response updates the beam parameters for each cavity. 
	
	def __init__(self, f0 = 0.0, f = 0.0, L = 0.0, V = 0.0, phase = 0.0, z = 0.0, 
		cavity_index = 0.0, ele = 0.0, name = 'None',
		QL = 1.0e6, RoverQ = 200, kp = 0, ki = 0,
		phase_error = 0, voltage_error = 0,
		feed_forward_magnitude = 0, feed_forward_phase = 0, 
		feed_forward_time_error = 0, feed_forward_width = 500e-6, 
		ki_ff = 0.2, ki_ff_time = 1.0e-8, group_delay = 2.5e-6):
		
		self.f = f
		self.f0 = f0
		self.w = 2*pi*self.f0
		self.QL = QL
		self.V = V
		self.L = L
		self.set_point_phase = phase * numpy.pi/180.
		self.phase = phase
		
		self.RoverQ = RoverQ
		self.R = RoverQ*QL/2.
		self.kp = kp
		self.ki = ki
		self.delay = group_delay
		self.dt = 1.0e-7
		
		self.phase_error = phase_error * numpy.pi/180.
		self.voltage_error = voltage_error
		
		self.feed_forward_magnitude = feed_forward_magnitude
		self.feed_forward_phase = feed_forward_phase * numpy.pi/180.
		self.feed_forward_time_error = feed_forward_time_error
		self.feed_forward_width = feed_forward_width
		self.feed_forward = self.feed_forward_magnitude * numpy.cos(self.feed_forward_phase) + self.feed_forward_magnitude * numpy.sin(self.feed_forward_phase) * 1j
			
		self.ki_ff = ki_ff
		self.ki_ff_time = ki_ff_time
		self.error_I0 = 0.
		self.error_Q0 = 0.
		self.error_I1 = 0.
		self.error_Q1 = 0.
		self.dI = 'init'
		self.dQ = 'init'
		self.kp_ff = self.ki_ff
		
		self.z = z
		self.name = name
		self.type = 'rf_cavity'
		self.index = ele
		self.cavity_index = cavity_index		
	
	def update_parameters(self):
		self.feed_forward = self.feed_forward_magnitude * numpy.cos(self.feed_forward_phase) + self.feed_forward_magnitude * numpy.sin(self.feed_forward_phase) * 1j
		self.set_point_phase = self.phase * numpy.pi/180.
	
			
	def propagate_beam(self,beam):
		""" This function computes the beam parameters through the RF cavity.
		
		
		Ths begins with a calculation of the phase advance through the first half of the cavity 
		which is treated as a drift. Then the RF cavity simulation is performed with beam loading 
		compensation added as defined by the user. Following this the beam parameters are updated again 
		and then the phase advance through the second half of the cavity is computed using the new 
		beam energy""" 
		
		""" Phase advance calculations """
		delta_phi = self.L/2.0 * self.w/(beam.beta_envelope[-1] * c0)
		delta_E = 0. * delta_phi			
		delta_phi_r = self.L/2.0 * self.w/(beam.beta_reference[-1] * c0)
		delta_E_r = 0. * delta_phi_r

		""" Update the beam parameters """
		beam.update_parameters(delta_E_r, delta_phi_r, delta_E, delta_phi, self.L/2.)

		amplitude_out,phase_out = self.rf_simulation(beam)

		""" Compute the change in energy from the cavity for the reference and for the beam""" 		
		self.delta_E_reference = self.V * numpy.cos(self.set_point_phase)
		self.delta_phi_reference = 0.
		self.delta_E = self.amplitude_response * numpy.cos(self.phase_response + self.set_point_phase + self.phase_error)
		self.delta_phi = 0. * self.delta_E
		
		""" Update the beam parameters """
		beam.update_parameters(self.delta_E_reference,self.delta_phi_reference,self.delta_E,self.delta_phi, 0.)

		""" Second phase advance calculations """
		delta_phi = self.L/2.0 * self.w/(beam.beta_envelope[-1] * c0)
		delta_E = 0. * delta_phi			
		delta_phi_r = self.L/2.0 * self.w/(beam.beta_reference[-1] * c0)
		delta_E_r = 0. * delta_phi_r

		""" Final update of beam parameters """
		beam.update_parameters(delta_E_r, delta_phi_r, delta_E, delta_phi, self.L/2.)

		#self.update_feed_forward()

		return beam
		
	
	def rf_simulation(self,beam):
		
		""" Get the beam phase for the beam-loading calculation """ 
		beam_phase = beam.phase_envelope[-1]
		
		""" Compute the complex beam_current using the beam phase and the cavity set point phase """
		beam_current = beam.current_vector * numpy.cos(beam_phase + self.set_point_phase + self.phase_error) + beam.current_vector * numpy.sin(beam_phase + self.set_point_phase + self.phase_error) * 1j
		
		""" Construct the complex feed-forward table """
		feed_forward_vector = numpy.zeros(len(beam_phase))
		for i in range(0,len(feed_forward_vector)):
			if (i*beam.dt >= self.feed_forward_time_error) and (i*beam.dt <= (self.feed_forward_width + self.feed_forward_time_error)):
				feed_forward_vector[i] = 1.
		
		self.feed_forward_vector = feed_forward_vector

		feed_forward_vector = feed_forward_vector.copy() * self.feed_forward 
	
		""" Create the simulation time vector using the timestep information defined in the initialization """
		beam_time = beam.time
		t0 = beam_time[0]
		t1 = beam_time[-1]
		n = t1 / self.dt + 1
		sim_time = numpy.linspace(t0,t1,n)
		
		""" Create the simulation vectors by interpolating from the beam timestep to the simulation timestep """
		beam_current_sim = numpy.interp(sim_time,beam.time,beam_current * beam.mask)
		feed_forward_sim = numpy.interp(sim_time,beam.time,feed_forward_vector) / self.R
		mask = numpy.interp(sim_time,beam.time,beam.mask)

		""" Compute the number of simulation steps for the feedback delay """
		delay_steps = int(self.delay / self.dt)	
	
		""" Initialize the complex cavity field vector """ 
		V_cav = numpy.zeros(len(sim_time)) + 1j * numpy.zeros(len(sim_time))
		
		""" Initialize the drive current and the error for the feedback controller """
		drive_current = 0 + 0 * 1j
		int_error = 0 + 0 * 1j
		
		""" Iterate through each timestep and calculate the cavity field at each point """
		for i in range(1,len(V_cav)):
			current = beam_current_sim[i] + drive_current + feed_forward_sim[i]
	
			V_cav[i] = V_cav[i-1] + self.dt * (self.RoverQ / 4. * self.w * current - self.w / (2. * self.QL) * V_cav[i-1])

			if i > delay_steps:
				error = - V_cav[i-delay_steps] / self.R 
				int_error = int_error + error * self.dt
	
				drive_current = self.kp * error  +  self.ki * int_error
	
		""" Compute the amplitude and phase of the cavity """ 
		amplitude_out = numpy.abs((self.V + self.voltage_error + V_cav * mask))
		phase_out = numpy.angle((self.V + self.voltage_error + V_cav * mask))
		
		""" Transform the result of the time domain simulation to the timestep of the beam """
		amplitude_out_reduced = numpy.interp(beam.time,sim_time,amplitude_out)
		phase_out_reduced = numpy.interp(beam.time,sim_time,phase_out)
				
		""" Compute the effective amplitude and phase response seen by the beam. Note we can use parmela phase convention for comparison """
		self.phase_response = phase_out_reduced + beam_phase*self.f/self.f0
		self.amplitude_response = amplitude_out_reduced
		
		return amplitude_out,phase_out
		
		
	def update_feed_forward(self):
	
		if self.dI == 'init':
			if self.name == 'ssr1.1':
				print self.feed_forward_width, self.feed_forward_time_error, self.feed_forward_I, self.feed_forward_Q
			self.feed_forward_I = self.feed_forward_I - self.ki_ff * numpy.sum(self.I_out[0:len(self.I_out)/2])*300.
			self.feed_forward_Q = self.feed_forward_Q - self.ki_ff * numpy.sum(self.Q_out[0:len(self.Q_out)/2])*300.
			
			#self.feed_forward_I = self.feed_forward_I + self.ki_ff * (self.error_I1)*0.01
			#self.feed_forward_Q = self.feed_forward_Q + self.ki_ff * (self.error_Q1)*0.01
			
			self.feed_forward_width = self.feed_forward_width + self.ki_ff * (self.error_I1 + self.error_Q1)*1.0e-10
			#self.feed_forward_time_error = self.feed_forward_time_error + self.ki_ff * (self.error_I1 + self.error_Q1)*0.1e-10

			self.error_I0 = self.error_I1.copy()
			self.error_Q0 = self.error_Q1.copy()
			self.dI = self.error_I1 - self.error_I0
			self.dQ = self.error_Q1 - self.error_Q0
	
		
			
		else: 

			self.dI = self.error_I1 - self.error_I0
			self.dQ = self.error_Q1 - self.error_Q0	

			self.feed_forward_I = self.feed_forward_I - self.ki_ff * numpy.sum(self.I_out[0:len(self.I_out)/2])*300.
			self.feed_forward_Q = self.feed_forward_Q - self.ki_ff * numpy.sum(self.Q_out[0:len(self.Q_out)/2])*300.
			
			#self.feed_forward_I = self.feed_forward_I + self.ki_ff * self.dI*0.1
			#self.feed_forward_Q = self.feed_forward_Q + self.ki_ff * self.dQ*0.1
			
				
			self.feed_forward_width = self.feed_forward_width - self.ki_ff * (self.dI + self.dQ)*2e-10
			#self.feed_forward_time_error = self.feed_forward_time_error + self.ki_ff * (self.dI + self.dQ)*2.0e-10

			self.error_I0 = self.error_I1.copy()
			self.error_Q0 = self.error_Q1.copy()
			
		
		if self.name == 'ssr1.1':
				print self.feed_forward_width, self.feed_forward_time_error, self.feed_forward_I, self.feed_forward_Q
			
		#self.feed_forward_time_error =  self.feed_forward_time_error - (self.error_I + self.error_Q)*self.ki_ff_time*1.0e-6		
	
		return
		


class magnet: 
	""" This class is to handle magnets. 
	
	
	Instead of doing dipole and quadrupoles separately, the code treats all magnets the same, it 
	is up to the user to properly specify the magnetic field components in both directions up to sextupole
	components """ 
	
	def __init__(self,f0,L,Bx,By,rotation,z,ele,type):
			
		self.f0 = f0
		self.L = L
		self.Bx = Bx
		self.By = By
		self.rotation = rotation
		self.z = z
		self.index = ele
		self.name = 'magnet'
		self.type = type
		self.matrix = transfer_matrix(self)
	
	def transfer_matrix(self):
		
		
		return matrix
		
		
	def propagate_beam(self,beam):

		
		return beam
		

class chopper:
	""" The copper element is a simple way to change the beam current.
	
	
	in this class the only thing that happens is the beam current is reduced by the 
	factor specified in the definition of the chopper. In the future other calculations can
	be added """
	
	def __init__(self,ratio,ele,z):
		""" Initialize the chopper elements and define parameters.
		
		ratio is the ratio of the output current to the input current
		name is always 'chopper' 
		type is always 'none' 
		index is the element number in the machine 
		z is the z position """
		
		self.ratio = ratio 
		self.name = 'chopper'
		self.type = 'none'
		self.index = ele
		self.z = z
		
	def propagate_beam(self,beam):
		""" Change the beam current based on chopping ratio """

		beam.current = self.ratio * beam.current
		
		return beam
		

class drift: 
	""" Drift space class for the simulation code
	
	
		The drift space element is quite simple. It calculates the phase advance of the beam 
		and of the reference particle based on the velocity of the beam and returns that information
		to the beam class."""

	def __init__(self, f0 = 0.0, L = 0.0, z = 0.0, ele = 0):
		"""Drift class initialization and definitions:
	
	
		L is the length of the drift space 
		f0 is the frequency of the drift space (used in the phase calculation)
		w0 is the angular frequency
		z is the position of the drift space in the linac
		name is 'drift'
		index is a specified element number 
		type is 'none' in the future type may be used to choose whether or not to calculate space-charge effects"""
		
		self.L = L
		self.f0 = f0
		self.w0 = 2*pi*self.f0
		self.z = z
		self.name = 'drift'
		self.index = ele
		self.type = 'none'
		
	def propagate_beam(self,beam):
		"""Function to move a beam through the drift:
	
		
		The phase advance of the beam and of the reference particle are calculated in this function and passed to the beam
		by telling the beam to update its parameters."""
		
		""" Compute phase advance of reference particle and beam """
		delta_phi = self.L*self.w0/(beam.beta_envelope[-1]*c0)
		delta_E = 0. * delta_phi			
		delta_phi_r = self.L*self.w0/(beam.beta_reference[-1]*c0)
		delta_E_r = 0. * delta_phi_r

		""" Update the beam parameters """
		beam.update_parameters(delta_E_r, delta_phi_r, delta_E, delta_phi, self.L)
		
		return beam
	

	
class beam:
	""" Beam class for the simulation code
	
	
		The beam class keeps track of the beam energy, phase, reference particle, and the 
		beam envelope. The beam envelope is the energy and phase of the beam as a function
		of time during the RF pulse. The energy and phase of the beam are relative to the 
		reference energy and phase. This allows for the introduction of cavity perturbations defined 
		in the cavity class. While the phase and energy vectors are absolute, the phase envelope is only relative
		to the beam phase. Note that all inputs are in degrees but calculations are in radians. """
		

	def __init__(self,phi0,E0,E0_r,I0,n,t0,t1,delay,mc2,hold_arrays = 1,profile = 'top_hat',profile_order = 20.,filename = None):
		
		self.phase = [phi0*numpy.pi/180]
		self.energy = [E0]
		self.reference_phase = [0.0]
		self.reference_energy = [E0_r]

		self.current = I0
		self.time = numpy.linspace(t0,t1,n)
		self.delay = delay
		self.n = n
		
		self.mask = numpy.zeros(len(self.time)) + 1.
		self.profile = profile 
		self.profile_order = profile_order
		
		if self.profile == 'top_hat':
			for i in range(0,len(self.time)):
				if self.time[i] >= delay:
					self.mask[i] = 1.0
		
		if self.profile == 'inverse_polynomial':
			x = numpy.linspace(-1.,1.,n)
			y = 1./(1. + 1*x**self.profile_order)
			self.mask = y * self.mask
			
		if self.profile == 'import':
			run_data = numpy.loadtxt(filename,delimiter=',',skiprows = 1) 
			t = run_data[:,0]/1000. - run_data[0,0]/1000. + self.delay
			I = run_data[:,1]
			self.time = numpy.linspace(0,t[-1]+self.delay,n)
			I_all = numpy.interp(self.time,t,I)
			self.mask = I_all / numpy.min(I_all)
		
		self.current_vector = self.current * self.mask		
		
		self.dt = numpy.mean(numpy.diff(self.time))
		
		self.phase_envelope = [phi0*numpy.ones(n)*numpy.pi/180.]
		self.energy_envelope = [E0*numpy.ones(n)]
		
		self.gamma = [self.energy[0]/mc2 + 1.]
		self.beta = [numpy.sqrt(1-1/self.gamma[0]**2)]
		
		self.gamma_envelope = [self.energy_envelope[0]/mc2 + 1.]
		self.beta_envelope = [numpy.sqrt(1-1/self.gamma_envelope[0]**2)]
		
		self.gamma_reference = [self.reference_energy[0]/mc2 + 1.]
		self.beta_reference = [numpy.sqrt(1-1/self.gamma_reference[0]**2)]

		self.z = [0]
		self.s = 0
		self.hold_arrays = hold_arrays
		
		self.name = 'beam'

		
	def update_parameters(self,delta_E_r,delta_phi_r,delta_E,delta_phi,L):
		""" Update the beam parameters 
		
		
		This increases the length of the list after each call of update_parameters. 
		Here the energy, phase, velocity, and gamma of the reference particle, the beam envelope
		and the beam centroid are updated and stored """
		
		""" Update the beam envelope parameters """
		self.energy_envelope.append(self.energy_envelope[-1] + delta_E)
		self.phase_envelope.append(self.phase_envelope[-1] + delta_phi - delta_phi_r)
		self.gamma_envelope.append(self.energy_envelope[-1]/mc2 + 1.)
		self.beta_envelope.append(numpy.sqrt(1-1/self.gamma_envelope[-1]**2))
		
		""" Update the reference particle parameters """
		self.reference_energy.append(self.reference_energy[-1] + delta_E_r)
		self.reference_phase.append(self.reference_phase[-1] + delta_phi_r)
		self.gamma_reference.append(self.reference_energy[-1]/mc2 + 1.)
		self.beta_reference.append(numpy.sqrt(1-1/self.gamma_reference[-1]**2))
		
		""" Update the beam centroid parameters (nominally when the RF is at steady state """ 
		self.energy.append(self.energy[-1] + delta_E[int(self.n)/2])
		self.phase.append(self.phase[-1] + delta_phi[int(self.n)/2])
		self.gamma.append(self.energy[-1]/mc2 + 1.)
		self.beta.append(numpy.sqrt(1-1/self.gamma[-1]**2))
		
		""" Update the position vector """
		self.s += L
		self.z.append(self.s)
		
		""" Update the current vector needed for chopping """ 
		self.current_vector = self.current * self.mask		

		
	
	def initialize_output_files(self,tape1 = 'tape1.txt', tape2 = 'tape2.txt'):
		""" Initialize the output files for saving the run data """
		self.tape1 = open(tape1, 'wa')
		self.tape2 = open(tape2, 'wa')
		
	def write_tape1(self,bl_element):
		""" Write the beam and reference particle energy and phase to a file """
		header =  bl_element.name + ',' + bl_element.type + ',' + str(bl_element.index) + ',' + 'z='+str(bl_element.z) + '\n'
		output_tape1 = str(self.reference_energy[-1]) + ',' + str(self.reference_phase[-1]*180./numpy.pi) +',' + str(self.energy[-1]) + ',' + str(self.phase[-1]*180./numpy.pi) +'\n'
		self.tape1.write(header)
		self.tape1.write(output_tape1)
		
		
	def write_tape2(self,bl_element):
		""" Write the vector data to a file. This is quite slow so only use if you really need it. """
		header =  bl_element.name + ',' + bl_element.type + ',' + str(bl_element.index) + ',' + 'z='+str(bl_element.z) + '\n'
		output_tape2_a = str(self.energy_envelope[-1]) + ',\n'
		output_tape2_b = str(self.phase_envelope[-1]*180./numpy.pi) + '\n'
		self.tape2.write(header)
		for element in self.energy_envelope[-1]: self.tape2.write(str(element) + ',')
		self.tape2.write('\n')
		for element in self.phase_envelope[-1]: self.tape2.write(str(element*180./numpy.pi))
		self.tape2.write('\n')
		
	
	def make_arrays(self):
		""" Convert the lists to numpy arrays: 
		
		
		This is a convence function so that the data can be plotted immediately after the simulation run """
		self.energy = numpy.asarray(self.energy)
		self.phase = numpy.asarray(self.phase)
		self.energy_envelope = numpy.asarray(self.energy_envelope)
		self.phase_envelope = numpy.asarray(self.phase_envelope)
		self.reference_energy = numpy.asarray(self.reference_energy)
		self.reference_phase = numpy.asarray(self.reference_phase)
		self.z = numpy.asarray(self.z)






class simulation:
	def __init__(self,lattice):
		self.lattice = lattice 
		self.name = 'simulation'
		

	def plot_elements(self):
	
	

		host=host_subplot(111, axes_class=AA.Axes)
		#plt.subplots_adjust(right=0.75)
		par1 = host.twinx()
		#par2 = host.twinx()
	
		#offset=75
		#new_fixed_axis = par2.get_grid_helper().new_fixed_axis
		#par2.axis["right"] = new_fixed_axis(loc="right",axes=par2,offset=(offset, 0))
		#par2.axis["right"].toggle(all=True)
	
		E = numpy.loadtxt('Parmela_Energy.csv',delimiter=',')
		sync_phase = []
		v = []
		pos = []
		for i in range(0,len(self.lattice)):
			element = self.lattice[i]
			

			if element.type == 'rf_cavity':
				title = element.name.split('.')
				name = title[0]
				ele = float(title[1])
				
				x0 = element.z
				L = element.L
				x = numpy.asarray([x0, x0])
				z = x0
				y = numpy.asarray([-35, -55])
				
				sync_phase.append(element.phase)
				v.append(element.V/1.0e6)
				pos.append(z)
				
				if 'buncher' in name:
					if ele == 1:
						par1.plot(x,y,'-r',linewidth = 1.5,label = name)
					else:	
						par1.plot(x,y,'-r',linewidth = 1.5)
				elif 'hwr' in name:
					if ele == 1:
						par1.plot(x,y,'-g',linewidth = 1.5,label = name)
					else:	
						par1.plot(x,y,'-g',linewidth = 1.5)			
				elif 'ssr1' in name: 
					if ele == 1:
						par1.plot(x,y,'-b',linewidth = 1.5,label = name)
					else:	
						par1.plot(x,y,'-b',linewidth = 1.5)					
				elif 'ssr2' in name:
					if ele == 1:
						par1.plot(x,y,'-c',linewidth = 1.5,label = name)
					else:	
						par1.plot(x,y,'-c',linewidth = 1.5)				
				elif 'lb650' in name:
					if ele == 1:
						par1.plot(x,y,'-y',linewidth = 1.5,label = name)
					else:	
						par1.plot(x,y,'-y',linewidth = 1.5)				
				elif 'hb650' in name: 
					if ele == 1:
						par1.plot(x,y,'-m',linewidth = 1.5,label = name)
					else:	
						par1.plot(x,y,'-m',linewidth = 1.5)				
				else:
					if ele == 1:
						par1.plot(x,y,'-k',linewidth = 1.5,label = 'rf_cavity')
					else:	
						par1.plot(x,y,'-k',linewidth = 1.5)
						
								
		#plt.ylim([-2,2])
		
		par1.plot(numpy.asarray(pos),numpy.asarray(sync_phase),'-',color = (1.0,0.25,0.25),linewidth = 2.0)
		
		host.plot(E[:,0]/100.,E[:,1],color = (0.25,0.25,1.0), linewidth = 2.0)
		#host.plot(pos,v,'-',color = (0.25,0.25,1.0),linewidth = 2.0)
		host.set_xlim([0,element.z])
		#host.set_ylim([-3,3])
		par1.set_ylim([-90,0])
		#host.set_ylim([0,8])
		host.set_xlabel("Position [m]")
		par1.set_ylabel('Synchronous phase [deg]',color="red")
		host.set_ylabel('Energy [MeV]',color="Blue")
		plt.grid()
		plt.draw()
		#plt.legend(loc=0)
		plt.show() 
	
	
	def last_module_correction(self,error,gain):

		list = ['hb650.19','hb650.20','hb650.21','hb650.22','hb650.23','hb650.24']
		number = {'hb650.19':113,'hb650.20':114,'hb650.21':115,'hb650.22':116,'hb650.23':118,'hb650.24':119}
		for i in range(0,len(self.lattice)):
			if self.lattice[i].name in list:
				
				dEdPhi = - self.lattice[i].V * numpy.sin(self.lattice[i].set_point_phase + self.lattice[i].phase_error)
				dE = error 
				dPhi = dE / dEdPhi / 6. # * cavity_gain
				
				self.lattice[i].phase_error = self.lattice[i].phase_error + dPhi
	
		
	def run(self,beam,write_tape1 = 1, write_tape2 = 0, tape1_name = 'tape1.txt', tape2_name = 'tape2.txt'):
		if (write_tape1 == 1) or (write_tape2 == 1):
			beam.initialize_output_files(tape1 = tape1_name, tape2 = tape2_name)
		for i in range(0,len(self.lattice)):
			beam = self.lattice[i].propagate_beam(beam)
			
			if write_tape1 == 1:
				beam.write_tape1(self.lattice[i])
			
			if write_tape2 == 1:
				beam.write_tape2(self.lattice[i])
		
		beam.make_arrays()
		beam.tape1.close()
		beam.tape2.close()

	
		