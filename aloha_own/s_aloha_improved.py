import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import csv
import scipy.special
import math
import time
import random
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linear_sum_assignment
np.set_printoptions(precision=2)

Re = 6378e3	# Radius of the earth [m]
G = 6.67259e-11	# Universal gravitational constant [m^3/kg s^2]
Me = 5.9736e24	# Mass of the earth
k = 1.38e-23	# Boltzmann's constant
c = 299792458	# Speed of light [m/s]
eff = 0.6 	# Efficiency of the parabolic antenna

###### S-ALOHA parameters
rate=0.2
nodes=6
num_packets=20
t_start=0
t_end=200
pkt_duration=1e-2
cw_min=16
cw_max=1024
max_attempts=int(np.log2(cw_max/cw_min))
slots_per_sec= 1/pkt_duration
t_slot_end=int(slots_per_sec*t_end)

base_lat=12.9716
base_long=77.5946
#######		GEO satellites
N_geos = 3
h_geo = 35786e3
inclination_geo = 0

random_rotation = False	# Set to True to rotate the orbital planes randomly at the beginning of the simulation

specific_constellation = "IridiumNEXT"	# Select the constellation design among the options provided below


if specific_constellation =="Kepler":
	print("Using Kepler constellation design")
	P = 7							# Number of orbital planes
	h = 600e3*np.ones(P)					# Altitude of deployment for each orbital plane (set to the same altitude here)
	inclination = 98.6					# Inclination angle for the orbital planes,
	N_p = 20						# Number of satellites per orbital plane
	N = N_p*P
	Walker_star = True					# True for Walker star and False for Walker Delta

elif specific_constellation =="IridiumNEXT":
	print("Using Iridium NEXT constellation design")
	P = 6							# Number of orbital planes
	h = 780e3*np.ones(P)					# Altitude of deployment for each orbital plane (set to the same altitude here)
	inclination = 86.4					# Inclination angle for the orbital planes,
	N_p = 11						# Number of satellites per orbital plane
	N = N_p*P
	Walker_star = True					# True for Walker star and False for Walker Delta

elif specific_constellation =="OneWeb":
	print("Using OneWeb constellation design")
	P = 18							# Number of orbital planes
	h = 1200e3*np.ones(P)					# Altitude of deployment for each orbital plane (set to the same altitude here)
	inclination = 86.4					# Inclination angle for the orbital planes,
	N = 648							# Total number of satellites
	N_p = int(N/P)
	Walker_star = True
elif specific_constellation =="Starlink":			# Phase 1 550 km altitude orbit shell
	print("Using Starlink constellation design")
	P = 72							# Number of orbital planes
	h = 550e3*np.ones(P)					# Altitude of deployment for each orbital plane (set to the same altitude here)
	inclination = 53					# Inclination angle for the orbital planes,
	N = 1584						# Total number of satellites
	N_p = int(N/P)
	Walker_star = False					# True for Walker star and False for Walker Delta
else:
	print("Using homemade constellation design")
	#######	Homemade NGEO constellation design
	P = 10			# Number of orbital planes
	N_p = 10		# Number of satellites per orbital plane
	N = N_p*P		# Total number of satellites
	h = 700e3*np.ones(P)	# Altitude of deployment for each orbital plane (set to the same altitude here)
	inclination = 80	# Inclination angle for the orbital planes, set to 90 for Polar
	Walker_star = True	# Set to True for Walker star and False for Walker Delta

class comm_params:
	def  __init__(self,frequency, bandwidth, Ptx, Gtx, Grx, pointingLoss, noiseFigure, noiseTemperature):
		self.f = frequency
		self.B = bandwidth
		self.Ptx = Ptx
		self.Ptx_db = 10*math.log10(self.Ptx)
		self.Gtx = Gtx
		self.Grx = Grx
		self.G =  self.Gtx+self.Grx - pointingLoss
		self.No = 10*math.log10(self.B*k)+noiseFigure + 10* math.log10(290+(noiseTemperature-290) *(10**(-noiseFigure/10)))
	def  __repr__(self):
		return '\n Carrier frequency = {} GHz\n Bandwidth = {} MHz\n Transmission power = {} W\n Gain per antenna: Tx {}  Rx {}\n Total antenna gain = {} dB\n Noise power = {} dBW'.format(
	self.f/1e9,
	self.B/1e6,
	self.Ptx,
	'%.2f'%self.Gtx,
	'%.2f'%self.Grx,
	'%.2f'%self.G,
	'%.2f'%self.No,
	)

class orbital_plane:
	def  __init__(self,h, longitude, inclination, n_sat, min_elev):
		self.h = h							# Altitude of deployment
		self.longitude = longitude					# longitude angle within the orbital plane [radians]
		self.inclination = inclination					# Inclination of the orbit
		self.n_sat = n_sat						# Number of satellites in plane
		self.period = 2 * math.pi * math.sqrt((self.h+Re)**3/(G*Me))	# Orbital period
		self.v = 2*math.pi * (h + Re) / self.period			# Orbital velocity
		self.min_elev = math.radians(min_elev)					# Minimum elevation angle for ground comm.
		self.max_alpha = math.acos(Re*math.cos(self.min_elev)/(self.h+Re))-self.min_elev	# Maximum angle at the center of the Earth w.r.t. yaw
		self.max_beta  =math.pi/2-self.max_alpha-self.min_elev		# Maximum angle at the satellite w.r.t. yaw
		self.max_distance_2_ground = Re*math.sin(self.max_alpha)/math.sin(self.max_beta)	# Maximum distance to ground station

	def  __repr__(self):
		return '\n altitude= {}, longitude= {}, inclination= {}, number of satellites= {}, period= {} hours, satellite speed= {}'.format(
	self.h,
	'%.2f'%self.longitude,
	'%.2f'%self.inclination,
	'%.2f'%self.n_sat,
	'%.2f'%(self.period/3600),
	'%.2f'%self.v)


class satellite(object):
	def  __init__(self,in_plane,i_in_plane,h,polar_angle):
		self.in_plane= in_plane						# Orbital plane where the satellite is deployed
		self.i_in_plane = i_in_plane					# Index in orbital plane
		self.x=0							# Cartesian coordinates  (x,y,z)
		self.y=0
		self.z=0
		self.h=h							# Altitude of deployment
		self.polar_angle=polar_angle					# Angle within orbital plane
		self.latitude = (math.pi/2-self.polar_angle+2*math.pi)%(2*math.pi)
		self.queue=[]

	def  __repr__(self):
		return '\n orbital plane= {}, index in plane= {}, pos x= {}, pos y= {}, pos z= {}\n polar angle = {} latitude = {} queue length = {}'.format(
	self.in_plane,
	self.i_in_plane,
	'%.2f'%self.x,
	'%.2f'%self.y,
	'%.2f'%self.z,
	'%.2f'%self.polar_angle,
	'%.2f'%self.latitude,
    len(self.queue))

	def rotate(self, delta_t):			# To rotate the satellites after a period delta_t using the polar angle
		self.polar_angle = (self.polar_angle+2 *math.pi * delta_t / Orbital_planes[self.in_plane].period)%(2*math.pi)
		self.polar_angle = self.polar_angle % (2 *math.pi)
		delta = math.pi/2-Orbital_planes[self.in_plane].inclination
		theta = Orbital_planes[self.in_plane].longitude
		phi = self.polar_angle
		self.x = (Orbital_planes[self.in_plane].h+Re) * (math.sin(phi)*math.cos(theta)+math.cos(phi)*math.sin(delta)*math.sin(theta))
		self.y = (Orbital_planes[self.in_plane].h+Re) * (math.sin(phi)*math.sin(theta)-math.cos(phi)*math.sin(delta)*math.cos(theta))
		self.z = (Orbital_planes[self.in_plane].h+Re) * math.cos(phi)*math.cos(delta)

'''
Every user is defined as a Node, with receive(), transmit(), and generate() methods.
receive() : appends Packets to the queue at slot t if arrival in (t-1, t]
transmit() : if the queue is non-empty, and the pkt at the top of queue has its t_time equal to t, the pkt is returned.
            Else, 0 is returned
(don't copy old transmit() )
generate() : generates more Packets by adding more inter_arrival_times
t is the slot number. t * slot_duration is the actual time
'''
from s_aloha_classes import Node, Packet, Network
def initialize_comm_params():
	####### NGEO ISL
	f = 26e9	# Carrier frequency GEO to ground (Hz)
	B = 500e6	# Maximum bandwidth
	Ptx = 1 	# Maximum tansmission power in W
	Gtx = 38.5	# Transmitter antenna gain [dB]
	Grx = 38.5 	# Receiver antenna gain [dB]
	pL = 0.6 	# Pointing loss in dB
	Nf = 2		# Noise figure in dB
	Tn = 290		# Noise temperature in K
	ngeoISL = comm_params(f, B, Ptx,Gtx, Grx, pL, Nf, Tn)
	print(f'NGEO inter-satellite links {ngeoISL}\n')

	####### NGEO to UE
	f = 2e9	# Carrier frequency GEO to ground (Hz)
	B = 30e6	# Bandwidth
	Ptx = 1 	# Maximum tansmission power in W
	Gtx = 30	# Transmitter antenna gain [dB]
	Grx = 0 	# Receiver antenna gain [dB]
	pL = 0.33 	# Pointing loss in dB
	Nf = 7		# Noise figure in dB
	Tn = 290	# Noise temperature in K
	ngeo2ue = comm_params(f, B, Ptx,Gtx, Grx, pL, Nf, Tn)
	print(f'NGEO to UE (DL) link {ngeo2ue}\n')

	####### UE to NGEO (feeder)
	f = 2e9		# Carrier frequency UE to  (Hz)
	B = 0.36e6	# Bandwidth
	Ptx = 0.2 	# Maximum tansmission power in W
	Gtx = 0		# Transmitter antenna gain[dB]
	Grx = 30.5 	# Receiver antenna gain [dB]
	pL = 0.33	 	# Pointing loss in dB
	Nf = 1.5	# Noise figure in dB
	Tn = 50		# Noise temperature in K
	ue2ngeo = comm_params(f, B, Ptx,Gtx, Grx, pL, Nf, Tn)
	print(f'UE to NGEO (UL) link {ue2ngeo}\n')
	return ngeoISL, ngeo2ue, ue2ngeo

def plot_constellation(closest_sat, fig_tag, i):
    Positions_NGEO = np.zeros((N,3))
    for n in range(N):
    	Positions_NGEO[n,:] = [NGEO[n].x/1e6, NGEO[n].y/1e6, NGEO[n].z/1e6]
    fig = plt.figure(fig_tag)
    ax = fig.gca(projection='3d')
    #ax.set_box_aspect((np.ptp(Positions_NGEO[:,0]), np.ptp(Positions_NGEO[:,1]), np.ptp(Positions_NGEO[:,2])))
    area = math.pi * (3**2)
    base_lat=np.radians(12.9716)
    base_long=np.radians(77.5946)
    base_z=Re*np.sin(base_lat)/1e6
    base_y=-Re*np.cos(base_lat)*np.cos(base_long)/1e6
    base_x=-Re*np.cos(base_lat)*np.sin(base_long)/1e6
    ax.scatter(Positions_NGEO[:,0],Positions_NGEO[:,1], Positions_NGEO[:,2], marker='x',label="NGEOs", color='b', alpha=0.2)
    ax.scatter(closest_sat.x/1e6, closest_sat.y/1e6, closest_sat.z/1e6, s=area, color='r', alpha=1)
#    print(closest_sat.x/1e6, closest_sat.y/1e6, closest_sat.z/1e6)
    if(i==0):
        ax.scatter(base_x, base_y, base_z, label="Base")
        ax.legend()
    #ax.set_aspect('equal', 'box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.axis('off')
    return Positions_NGEO

def get_max_slant_range():
	max_slant_range_ngeo = np.zeros((P,P))
	for m in range(P):
		max_slant_range_ngeo[m,m] = np.sqrt(Orbital_planes[m].h*(Orbital_planes[m].h+2*Re))
		for n in range(m):
			max_slant_range_ngeo[m,n] = np.sqrt(Orbital_planes[m].h*(Orbital_planes[m].h+2*Re))+np.sqrt(Orbital_planes[n].h*(Orbital_planes[n].h+2*Re))
	max_slant_range_ngeo+=np.transpose(max_slant_range_ngeo)
	return max_slant_range_ngeo

def get_SNR_NGEOs(pathLoss,Ptx_db):
	SNR = Ptx_db + ngeoISL.G- pathLoss - ngeoISL.No
	return SNR


def get_SNR_NGEO2UE(pathLoss,Ptx_db):
	SNR = Ptx_db + ngeo2ue.G- pathLoss - ngeo2ue.No
	return SNR


def get_SNR_UE2NGEO(pathLoss,Ptx_db):
	SNR = Ptx_db + ue2ngeo.G- pathLoss - ue2ngeo.No
	return SNR

def get_SNR_ground_link(slant_range, max_slant_range, infty_proxy):
	if slant_range<=max_slant_range:
		pathLoss_ngeo2ue = 10*math.log10((4 * math.pi * slant_range * ngeo2ue.f /c)**2)
		pathLoss_ue2ngeo = 10*math.log10((4 * math.pi * slant_range * ue2ngeo.f /c)**2)
		SNR_ngeo2ue = ngeo2ue.Ptx_db + ngeo2ue.G- pathLoss_ngeo2ue - ngeo2ue.No
		SNR_ue2ngeo = ue2ngeo.Ptx_db + ue2ngeo.G- pathLoss_ue2ngeo - ue2ngeo.No
	else:
		pathLoss_ngeo2ue = infty_proxy
		pathLoss_ue2ngeo = infty_proxy
		SNR_ngeo2ue = -infty_proxy
		SNR_ue2ngeo = -infty_proxy
	return pathLoss_ngeo2ue, pathLoss_ue2ngeo, SNR_ngeo2ue, SNR_ue2ngeo

def get_path_loss(maxSlantRange_ngeo, infty_proxy):	# To get the path loss between satellites
	slant_range_ngeo = np.zeros((N,N), dtype=np.float64)
	pathLoss_ngeo = np.zeros((N,N), dtype=np.float64)
	LoS_ngeo = np.zeros((N+N_geos,N+N_geos),dtype=int)

	for i in range(N):

		################## NGEO to NGEO
		for j in range(i+1,N):
			slant_range_ngeo[i,j] = math.sqrt((NGEO[i].x-NGEO[j].x)**2 +(NGEO[i].y-NGEO[j].y)**2+(NGEO[i].z-NGEO[j].z)**2)
			if slant_range_ngeo[i,j]>maxSlantRange_ngeo[NGEO[i].in_plane,NGEO[j].in_plane]:
				pathLoss_ngeo[i,j] = infty_proxy
			else:
				pathLoss_ngeo[i,j] = 10*math.log10((4 * math.pi * slant_range_ngeo[i,j] * ngeoISL.f / c)**2)
				LoS_ngeo[i,j] +=1
				LoS_ngeo[j,i] +=1
		################## NGEO to GEO

	slant_range_ngeo += np.transpose(slant_range_ngeo)			# Distance between each satellite pair
	pathLoss_ngeo += np.transpose(pathLoss_ngeo)			# Distance between each satellite pair
	return pathLoss_ngeo, slant_range_ngeo, LoS_ngeo
def sat_visible_from(base_lat, base_long): #base_lat and base_long to be in degrees
    base_lat=np.radians(base_lat)
    base_long=np.radians(base_long)
    base_z=Re*np.sin(base_lat)
    base_y=-Re*np.cos(base_lat)*np.cos(base_long)
    base_x=-Re*np.cos(base_lat)*np.sin(base_long)
    visible_satellites=[]
    closest_sat=0
    min_distance=Re
    for sat in NGEO:
        distance=np.sqrt((sat.x-base_x)**2+(sat.y-base_y)**2+(sat.z-base_z)**2)
        max_distance_2_ground=Orbital_planes[sat.in_plane].max_distance_2_ground
        if(distance<=max_distance_2_ground):
            visible_satellites.append(sat)
            if(distance<=min_distance):
                min_distance=min(distance, min_distance)
            if visible_satellites:
                closest_sat=visible_satellites[np.argmin([np.sqrt((sat.x-base_x)**2+(sat.y-base_y)**2+(sat.z-base_z)**2) for sat in visible_satellites])]
                closest_sat_distance=np.sqrt((closest_sat.x-base_x)**2+(closest_sat.y-base_y)**2+(closest_sat.z-base_z)**2)
    return visible_satellites, closest_sat, closest_sat_distance
def pkt_forward(sat_from, sat_to, t_slot):#id of the two satellites as input
    if(LoS_ngeo[sat_from, sat_to]):
        pkt=NGEO[sat_from].queue.pop(0)
        delay=slantRange_ngeo[sat_from, sat_to]/c
        pkt.sat_r_time+=delay/pkt_duration
        NGEO[sat_to].queue.append(pkt)
    elif(sat_from<sat_to):
        pass
        #print("Not visible for forwarding", end='\r')

#########################################################################################################################
######				Beginning of main()
#########################################################################################################################

ngeoISL, ngeo2ue, ue2ngeo = initialize_comm_params()
Orbital_planes = []
NGEO = []
op_of_the_satellites = np.zeros(N)
ID = 0									# ID for the satellites
displacement = np.zeros(P)	# Rotation of the orbital planes at the beginning of the simulation, can be set to specific values
if random_rotation:		# Rotate the orbital planes randomly at the beginning of the simulation
	displacement = np.random.random(P)


min_elev=30
if(specific_constellation=="IridiumNEXT"):
	min_elev=8.2

##########	Create the NGEO constellation
distribution_angle = 2*math.pi
if Walker_star:
	distribution_angle /= 2
for p in range(P):
	longitude = distribution_angle*p/P
	Orbital_planes.append(orbital_plane(h[p], longitude, math.radians(inclination), N_p, min_elev)) # The min. elevation angle for NGEOs (to define the coverage of GSs )is 30 degrees
	for n in range(N_p):
		polar_angle = 2 *math.pi *(displacement[p]+n/N_p)
		NGEO.append(satellite(p,n,h[p],polar_angle))
		NGEO[ID].rotate(0)				# To calculate cartesian coordinates
		op_of_the_satellites[ID] = p
		#print(NGEO[ID])
		#print(ID, NGEO[ID].i_in_plane+NGEO[ID].in_plane*N_p)
		ID +=1
print(f'NGEO orbital plane params.:{Orbital_planes[p]}\n')

maxSlantRange_NGEO = get_max_slant_range()

pathLoss_ngeo2ue, pathLoss_ue2ngeo, SNR_ngeo2ue, SNR_ue2ngeo = get_SNR_ground_link(Orbital_planes[-1].max_distance_2_ground, Orbital_planes[-1].max_distance_2_ground, 1e-4)

print(f'Max_slant range [m]\n NGEO = {maxSlantRange_NGEO}\n\n Max_slant range NGEO to UE [m] = {Orbital_planes[0].max_distance_2_ground}\n Max pathloss NGEO to UE (within coverage): DL {pathLoss_ngeo2ue} dB and UL {pathLoss_ue2ngeo} dB\n Min SNR to ground: DL = {SNR_ngeo2ue} dB and UL = {SNR_ue2ngeo} dB')

pathLoss_ngeo, slantRange_ngeo, LoS_ngeo = get_path_loss(maxSlantRange_NGEO, 1e4)

# The matrices above contain the pathloss (pathLoss_X), the distances (slantRange_X), and line or no line of sight (LoS_X, 1 if True)

SNR_ngeo = np.zeros((N,N))

for i in range(N):
	for j in range(i+1,N):
		SNR_ngeo[i,j] = get_SNR_NGEOs(pathLoss_ngeo[i,j],ngeoISL.Ptx_db)
		SNR_ngeo[j,i] = SNR_ngeo[i,j]
np.savetxt(f'{specific_constellation}_snr_NGEO.csv',SNR_ngeo, delimiter=",")		# Matrix with the SNR between NGEOs
np.savetxt(f'{specific_constellation}_slant_range_NGEO.csv',slantRange_ngeo, delimiter=",")	# Matrix with the distances between NGEOs

'''
Slotted ALOHA to be implemented here. Returns a packet to be stored in the Base Station's queue. Implements
collision detection for each slot t .
Each packet is directed towards a particular satellite. If the satellite goes out of view, the packets are to be forwarded by the visible satellite.
'''

TestNetwork=Network(rate, nodes, num_packets, t_start, t_end, pkt_duration)
#print([node.arrival_times for node in TestNetwork.nodes])
blind_time=0
f = open("data/packets_sent_to_sat.csv", "w")
columns=["i", "timestamp", "t_time", "collisions", "base_t_time", "satellite", "sat_r_time"]
writer=csv.writer(f)
writer.writerow(columns)
print("{} constellation with base at ({}, {})".format(specific_constellation, base_lat, base_long))
print("Run started")
base_queue=[]
sat_number=-1
for t in range(t_slot_end):
    visible_sats, closest_sat, closest_sat_distance=sat_visible_from(base_lat, base_long)
    TestNetwork.s_aloha(t)
    if TestNetwork.transmitted_packets:
        pkt=TestNetwork.transmitted_packets[-1]
        if(pkt.t_time==t):
            pkt.base_t_time=t+1
            base_queue.append(pkt)
        if visible_sats:
            if base_queue:
                if(base_queue[0].base_t_time==t):
                    b_to_s_pkt = base_queue.pop(0)
                    closest_sat_id=closest_sat.i_in_plane+closest_sat.in_plane*N_p
                    b_to_s_pkt.sat_id=closest_sat_id
                    delay=closest_sat_distance/c
                    b_to_s_pkt.sat_r_time=t+delay/pkt_duration
                    closest_sat.queue.append(b_to_s_pkt)
                    if(sat_number==-1):
                        sat_number=closest_sat_id
                    pkt_forward(closest_sat_id, sat_number, t)

        else:
            blind_time+=1
    if(visible_sats):
        if(t%slots_per_sec==0):
            print("{} out of {} seconds, sat={}, plane={}, sat_queue={}, base_queue={}".format(t/slots_per_sec, t_end, closest_sat.i_in_plane, closest_sat.in_plane, len(closest_sat.queue), len(base_queue)), end="\r")
    else:
        if(t%slots_per_sec==0):
            print("{} out of {} seconds, sat={}".format(t/slots_per_sec, t_end, closest_sat), end="\r")
    if(t%(slots_per_sec*20)==0):
        if(visible_sats):
        	Positions_NGEO = plot_constellation(closest_sat,  2, t)
    rotate_by = pkt_duration
    for n in range(N):
        NGEO[n].rotate(rotate_by)
print("Run finished, blind time fraction is {}".format(float(blind_time)/t_slot_end))
for pkt in NGEO[sat_number].queue:
    writer.writerow([pkt.i, pkt.timestamp, pkt.t_time, pkt.collisions, pkt.base_t_time, sat_number, pkt.sat_r_time])
for n in range(N):
    if(NGEO[n].queue):
        print(NGEO[n])
plt.show()
'''
    for node in TestNetwork.nodes:
        if(node.transmitted_packets):
            if(node.transmitted_packets[-1].t_time==t):
                if(node.transmitted_packets[-1].t_time-np.ceil(node.transmitted_packets[-1].timestamp) <= aoi[node.i][t-1]):
                    aoi[node.i][t]=node.transmitted_packets[-1].t_time-np.ceil(node.transmitted_packets[-1].timestamp)
                else:
                    aoi[node.i][t]=aoi[node.i][t-1]+1
            else:
                aoi[node.i][t]=aoi[node.i][t-1]+1
        else:
            aoi[node.i][t]=aoi[node.i][t-1]+1
plt.figure(0)
plt.grid()
plt.xlabel("Time (pk duration)")
plt.ylabel("AoI (pk duration)")
plt.title("Age of Information variation with time for rate {} pk/pk duration".format(rate))
for node in TestNetwork.nodes:
    plt.plot(aoi[node.i], label="Node {}".format(node.i))
plt.legend()
plt.savefig("plots/AoI_individual_highrate.png")
plt.show()
    #print("t_packets {}, {} , time {}, {} ".format(len(TestNetwork.transmitted_packets),[node.i for node in TestNetwork.occupied_buffers] ,t, [node.i for node in TestNetwork.actives]))
df=pd.read_csv("data/packets.csv")
'''
plt.show()
