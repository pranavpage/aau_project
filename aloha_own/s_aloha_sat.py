import numpy as np
import scipy.special
import math
import matplotlib.pyplot as plt
import time
import random
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linear_sum_assignment

random.seed(2)
np.random.seed(2)
np.set_printoptions(precision=2)

Re = 6378e3	# Radius of the earth [m]
G = 6.67259e-11	# Universal gravitational constant [m^3/kg s^2]
Me = 5.9736e24	# Mass of the earth
k = 1.38e-23	# Boltzmann's constant
c = 299792458	# Speed of light [m/s]
eff = 0.6 	# Efficiency of the parabolic antenna


#######		GEO satellites
N_geos = 3
h_geo = 35786e3
inclination_geo = 0

random_rotation = False	# Set to True to rotate the orbital planes randomly at the beginning of the simulation

specific_constellation = "Kepler"	# Select the constellation design among the options provided below


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

	def  __repr__(self):
		return '\n orbital plane= {}, index in plane= {}, pos x= {}, pos y= {}, pos z= {}\n polar angle = {} latitude = {}'.format(
	self.in_plane,
	self.i_in_plane,
	'%.2f'%self.x,
	'%.2f'%self.y,
	'%.2f'%self.z,
	'%.2f'%self.polar_angle,
	'%.2f'%self.latitude)

	def rotate(self, delta_t):			# To rotate the satellites after a period delta_t using the polar angle
		self.polar_angle = (self.polar_angle+2 *math.pi * delta_t / Orbital_planes[self.in_plane].period)%(2*math.pi)
		self.polar_angle = self.polar_angle % (2 *math.pi)
		delta = math.pi/2-Orbital_planes[self.in_plane].inclination
		theta = Orbital_planes[self.in_plane].longitude
		phi = self.polar_angle
		self.x = (Orbital_planes[self.in_plane].h+Re) * (math.sin(phi)*math.cos(theta)+math.cos(phi)*math.sin(delta)*math.sin(theta))
		self.y = (Orbital_planes[self.in_plane].h+Re) * (math.sin(phi)*math.sin(theta)-math.cos(phi)*math.sin(delta)*math.cos(theta))
		self.z = (Orbital_planes[self.in_plane].h+Re) * math.cos(phi)*math.cos(delta)
class Node:
    def __init__(self, l, num_packets, i):
        self.i=i
        self.l=l
        self.num_packets=num_packets
        self.inter_arrival_times=np.random.exponential(1/l, (num_packets))
        self.arrival_times=np.cumsum(self.inter_arrival_times)
        self.max_time=int(np.ceil(np.amax(self.arrival_times))+1) #transmitting till max_time
        self.queue=[]
        self.collisions=0
        self.successes=0
        self.transmit_flag=0

    def receive(self, t):
        #arrivals in (t-1, t) are appended to the queue
        for timestamp in self.arrival_times:
            if(t==np.ceil(timestamp)):
                self.queue.append(Packet(timestamp, self.i))
    def transmit(self, t):
        #if no collisions, and the queue is non-empty, transmit immediately
        if(self.collisions==0):
            if(len(self.queue)>0):
                transmitted_packet=self.queue[0]
                transmitted_packet.t_time=t
                self.transmit_flag=1
                return transmitted_packet
            else:
                self.transmit_flag=0
                return 0
        else:
            #collided packet
            if(len(self.queue)>0):
                if(self.queue[0].t_time==t):
                    transmitted_packet=self.queue[0]
                    transmitted_packet.t_time=t
                    self.transmit_flag=1
                    return transmitted_packet
                else:
                    self.transmit_flag=0
                    return 0
    def generate(self, t):
        if(t==int(self.max_time-10)):
            self.inter_arrival_times=np.append(self.inter_arrival_times, (np.random.exponential(1/self.l, (15))))
            self.arrival_times=np.cumsum(self.inter_arrival_times)
            self.max_time=int(np.ceil(np.amax(self.arrival_times))+1)


class Packet:
    def __init__(self, timestamp, i):
        self.timestamp=timestamp
        self.t_time=0
        self.i=i
        self.collisions=0
        self.base_t_time=0
        self.sat_time=0
########################################################################################################################

def end_sim(mynodes, t, t_end):
    temp=1
    '''    for node in mynodes:

        if(t<=node.max_time):
            temp=0
        elif(node.queue and t>node.max_time):
            temp=0
        '''

    '''
    if(not node.queue and t>node.max_time):
        temp=1
    else:
        temp=0
    '''
    if(t<=t_end):
        temp=0
    else:
        temp=1


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

def plot_constellation(meta_NGEO, fig_tag):
	Positions_NGEO = np.zeros((N,3))
	for n in range(N):
		Positions_NGEO[n,:] = [NGEO[n].x/1e6, NGEO[n].y/1e6, NGEO[n].z/1e6]
	fig = plt.figure(fig_tag)
	ax = fig.gca(projection='3d')
	#ax.set_box_aspect((np.ptp(Positions_NGEO[:,0]), np.ptp(Positions_NGEO[:,1]), np.ptp(Positions_NGEO[:,2])))
	area = math.pi * (5**2)
	base_lat=np.radians(0)
	base_long=np.radians(90)
	base_z=Re*np.sin(base_lat)/1e6
	base_y=-Re*np.cos(base_lat)*np.cos(base_long)/1e6
	base_x=-Re*np.cos(base_lat)*np.sin(base_long)/1e6
	ax.scatter(Positions_NGEO[:,0],Positions_NGEO[:,1], Positions_NGEO[:,2], c=meta_NGEO, s=area,label="NGEOs")
	ax.scatter(base_x, base_y, base_z, label="Base")
	#ax.set_aspect('equal', 'box')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.legend()
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
def sat_visible_from(base_lat, base_long):
	base_lat=np.radians(base_lat)
	base_long=np.radians(base_long)
	base_z=Re*np.sin(base_lat)
	base_y=-Re*np.cos(base_lat)*np.cos(base_long)
	base_x=-Re*np.cos(base_lat)*np.sin(base_long)
	visible_satellites=[]
	min_distance=Re
	for sat in NGEO:
		distance=np.sqrt((sat.x-base_x)**2+(sat.y-base_y)**2+(sat.z-base_z)**2)
		max_distance_2_ground=Orbital_planes[sat.in_plane].max_distance_2_ground
		if(distance<=max_distance_2_ground):
			visible_satellites.append(sat)
			if(distance<=min_distance):
				closest_sat=sat
				min_distance=min(distance, min_distance)
		else:
			closest_sat=0
	return visible_satellites, closest_sat
def s_aloha_with_satellite(G, nodes, packets=20, cw_min=32, cw_max=256, t_end=500, max_attempts=8):
    r_pkt=[]
    mynodes=[Node(G/nodes, packets, i) for i in range(nodes)]
#        max_time=np.amax([mynodes[i].max_time for i in range(nodes)])
    t=0
    start_flag=0
    t_start=100
    age=np.zeros(t_end+1)
    peaks=[]
#    print("Time, transmission time, Node, Collisions, Timestamp")
    while(t<=t_end):
        actives=[]
        t_packets=[]
        for node in mynodes:
            node.generate(t)
            node.receive(t)
            pkt=node.transmit(t)
            if(pkt!=0):
                #print(pkt.timestamp, pkt.i)
                actives.append(node)
                t_packets.append(pkt)
            if(pkt==0):
                pass
                #print(t)
        #print([active.i for active in actives])
        if(len(actives)>1):
            #collision
            for active in actives:
                active.collisions+=1
                active.queue[0].collisions+=1
                if(1<=active.queue[0].collisions<=max_attempts):
                    backoff=np.random.randint(1, int(2**(active.collisions)))
                    #backoff=np.random.randint(0, int(2**(active.collisions)*cw_min)-1)
                    active.queue[0].t_time=t+backoff
                else:
                    dropped_pkt=active.queue.pop(0)
                    active.collisions=0
            if(t>=t_start):
                age[t]=age[t-1]+1
            else:
                age[t]=age[t-1]
                #print(t, t+backoff, active.i, active.collisions, np.around(active.queue[0].timestamp, 2))
        elif(len(actives)==1):
            actives[0].collisions=0
            actives[0].successes+=1
            s_pkt=actives[0].queue.pop(0)
            s_pkt.t_time=t
            r_pkt.append(s_pkt)
            if(t>=t_start):
                if((t-s_pkt.timestamp)<=age[t-1]):
                    age[t]=(t-s_pkt.timestamp)
                    peaks.append(age[t-1]-age[t])
                else:
                    age[t]=age[t-1]+1
            else:
                age[t]=age[t-1]
        else:
            if(t>=t_start):
                age[t]=age[t-1]+1
            else:
                age[t]=age[t-1]
        t+=1
    max_time=t
    delay=0
    tput=0
    t_pkt=[]
    for pkt in r_pkt:
        if(pkt.t_time>=t_start):
            delay+=pkt.t_time-pkt.timestamp
            tput+=1
            t_pkt.append(pkt)
    delay/=len(t_pkt)
    #print(len(t_pkt), G, nodes)
    tput/=float(max_time-t_start)
    peaks=np.array(peaks)
    return t_pkt, tput, delay, age, peaks

def transmit_to_base(t, mynodes, max_attempts, t_end):
	r_pkt=[]
	if(t<=t_end):
		actives=[]
		t_packets=[]
		for node in mynodes:
			node.generate(t)
			node.receive(t)
			pkt=node.transmit(t)
			if(pkt!=0):
				#print(pkt.timestamp, pkt.i)
				actives.append(node)
				t_packets.append(pkt)
			if(pkt==0):
				pass
				#print(t)
		#print([active.i for active in actives])
		if(len(actives)>1):
			#collision
			for active in actives:
				active.collisions+=1
				active.queue[0].collisions+=1
				if(1<=active.queue[0].collisions<=max_attempts):
					backoff=np.random.randint(1, int(2**(active.collisions)))
					#backoff=np.random.randint(0, int(2**(active.collisions)*cw_min)-1)
					active.queue[0].t_time=t+backoff
				else:
					dropped_pkt=active.queue.pop(0)
					active.collisions=0
		elif(len(actives)==1):
			actives[0].collisions=0
			actives[0].successes+=1
			s_pkt=actives[0].queue.pop(0)
			s_pkt.t_time=t
			r_pkt.append(s_pkt)
	return(r_pkt)
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



##########	Create the NGEO constellation
distribution_angle = 2*math.pi
if Walker_star:
	distribution_angle /= 2
for p in range(P):
	longitude = distribution_angle*p/P
	Orbital_planes.append(orbital_plane(h[p], longitude, math.radians(inclination), N_p, 30)) # The min. elevation angle for NGEOs (to define the coverage of GSs )is 30 degrees
	for n in range(N_p):
		polar_angle = 2 *math.pi *(displacement[p]+n/N_p)
		NGEO.append(satellite(p,n,h[p],polar_angle))
		NGEO[ID].rotate(0)				# To calculate cartesian coordinates
		op_of_the_satellites[ID] = p
		#print(NGEO[ID])
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
plt.figure()
pl=plt.imshow(pathLoss_ngeo)
plt.colorbar(pl)
plt.title('Path loss in dB')

plt.figure()
pl=plt.imshow(LoS_ngeo)
plt.colorbar(pl)
plt.title('Indicator of line of sight')


plt.figure()
pl=plt.imshow(SNR_ngeo, vmin=-100)
plt.colorbar(pl)
plt.title('Signal-to-noise ratio (SNR) in dB')

'''
t=0
blind_time=0
base_queue=[]
nodes=10
G=10
packets=20
t_start=10
t_end=300
base_lat=45
base_long=90
base_lat=np.radians(base_lat)
base_long=np.radians(base_long)
base_z=Re*np.sin(base_lat)
base_y=-Re*np.cos(base_lat)*np.cos(base_long)
base_x=-Re*np.cos(base_lat)*np.sin(base_long)
mynodes=[Node(G/nodes, packets, i) for i in range(nodes)]
max_attempts=8
for t in range(t_end+1):
	visible_sats, closest_sat=sat_visible_from(np.degrees(base_lat), np.degrees(base_long))
	r_pkt=transmit_to_base(t, mynodes, max_attempts, t_end)
	base_queue.append(r_pkt)
	if visible_sats:
		closest_sat=visible_sats[np.argmin([np.sqrt((sat.x-base_x)**2+(sat.y-base_y)**2+(sat.z-base_z)**2) for sat in visible_sats])]
		print(closest_sat.i_in_plane, closest_sat.in_plane, t, len(base_queue))
	else:
		blind_time+=1
	rotate_by = 1			# time to rotate the constellation in seconds
	for n in range(N):
		NGEO[n].rotate(rotate_by)
print("blind_time_fraction is {}, length of base_queue {}".format(float(blind_time)/(t_end), len(base_queue)))
		#print("{}, {}, {}, {}, {}".format('%.2f'%np.rad2deg(NGEO[n].polar_angle), '%.3f'%NGEO[n].x, '%.3f'%NGEO[n].y, '%.3f'%NGEO[n].z, '%.1f'%np.rad2deg(Orbital_planes[NGEO[n].in_plane].longitude)))
	#Positions_NGEO = plot_constellation(op_of_the_satellites, 2)

plt.show()
