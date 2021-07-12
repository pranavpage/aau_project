import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import csv
'''
Every user is defined as a Node, with receive(), transmit(), and generate() methods.
receive() : appends Packets to the queue at slot t if arrival in (t-1, t]
transmit() : if the queue is non-empty, and the pkt at the top of queue has its t_time equal to t, the pkt is returned.
            Else, 0 is returned
(don't copy old transmit() )
generate() : generates more Packets by adding more inter_arrival_times
t is the slot number. t * slot_duration is the actual time
'''
class Packet:
    def __init__(self, timestamp, i):
        self.timestamp=timestamp
        self.t_time=0
        self.i=i
        self.collisions=0
    def __repr__(self):
        return "({}:{},{}, c{})".format(self.i, np.around(self.timestamp, 2), self.t_time, self.collisions)

class Node:
    def __init__(self, l, num_packets, i, cw_min, cw_max):
        self.i=i
        self.l=l
        self.num_packets=num_packets
        self.inter_arrival_times=np.random.exponential(1/l, (num_packets))
        self.arrival_times=np.cumsum(self.inter_arrival_times)
        self.max_time=int(np.ceil(np.amax(self.arrival_times))+1)
        self.queue=[]
        self.successes=0
        self.busy=0
        self.transmitted_packets=[]
        self.cw_min=cw_min
        self.cw_max=cw_max
    def __repr__(self):
        if(self.queue):
            return "| n{}, q:{}, c:{}, t:{} |".format(self.i, len(self.queue), self.queue[0].collisions, self.queue[0].t_time)
        else:
            return "n:{}, q:0".format(node.i)

    def receive(self, t):
        #arrivals in (t-1, t) are appended to the queue
        for timestamp in self.arrival_times:
            if(t==np.ceil(timestamp)):
                self.queue.append(Packet(timestamp, self.i))
        if(self.queue):
            self.busy+=1

    def transmit(self, t):
        if(self.queue):
            t_pkt=self.queue[0]
            if(t_pkt.t_time==0):
                backoff=np.random.randint(0, int(self.cw_min))
                t_pkt.t_time=t+backoff
            if(t_pkt.t_time==t):
                return t_pkt
            else:
                return 0
        else:
            return 0

    def generate(self, t):
        if(t==int(self.max_time-1)):
            self.inter_arrival_times=np.append(self.inter_arrival_times, (np.random.exponential(1/self.l, (15))))
            self.arrival_times=np.cumsum(self.inter_arrival_times)
            self.max_time=int(np.ceil(np.amax(self.arrival_times)+1))
class Network:
    def __init__(self, rate, nodes, num_packets, t_start, t_end, pkt_duration, cw_min=16, cw_max=1024):
        self.rate=rate
        self.nodes=[Node(float(self.rate)/nodes, num_packets, i, cw_min, cw_max) for i in range(nodes)]
        self.individual_rate=float(rate)/nodes
        self.t_start=t_start
        self.t_end=t_end
        self.max_attempts=np.log2(cw_max/cw_min)
        self.transmitted_packets=[]
        self.pkt_duration=pkt_duration
        self.actives=[]
        self.occupied_buffers=[]
        self.cw_min=cw_min
        self.cw_max=cw_max
    def s_aloha(self, t):
        actives=[]
        occupied_buffers=[]
        for node in self.nodes:
            node.generate(t)
            node.receive(t)
            pkt=node.transmit(t)
            if(node.queue):
                occupied_buffers.append(node)
            if(pkt):
                actives.append(node)
        self.actives=actives
        self.occupied_buffers=occupied_buffers
        if(actives):
            if(len(actives)>1):
                #print([active.queue[0].i for active in actives], t)
                for active in actives:
                    active.queue[0].collisions+=1
                    if(1<=active.queue[0].collisions<=self.max_attempts):
                        backoff=np.random.randint(1, int(self.cw_min*(2**(active.queue[0].collisions))))
                        active.queue[0].t_time=t+backoff
                    else:
                        #drop packet
                        dropped_pkt=active.queue.pop(0)
            elif(len(actives)==1):
                actives[0].successes+=1
                transmitted_packet=actives[0].queue.pop(0)
                transmitted_packet.t_time=t
                self.transmitted_packets.append(transmitted_packet)
                actives[0].transmitted_packets.append(transmitted_packet)


'''
Slotted ALOHA to be implemented here. Returns a packet to be stored in the Base Station's queue. Implements
collision detection for each slot t .


class Base:
    def __init__(self, base_lat_deg, base_long_deg):
        self.queue=[]
        self.base_lat_deg=base_lat_deg
        self.base_long_deg=base_long_deg
'''
def main():
    rate=0.2
    nodes=10
    num_packets=20
    t_start=100
    t_end=5000
    pkt_duration=1e-3
    cw_min=16
    cw_max=1024
    max_attempts=int(np.log2(cw_max/cw_min))
    #np.random.seed(0)
    aoi=np.zeros((nodes, int(t_end)))
    occupancy=np.zeros((nodes, int(t_end)))
    TestNetwork=Network(rate, nodes, num_packets, t_start, t_end, pkt_duration)
    #print([node.arrival_times for node in TestNetwork.nodes])
    for t in range(t_end):
        TestNetwork.s_aloha(t)
        occupied_buffers=TestNetwork.occupied_buffers
        actives=TestNetwork.actives
    #    print("At {},transmitted={}, actives={}, occupied={}, ".format(t,len(TestNetwork.transmitted_packets),actives, occupied_buffers))
        for node in TestNetwork.nodes:
            if(node.queue):
                occupancy[node.i][t]=len(node.queue)
            else:
                occupancy[node.i][t]=0
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
    plt.title("AoI variation with time for rate {} pk/pk duration, max_attempts {}".format(rate, max_attempts))
    for node in TestNetwork.nodes:
        plt.plot(aoi[node.i], label="Node {}".format(node.i))
    plt.legend()
    plt.savefig("plots/AoI_individual_CW_maxattempts_{}_{}s.png".format(max_attempts, t_end))
    plt.show()
    plt.figure(1)
    plt.grid()
    plt.xlabel("Time (pk duration)")
    plt.ylabel("Instantaneous queue size")
    plt.title("Instantaneous queue size for rate {}, max_attempts {}".format(rate, max_attempts))
    for node in TestNetwork.nodes:
        plt.plot(occupancy[node.i], label="Node {}, average_size {}".format(node.i, np.mean(occupancy[node.i])))
    plt.legend()
    plt.savefig("plots/Inst_queue_size_maxattempts_{}_{}s".format(max_attempts, t_end))
    plt.show()

        #print("t_packets {}, {} , time {}, {} ".format(len(TestNetwork.transmitted_packets),[node.i for node in TestNetwork.occupied_buffers] ,t, [node.i for node in TestNetwork.actives]))
    f = open("data/packets_CW.csv", "w")
    columns=["i", "timestamp", "t_time", "collisions"]
    writer=csv.writer(f)
    writer.writerow(columns)
    for pkt in TestNetwork.transmitted_packets:
        writer.writerow([pkt.i, pkt.timestamp, pkt.t_time, pkt.collisions])
    #df=pd.read_csv("data/packets_CW.csv")
if __name__=="__main__":
    main()
