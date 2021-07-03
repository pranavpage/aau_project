
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import csv
from s_aloha_improved import Node, Packet, Network
from scipy.optimize import curve_fit
nodes=4
num_packets=20
t_start=100
t_end=1000
pkt_duration=1e-3
cw_min=16
cw_max=1024
max_attempts=int(np.log2(cw_max/cw_min))
occupancy=np.zeros((nodes, int(t_end)))
#print([node.arrival_times for node in TestNetwork.nodes])
rate_arr=np.linspace(0.01, 0.5, 100)
average_queue_size=np.zeros((nodes, len(rate_arr)))
average_tput=np.zeros_like(rate_arr)
runs=3
def objective(x, a, b, c, d, e):
    return a*np.exp(-(x-b))+c*np.exp(x-d)+e

for i in range(len(rate_arr)):
    tput_temp=0
    queue_temp=np.zeros((nodes))
    for run in range(runs):
        TestNetwork=Network(rate_arr[i], nodes, num_packets, t_start, t_end, pkt_duration)
        for t in range(t_end):
            TestNetwork.s_aloha(t)
            for node in TestNetwork.nodes:
                if(node.queue):
                    occupancy[node.i][t]=len(node.queue)
                else:
                    occupancy[node.i][t]=0
        tput_temp+=len(TestNetwork.transmitted_packets)/float(t_end)
        queue_temp+=np.mean(occupancy, 1)
    average_tput[i]=tput_temp/float(runs)
    average_queue_size[:,i]=queue_temp/float(runs)
    print("{}/{} done".format(i+1, len(rate_arr)), end='\r')
plt.figure(0)
plt.grid()
plt.title("Average queue size variation with offered load \n cw_min={}, cw_max={}, nodes={}".format(cw_min, cw_max, nodes))
for i in range(nodes):
    plt.plot(rate_arr, average_queue_size[i], label="Node {}".format(i))
#plt.axvline(1/np.exp(1), linestyle="--")
plt.legend()
plt.savefig("plots/queue_size_variation_detailed.png")
plt.show()
plt.figure(1)
plt.grid()
plt.title("Average tput variation with offered load \n cw_min={}, cw_max={}, nodes={}".format(cw_min, cw_max, nodes))
#popt, _=curve_fit(objective, rate_arr, average_tput)
plt.plot(rate_arr, average_tput, label="System throughput")
plt.plot(rate_arr, rate_arr, label="Input load")
#plt.plot(rate_arr, objective(rate_arr, *popt), label="Fit")
plt.legend()
plt.savefig("plots/average_tput_variation_detailed_3.png")
plt.show()
