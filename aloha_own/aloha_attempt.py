import numpy as np
import matplotlib.pyplot as plt
input_G=np.arange(0.1, 5, 0.1)
nodes=6
packets=200
avr_throughput=np.zeros(len(input_G))
for epoch in range(10):
    throughput=[]
    for G in input_G:
        l=[G/nodes]*nodes
        interarrival_times= np.zeros((nodes, packets))
        timestamps=np.zeros((nodes,packets))
        for node in range(nodes):
            interarrival_times[node] = np.around(np.random.exponential(1/l[node], (1,packets)), 0)
    #            if(interarrival_times[node][j] < 1):
    #                interarrival_times[node][j]=1
            timestamps[node]=np.cumsum(interarrival_times[node])
        t_max=np.amin(np.amax(timestamps, axis=1))
        #one packet duration is one second
        #slots are also of one second each
        #Slotted ALOHA
        total_packets=0
        cnt_success=0
        success_array=np.zeros(int(np.ceil(t_max)+2))
        for t in range(int(np.ceil(t_max)+1)):
            cnt_arrive=0
            for node in range(nodes):
                for timestamp in timestamps[node]:
                    if(timestamp>=t-1 and timestamp<t):
                        cnt_arrive+=1
                        total_packets+=1
            if(cnt_arrive==1):
                success_array[t]=1
                cnt_success+=1
        throughput.append(cnt_success*1.0/int(np.ceil(t_max)+1))
    throughput=np.array(throughput)
    avr_throughput+=throughput
avr_throughput/=10
G_max=input_G[np.argmax(avr_throughput)]
throughput_max=np.amax(avr_throughput)
print("The maximum throughput {} is attained at load {} pk/pk duration ".format(throughput_max,G_max))
'''
    for j in range(len(timestamps[node])):
            if(j>0):
                if(np.floor(timestamps[node][j])-np.floor(timestamps[node][j-1])==0):
                    timestamps[node][j]=np.floor(timestamps[node][j-1])+1
'''
