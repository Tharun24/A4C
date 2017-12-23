import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
FMT_date = '%Y-%m-%d'
FMT_time = '%H:%M:%S'
def moving_average(arr,length):
    return np.array([np.mean(arr[i-length:i]) for i in range(length,len(arr))])

game_name = 'pong' # pong/qbert/spaceinvaders/beamrider

def diff(start_time,end_time,date_diff):
    if end_time>start_time:
        time_diff =  datetime.strptime(end_time, FMT_time) - datetime.strptime(start_time, FMT_time)
        return 24.0*date_diff.days + (time_diff.seconds*24.0)/86400
    else:
        time_diff = datetime.strptime(start_time, FMT_time) - datetime.strptime(end_time, FMT_time)
        return 24.0*date_diff.days - (time_diff.seconds*24.0)/86400


#############################################   GA3C   ##############################################################
data = np.loadtxt('GA3C/{}_ga3c.txt'.format(game_name), dtype=str,delimiter='\n')
all_dates = np.array([itm.split()[0] for itm in data])
all_date_diff = np.array([datetime.strptime(itm, FMT_date) - datetime.strptime(all_dates[0], FMT_date) for itm in all_dates])
all_time = np.array([itm.split()[1][:-1] for itm in data])
all_diff = np.array([diff(all_time[0],all_time[i],all_date_diff[i]) for i in range(len(data))])
reward1 = np.array([int(itm.split()[2][:-1]) for itm in data])
r_mean1 = moving_average(reward1,1000)
idx1 = all_diff[1000:]

#############################################   Switching   ##############################################################
data = np.loadtxt('Switching/{}_a4c.txt'.format(game_name), dtype=str,delimiter='\n')
all_dates = np.array([itm.split()[0] for itm in data])
all_date_diff = np.array([datetime.strptime(itm, FMT_date) - datetime.strptime(all_dates[0], FMT_date) for itm in all_dates])
all_time = np.array([itm.split()[1][:-1] for itm in data])
all_diff = np.array([diff(all_time[0],all_time[i],all_date_diff[i]) for i in range(len(data))])
reward2 = np.array([int(itm.split()[2][:-1]) for itm in data])
r_mean2 = moving_average(reward2,1000)
idx2 = all_diff[1000:]

############################################ Actual Plotting Begins #################################################
plotby = 'time'

if plotby=='episodes':
    plt1, = plt.plot(r_mean1,'r',label='GA3C')
    plt2, = plt.plot(r_mean2,'g',label='A4C')
    lgd = plt.legend(handles=[plt1,plt2],loc=2)
    plt.title(game_name)
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.savefig('{}_'.format(game_name)+plotby+'.png')
    plt.clf()
elif plotby=='time':
    plt1, = plt.plot(idx1, r_mean1,'r',label='GA3C')
    plt2, = plt.plot(idx2, r_mean2,'g',label='A4C')
    lgd = plt.legend(handles=[plt1,plt2],loc=2)
    plt.title(game_name)
    plt.xlabel('Training time(hours)')
    plt.ylabel('Score')
    plt.savefig('{}_'.format(game_name)+plotby+'.png')
    plt.clf()

