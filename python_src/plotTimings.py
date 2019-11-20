#plotTimings.py


import numpy as np
import matplotlib.pyplot as plt

from resultParser import *

fsmed = 16

# Number of processors is static (does not dynamically change based on degrees of freedom)
# RB ordering
# N = 1025 (held constant)

#filepathStatic1 = 'timings_strongScaling_staticProcs_3runs.txt'
##nruns = 3
#filepathStatic2 = 'timings_strongScaling_staticProcs_7runs.txt'
##nruns2 = 7
#
#filepath = 'timings_strongScaling_dynamicProcs_1e4ratio_3runs.txt'
#nruns = 3
#
#nprocsList, timingList, errorList = parse_file(filepath, nruns)
#
#nproc = np.array(nprocsList) #+ 0.7*np.array(nprocsList2)
#timings = np.array(timingList)# + 0.7*np.array(timingList2)
#error = np.array(errorList)# + 0.7*np.array(errorList2)
#
#
#nrunsST1 = 3
#nrunsST2 = 7
#nprocsListStatic, timingListStatic, errorListStatic = parse_file(filepathStatic1, nrunsST1)
#nprocsListStatic2, timingListStatic2, errorListStatic2 = parse_file(filepathStatic2, nrunsST2)
#nprocStatic = 0.3*np.array(nprocsListStatic) + 0.7*np.array(nprocsListStatic2)
#timingsStatic = 0.3*np.array(timingListStatic) + 0.7*np.array(timingListStatic2)
#errorStatic = 0.3*np.array(errorListStatic) + 0.7*np.array(errorListStatic2)


#plt.plot(nproc,timings)
#plt.show()


filepath = 'weakscaling.txt'
nruns=3
nprocsList,timingList,errorList = parse_file(filepath,nruns)
nprocs = np.array(nprocsList)
timings = np.array(timingList)


#slope = timingList[0]/2
xsamples = np.arange(35)
#ysamples = [slope*x for x in xsamples]
#plt.plot(xsamples, ysamples, '-', linewidth=3)

#plt.plot(nprocsList,timingList,'*', linewidth=3, markersize=14)
plt.plot(xsamples, timings[0]*np.ones(35), '-', label='ideal')
plt.plot(nprocs,timings,'*', linewidth=3, markersize=14)
plt.title('Weak scaling: 125,000 DOF/CPU', fontsize=fsmed)
plt.xlabel('Number of processors', fontsize=fsmed)

#plt.plot(nproc,timings,'-*', linewidth=3, markersize=14, label='Dynamically Controlled Processor Number')
#plt.plot(nprocStatic, timingsStatic, '-*', linewidth=3, markersize=14, label='Static Processor Number')

#plt.title('Walltime vs Ncpu using RB ordering (~1e6 DOF)', fontsize=fsmed)
#plt.xlabel('Number of processors at finest grid', fontsize=fsmed)
plt.ylabel('Walltime [s]', fontsize=fsmed)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=fsmed*0.8)
ax.legend()

plt.show()



