#plotTimings.py


import numpy as np
import matplotlib.pyplot as plt

from resultParser import *

fsmed = 16

# Number of processors is static (does not dynamically change based on degrees of freedom)
# RB ordering
# N = 1025 (held constant)



filepath1025 = 'timings_strongScaling_dynamicProcs_3runs.txt'
nruns1025 = 3
filepath65 = 'VariableTimings/ndof65'
filepath129 = 'VariableTimings/ndof129'
filepath257 = 'VariableTimings/ndof257'
filepath513 = 'VariableTimings/ndof513'
nrunsrest = 5

nprocsList1025, timingList1025, errorList1025 = parse_file(filepath1025, nruns1025)
nprocsList65, timingList65, errorList65 = parse_file(filepath65, nrunsrest)
nprocsList129, timingList129, errorList129 = parse_file(filepath129, nrunsrest)
nprocsList257, timingList257, errorList257 = parse_file(filepath257, nrunsrest)
nprocsList513, timingList513, errorList513 = parse_file(filepath513, nrunsrest)

nproc1025 = np.array(nprocsList1025) #+ 0.7*np.array(nprocsList2)
timings1025 = np.array(timingList1025)# + 0.7*np.array(timingList2)
error1025 = np.array(errorList1025)# + 0.7*np.array(errorList2)

nproc65 = np.array(nprocsList65)
timings65 = np.array(timingList65)
error65 = np.array(errorList65)

nproc129 = np.array(nprocsList129)
timings129 = np.array(timingList129)
error129 = np.array(errorList129)

nproc257 = np.array(nprocsList257)
timings257 = np.array(timingList257)
error257 = np.array(errorList257)

nproc513 = np.array(nprocsList513)
timings513 = np.array(timingList513)
error513 = np.array(errorList513)

#nrunsST1 = 3
#nrunsST2 = 7
#nprocsListStatic, timingListStatic, errorListStatic = parse_file(filepathStatic1, nrunsST1)
#nprocsListStatic2, timingListStatic2, errorListStatic2 = parse_file(filepathStatic2, nrunsST2)
#nprocStatic = 0.3*np.array(nprocsListStatic) + 0.7*np.array(nprocsListStatic2)
#timingsStatic = 0.3*np.array(timingListStatic) + 0.7*np.array(timingListStatic2)
#errorStatic = 0.3*np.array(errorListStatic) + 0.7*np.array(errorListStatic2)


#plt.plot(nproc,timings)
#plt.show()

plt.plot(nproc65, timings65/timings65[0], '-*', linewidth=3, markersize=14, label='Ndof=3969')
plt.plot(nproc129, timings129/timings129[0], '-*', linewidth=3, markersize=14, label='Ndof=16129')
plt.plot(nproc257, timings257/timings257[0], '-*', linewidth=3, markersize=14, label='Ndof=65025')
plt.plot(nproc513, timings513/timings513[0], '-*', linewidth=3, markersize=14, label='Ndof=261121')
plt.plot(nproc1025, timings1025/timings1025[0], '-*', linewidth=3, markersize=14, label='Ndof=1046529')

plt.title('Scaling for Different Problem Dimensions', fontsize=fsmed)
plt.xlabel('Number of processors', fontsize=fsmed)
plt.ylabel('Normalized Walltime', fontsize=fsmed)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=fsmed*0.8)
ax.legend()

plt.show()



