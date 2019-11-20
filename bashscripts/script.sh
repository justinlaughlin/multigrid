#!/bin/bash

#1 3 4 5 6 7
rm -rf ~/gitrepos/graduate_homework/phys244/final/multigrid/build/*
cd ~/gitrepos/graduate_homework/phys244/final/multigrid/build
cmake ../
make -j12

filename='../bashscripts/weakscaling.txt'

for N in 33 65 129 257 513 1025 2049 4098
for N in 1025
do
    echo ' ' >> $filename
    echo "Gridpoints per dimension: $N" >> $filename
    for i in 1 3 4 5 6 7 9 11 13 15 17 19 22 25 30 35 45
    do 
        echo ' ' >> $filename
        echo "Timing for nProcs = $i" >> $filename
        for j in {1..3}
        do
            mpiexec -np $i ~/gitrepos/graduate_homework/phys244/final/multigrid/build/bin/main $N | grep '^^^'  >> $filename
        done
    done
done


# weak scaling
#N=513
#nProc=3
#echo "weak scaling test. (N=513,nProc=3), (N=1025,nProc=9), (N=2049,nProc=33)" >> $filename
#echo "Gridpoints per dimension: $N" >> $filename
#echo "Timing for nProcs = $nProc" >> $filename
#for j in {1..3}
#do
#    mpiexec -np $nProc ~/gitrepos/graduate_homework/phys244/final/multigrid/build/bin/main $N | grep '^^^'  >> $filename
#done
#
#N=1025
#nProc=9
#echo "Gridpoints per dimension: $N" >> $filename
#echo "Timing for nProcs = $nProc" >> $filename
#for j in {1..3}
#do
#    mpiexec -np $nProc ~/gitrepos/graduate_homework/phys244/final/multigrid/build/bin/main $N | grep '^^^'  >> $filename
#done
#
#N=2049
#nProc=33
#echo "Gridpoints per dimension: $N" >> $filename
#echo "Timing for nProcs = $nProc" >> $filename
#mpiexec -np $nProc ~/gitrepos/graduate_homework/phys244/final/multigrid/build/bin/main $N | grep '^^^'  >> $filename
#
