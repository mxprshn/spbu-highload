mpic++ -o MPI_LUP MPI_LUP.cpp

for i in {2..6}
do
    mpiexec -n $i ./MPI_LUP 1000 73
done

for i in {100..1000..100}
do
    mpiexec -n 6 ./MPI_LUP $i 73
done
