mpic++ -o MPI_LUP MPI_LUP.cpp

for i in {2..12}
do
    mpiexec -n $i ./MPI_LUP 1000
done

for i in {100..1000..100}
do
    mpiexec -n 12 ./MPI_LUP $i
done
