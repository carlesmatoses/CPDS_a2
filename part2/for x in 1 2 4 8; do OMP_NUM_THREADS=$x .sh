make heat-omp
for x in 1 2 3 4 5 6 7 8; do OMP_NUM_THREADS=$x ./heat-omp test.dat; done