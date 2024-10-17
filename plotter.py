import matplotlib.pyplot as plt

pixels = [256, 512, 1024, 2048]
methods = ["openmp", "mpi", "cuda"]
solvers = ["jacobi", "gauss_seidel"]
threads = [1, 2, 4, 8, 16]

# Initialize the dictionary
tests = {method: {solver: {thread: {"x": pixels, "y": [0] * len(pixels)} for thread in threads} for solver in solvers} for method in methods}

# OpenMP
tests["openmp"]["jacobi"][1]["y"] = [10, 20, 30, 40]
tests["openmp"]["jacobi"][2]["y"] = [15, 25, 35, 45]
tests["openmp"]["jacobi"][4]["y"] = [20, 30, 40, 50]
tests["openmp"]["jacobi"][8]["y"] = [25, 35, 45, 55]
tests["openmp"]["jacobi"][16]["y"] = [30, 40, 50, 60]

# tests["openmp"]["gauss_seidel"][1]["y"] = [10, 20, 30, 40]
# tests["openmp"]["gauss_seidel"][2]["y"] = [10, 20, 30, 40]
# tests["openmp"]["gauss_seidel"][4]["y"] = [10, 20, 30, 40]
# tests["openmp"]["gauss_seidel"][8]["y"] = [10, 20, 30, 40]
# tests["openmp"]["gauss_seidel"][16]["y"] = [10, 20, 30, 40]

# MPI
# tests["mpi"]["gauss_seidel"][4]["y"] = [15, 25, 35, 45]

# Plotting
plt.figure()
for thread in threads:
    plt.plot(tests["openmp"]["jacobi"][thread]["x"], tests["openmp"]["jacobi"][thread]["y"], label=f'Threads {thread}')
plt.title('OpenMP Jacobi Solver')
plt.xlabel('Pixels')
plt.ylabel('Time')
plt.legend()
plt.show()