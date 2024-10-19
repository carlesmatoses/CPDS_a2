import matplotlib.pyplot as plt
plt.style.use('bmh')

pixels = [256, 512, 1024, 2048]
methods = ["openmp", "mpi", "cuda"]
solvers = ["jacobi", "gauss_seidel"]
threads = [1, 2, 4, 8, 16]

# Initialize the dictionary
tests = {method: {solver: {thread: {"x": pixels, "y": [0] * len(pixels)} for thread in threads} for solver in solvers} for method in methods}

# OpenMP
tests["openmp"]["jacobi"][1]["y"] = [2.392, 20.846, 140.121] #8
tests["openmp"]["jacobi"][1]["x"] = pixels[:3]
tests["openmp"]["jacobi"][2]["y"] = [1.248, 10.939, 76.065] #15
tests["openmp"]["jacobi"][2]["x"] = pixels[:3]
tests["openmp"]["jacobi"][4]["y"] = [0.714, 5.001, 34.738, 271.630]
tests["openmp"]["jacobi"][8]["y"] = [0.488, 2.804, 19.375, 163.013]
tests["openmp"]["jacobi"][16]["y"] = [0.450, 1.831, 9.097, 110.765] 

# tests["openmp"]["gauss_seidel"][1]["y"] = [10, 20, 30, 40]
# tests["openmp"]["gauss_seidel"][2]["y"] = [10, 20, 30, 40]
# tests["openmp"]["gauss_seidel"][4]["y"] = [10, 20, 30, 40]
# tests["openmp"]["gauss_seidel"][8]["y"] = [10, 20, 30, 40]
# tests["openmp"]["gauss_seidel"][16]["y"] = [10, 20, 30, 40]

# MPI
# tests["mpi"]["gauss_seidel"][4]["y"] = [15, 25, 35, 45]

# Plotting
line_256 = [[2.392, 1.248,  0.714, 0.488, 0.450],
            [1,2,4,8,16],"256"]
line_512 = [[20.846, 10.939, 5.001, 2.804, 1.831],
            [1,2,4,8,16],"512"]
line_1024 = [[140.121, 76.065, 34.738, 19.375, 9.097],
             [1,2,4,8,16],"1024"]
line_2048 = [[271.630, 163.013, 110.765],
             [4,8,16],"2048"]

# execution time
plt.figure()
for line in [line_256, line_512, line_1024, line_2048]:
    x = line[1]
    y = line[0]
    plt.plot(x, y, label=f'Resolution {line[2]}', marker='o')
plt.title('OpenMP Jacobi Solver')
plt.xlabel('Number of Threads')
plt.ylabel('Execution Time')
plt.legend()
plt.show()

# speedup
line_perfect = [[1, 2, 4, 8, 16],[1,2,4,8,16],"Perfect"]
plt.figure()
for line in [line_256, line_512, line_1024]:
    x = line[1]
    y = [line[0][0]/val for val in line[0]]
    plt.plot(x, y, label=f'Resolution {line[2]}', marker='o')
plt.plot(line_perfect[0], line_perfect[0], linestyle='dotted', color='black')
plt.title('OpenMP Jacobi Solver Speedup')
plt.xlabel('Number of Threads')
plt.ylabel('Speedup')
plt.legend()
plt.show()

# efficiency
plt.figure()
for line in [line_256, line_512, line_1024]:
    x = line[1]
    y = [line[0][0]/(Tp*P) for Tp, P in zip(line[0],line[1])]
    plt.plot(x, y, label=f'Resolution {line[2]}', marker='o')
plt.title('OpenMP Jacobi Solver Efficiency')
plt.xlabel('Number of Threads')
plt.ylabel('Efficiency')
plt.legend()
plt.show()