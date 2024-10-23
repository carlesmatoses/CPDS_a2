import matplotlib.pyplot as plt
plt.style.use('bmh')

pixels = [256, 512, 1024]
methods = ["openmp", "mpi", "cuda"]
solvers = ["jacobi", "gauss_seidel"]
threads = [1, 2, 4, 8, 16]

# Initialize the dictionary
tests = {method: {solver: {thread: {"x": pixels, "y": [0] * len(pixels)} for thread in threads} for solver in solvers} for method in methods}

# OpenMP
tests["mpi"]["jacobi"][1]["y"] =  [2.392, 20.846, 140.121] #8
tests["mpi"]["jacobi"][2]["y"] =  [1.248, 10.939, 76.065] #15
tests["mpi"]["jacobi"][4]["y"] =  [0.714, 5.001, 34.738]
tests["mpi"]["jacobi"][8]["y"] =  [0.488, 2.804, 19.375]
tests["mpi"]["jacobi"][16]["y"] = [0.450, 1.831, 9.097] 

# tests["openmp"]["gauss_seidel"][1]["y"] = [10, 20, 30, 40]
# tests["openmp"]["gauss_seidel"][2]["y"] = [10, 20, 30, 40]
# tests["openmp"]["gauss_seidel"][4]["y"] = [10, 20, 30, 40]
# tests["openmp"]["gauss_seidel"][8]["y"] = [10, 20, 30, 40]
# tests["openmp"]["gauss_seidel"][16]["y"] = [10, 20, 30, 40]

# MPI
# tests["mpi"]["gauss_seidel"][4]["y"] = [15, 25, 35, 45]

# Plotting
line_256 = [[1.594, 1.590, 0.450, 0.429],
            [1,2,4,8],"256"]
line_512 = [[11.950, 12.054, 3.271, 3.599],
            [1,2,4,8],"512"]
line_1024 = [[88.160, 84.740, 40.630, 43.701],
             [1,2,4,8],"1024"]
line_2048 = [[],
             [1,2,4],"2048"]

# execution time
plt.figure()
for line in [line_256, line_512, line_1024]:
    x = line[1]
    y = line[0]
    plt.plot(x, y, label=f'Resolution {line[2]}', marker='o')
plt.title('MPI Jacobi Solver LOCAL')
plt.xlabel('Number of Threads')
plt.ylabel('Execution Time')
plt.legend()
plt.xlim(1, 16)
plt.show()

# speedup
line_perfect = [[1, 2, 4, 8, 16],[1,2,4,8,16],"Perfect"]
plt.figure()
for line in [line_256, line_512, line_1024]:
    x = line[1]
    y = [line[0][0]/val for val in line[0]]
    plt.plot(x, y, label=f'Resolution {line[2]}', marker='o')
plt.plot(line_perfect[0], line_perfect[0], linestyle='dotted', color='black')
plt.title('MPI Jacobi Solver LOCAL Speedup')
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
plt.title('MPI Jacobi Solver LOCAL Efficiency')
plt.xlabel('Number of Threads')
plt.ylabel('Efficiency')
plt.legend()
plt.xlim(1, 16)
plt.show()