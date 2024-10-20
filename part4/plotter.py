cuda_2 = {
    256: {"cpu": 11426.086914/1000, "gpu":7583.697754/1000},
    512: {"cpu": 76074.523438/1000, "gpu": 38464.609375/1000},
    1024: {"cpu": 466527.000000/1000, "gpu": 216369.687500/1000},
}

speedup = {
    256:cuda_2[256]["cpu"]/cuda_2[256]["gpu"],
    512:cuda_2[512]["cpu"]/cuda_2[512]["gpu"],
    1024:cuda_2[1024]["cpu"]/cuda_2[1024]["gpu"],
}

import matplotlib.pyplot as plt
plt.style.use('bmh')

sizes = [256, 512, 1024]
cpu_times = [cuda_2[size]["cpu"] for size in sizes]
gpu_times = [cuda_2[size]["gpu"] for size in sizes]

x = range(len(sizes))
plt.bar(x, cpu_times, width=0.4, label='CPU', align='center')
plt.bar(x, gpu_times, width=0.4, label='GPU', align='edge')
plt.xlabel('Sizes')
plt.ylabel('Time (s)')
plt.title('CPU vs GPU Execution Time')
plt.xticks(x, sizes)
plt.legend()
plt.show()

plt.figure()
speedup_values = [speedup[size] for size in sizes]
plt.bar(x, speedup_values, width=0.4, label='Speedup', align='center')
plt.xlabel('Sizes')
plt.ylabel('Speedup (CPU/GPU)')
plt.title('Speedup of CPU over GPU')
plt.xticks(x, sizes)
plt.legend()
plt.show()


efficiency = {size: speedup[size] / size for size in sizes}
efficiency_values = [efficiency[size] for size in sizes]
plt.figure()
plt.bar(x, efficiency_values, width=0.4, label='Efficiency', align='center')
plt.xlabel('Sizes')
plt.ylabel('Efficiency (Speedup/Size)')
plt.title('Efficiency of CPU over GPU')
plt.xticks(x, sizes)
plt.legend()
plt.show()
