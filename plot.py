import matplotlib.pyplot as plt
import numpy as np

# Measured execution time for serial execution (1 core)
serial_time = 127.73628807067871  # Replace with your measured serial execution time

# Proportion of the program that is sequential
sequential_proportion = 0.75  # Adjust as needed

# Calculate the theoretical speedup using Amdahl's law
def theoretical_speedup(P, k):
    return 1 / ((1 - P) + P / k)

# Number of cores (k)
cores = [1, 2, 4, 8, 16]

# Calculate the measured speedup
measured_speedup = [serial_time / (serial_time / core) for core in cores]

# Calculate the theoretical speedup based on Amdahl's law
theoretical_speedup_values = [theoretical_speedup(sequential_proportion, core) for core in cores]

# Create a plot
plt.figure(figsize=(8, 6))
plt.plot(cores, measured_speedup, marker='o', label='Measured Speedup')
plt.plot(cores, theoretical_speedup_values, marker='x', label='Theoretical Speedup (Amdahl)')
plt.xlabel('Number of Cores (k)')
plt.ylabel('Speedup')
plt.title('Measured vs. Theoretical Speedup')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()