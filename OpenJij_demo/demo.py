import openjij as oj
n = 10
h, J = {}, {}
for i in range(n-1):
    for j in range(i+1, n):
        J[i, j] = -1
sampler = oj.SASampler()
response = sampler.sample_ising(h, J, num_reads=100)
# minimum energy state
print(response.first.sample)
# {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1}
# or
# {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
# indices (labels) of state (spins)
print(response.indices)
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]