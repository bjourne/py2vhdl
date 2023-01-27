N = 4
__meta__ = {
    'types' : ['real', 'real_vector'],
    'inputs' : [('x', (N,))], 'outputs' : [('y', (N,))],
    'libraries' : ['ieee'],
    'uses' : [
        'ieee.std_logic_1164.all',
        'ieee.numeric_std.all',
        'ieee.math_real.all'
    ]
}
s = 0
for i in range(N):
    s += x[i] * x[i]
for i in range(N):
    y[i] = x[i] / sqrt(s)
