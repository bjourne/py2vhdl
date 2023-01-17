N = 5
__meta__ = {
    'neutral' : '0',
    'types' : ['integer', 'integer_vector', 'integer_array2d_t'],
    'inputs' : [
        ('a', (N, N)),
        ('b', (N, N)),
    ],
    'outputs' : [
        ('c', (N, N))
    ],

    # Declarations
    'libraries' : ['bjourne', 'ieee'],
    'uses' : [
        'bjourne.types.all',
        'ieee.std_logic_1164.all',
        'ieee.numeric_std.all'
    ]
}

for i in range(N):
    for j in range(N):
        for k in range(N):
            c[i][j] += a[i][j] * b[j][k]
