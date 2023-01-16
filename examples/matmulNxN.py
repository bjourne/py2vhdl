N = 5
__meta__ = {
    'value_type' : 'integer',
    'neutral' : '0',
    'array_types' : {
        'integer' : ['integer_vector', 'integer_array2d_t']
    },
    'entity' : 'matmulNxN',
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

c = zeros(N, N)
for i in range(N):
    for j in range(N):
        for k in range(N):
            c[i][j] += a[i][j] * b[j][k]
