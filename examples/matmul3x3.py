__meta__ = {
    'types' : ['integer', 'integer_vector', 'integer_array2d_t']
    'neutral' : '0',
    'entity' : 'matmul3x3',
    'inputs' : [('a', (3, 3)), ('b', (3, 3))],
    'outputs' : [('c', (3, 3))],

    # Declarations
    'libraries' : ['bjourne', 'ieee'],
    'uses' : [
        'bjourne.types.all',
        'ieee.std_logic_1164.all',
        'ieee.numeric_std.all'
    ]
}

c = zeros(3, 3)
for i in range(3):
    for j in range(3):
        for k in range(3):
            c[i][j] += a[i][j] * b[j][k]
