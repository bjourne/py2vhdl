__meta__ = {
    'value_type' : 'real',
    'neutral_value' : '0.0',
    'array_types' : {
        'real' : ['real_vector', 'real_array2d_t']
    },
    'inputs' : [
        ('x', (8, 8))
    ],
    'outputs' : [
        ('y', (8, 8))
    ],
    'libraries' : ['bjourne', 'ieee'],
    'uses' : [
        'bjourne.types.all',
        'ieee.std_logic_1164.all',
        'ieee.numeric_std.all'
    ]
}

def loeffler(x):
    C1_A = -0.78569495838
    C1_B = -1.17587560241
    C1_C =  0.98078528040
    C3_A = -0.27589937928
    C3_B = -1.38703984532
    C3_C =  0.83146961230

    C6_A =  0.76536686473
    C6_B = -1.84775906502
    C6_C =  0.54119610014

    NORM1 = 0.35355339059
    NORM2 = 0.5

    s10 = x[0] + x[7]
    s11 = x[1] + x[6]
    s12 = x[2] + x[5]
    s13 = x[3] + x[4]

    s14 = x[3] - x[4]
    s15 = x[2] - x[5]
    s16 = x[1] - x[6]
    s17 = x[0] - x[7]

    s20 = s10 + s13
    s21 = s11 + s12
    s22 = s11 - s12
    s23 = s10 - s13

    c3_rot_tmp = C3_C * (s14 + s17)
    c1_rot_tmp = C1_C * (s15 + s16)

    s24 = C3_A * s17 + c3_rot_tmp
    s25 = C1_A * s16 + c1_rot_tmp
    s26 = C1_B * s15 + c1_rot_tmp
    s27 = C3_B * s14 + c3_rot_tmp

    c6_rot_tmp = C6_C * (s22 + s23)

    s30 = s20 + s21
    s31 = s20 - s21
    s34 = s24 + s26
    s35 = s27 - s25
    s36 = s24 - s26
    s37 = s27 + s25

    y = zeros(8)
    y[0] = NORM1 * s30
    y[1] = NORM1 * (s37 + s34)
    y[2] = NORM1 * (C6_A * s23 + c6_rot_tmp)
    y[3] = NORM2 * s35
    y[4] = NORM1 * s31
    y[5] = NORM2 * s36
    y[6] = NORM1 * (C6_B * s22 + c6_rot_tmp)
    y[7] = NORM1 * (s37 - s34)
    return y

def transp8x8(x):
    y = zeros(8, 8)
    for i in range(8):
        for j in range(8):
            y[i][j] = x[j][i]
    return y
y = zeros(8)
for i in range(8):
    y[i] = loeffler(x[i])
y = transp8x8(y)
for i in range(8):
    y[i] = loeffler(y[i])
y = transp8x8(y)
