from ast import *
import operator

PROG = '''
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
'''

LEAF_NODES = {
    Add, Constant, FloorDiv, Sub,
    Eq, Lt, LtE,
    Mod, Mult, Name, Pass,
    USub
}

def rewrite_childs(childs, prefun, postfun):
    childs = [rewrite(n, prefun, postfun) for n in childs]
    childs = [c for c in childs if c]
    childs2 = []
    for c in childs:
        if isinstance(c, list):
            childs2.extend(c)
        else:
            childs2.append(c)
    return childs2

def rewrite(node, prefun, postfun):
    node = prefun(type(node), node)
    if not node:
        return node
    tp = type(node)
    if tp == Assign:
        targets = [rewrite(t, prefun, postfun) for t in node.targets]
        value = rewrite(node.value, prefun, postfun)
        node = tp(targets, value)
    elif tp == AugAssign:
        target = rewrite(node.target, prefun, postfun)
        op = rewrite(node.op, prefun, postfun)
        value = rewrite(node.value, prefun, postfun)
        node = tp(target, op, value)
    elif tp == BinOp:
        left = rewrite(node.left, prefun, postfun)
        op = rewrite(node.op, prefun, postfun)
        right = rewrite(node.right, prefun, postfun)
        node = tp(left, op, right)
    elif tp == Call:
        pass
    elif tp == Compare:
        left = rewrite(node.left, prefun, postfun)
        ops = [rewrite(n, prefun, postfun) for n in node.ops]
        comparators = [rewrite(n, prefun, postfun) for n in node.comparators]
        node = tp(left, ops, comparators)
    elif tp == Expr:
        node = tp(rewrite(node.value, prefun, postfun))
    elif tp == For:
        target = rewrite(node.target, prefun, postfun)
        iter = rewrite(node.iter, prefun, postfun)
        childs = rewrite_childs(node.body, prefun, postfun)
        node = tp(target, iter, childs)
    elif tp == FunctionDef:
        childs = rewrite_childs(node.body, prefun, postfun)
        node = tp(node.name, node.args, childs, [])
    elif tp == If:
        test = rewrite(node.test, prefun, postfun)
        trues = rewrite_childs(node.body, prefun, postfun)
        falses = rewrite_childs(node.orelse, prefun, postfun)
        node = tp(test, trues, falses)
    elif tp == List:
        childs = rewrite_childs(node.elts, prefun, postfun)
        node = tp(childs, node.ctx)
    elif tp == Dict:
        pass
    elif tp == Module:
        childs = rewrite_childs(node.body, prefun, postfun)
        node = tp(childs, [])
    elif tp == Return:
        value = rewrite(node.value, prefun, postfun)
        node = tp(value)
    elif tp == Subscript:
        value = rewrite(node.value, prefun, postfun)
        slice = rewrite(node.slice, prefun, postfun)
        node = tp(value, slice, node.ctx)
    elif tp == Tuple:
        childs = [rewrite(n, prefun, postfun) for n in node.elts]
        childs = [c for c in childs if c]
        node = Tuple(childs, node.ctx)
    elif tp == While:
        test = rewrite(node.test, prefun, postfun)
        childs = rewrite_childs(node.body, prefun, postfun)
        node = While(test, childs, [])
    elif tp == UnaryOp:
        node = UnaryOp(rewrite(node.op, prefun, postfun),
                       rewrite(node.operand, prefun, postfun))
    elif tp in LEAF_NODES:
        pass
    else:
        print(tp)
        assert False
    fix_missing_locations(node)
    return postfun(tp, node)

def post_rewrite(node, postfun):
    return rewrite(node, lambda tp, n: n, postfun)

def pre_rewrite(node, prefun):
    return rewrite(node, prefun, lambda tp, n: n)

def intrinsic_zeros(args):
    if not args:
        return Constant(0)
    childs = [intrinsic_zeros(args[1:]) for _ in range(args[0].value)]
    return List(childs, Load())

INTRINSICS = {
    'zeros' : intrinsic_zeros
}

def setdef(defs, key, value):
    defs[-1][key] = value

def setelem(node, path, src):
    tp = type(node)
    if tp == List:
        for p in path[:-1]:
            node = node.elts[p]
        node.elts[path[-1]] = src
    elif tp == Subscript:
        return setelem(node.value, [node.slice.value] + path, src)

def setdef2(defs, dst, src):
    tp = type(dst)
    if tp == Name:
        setdef(defs, dst.id, src)
    elif tp == Subscript:
        setelem(dst, [], src)

def getdef(defs, key, default = None):
    for d in reversed(defs):
        if key in d:
            return d[key]
    return default

def make_assign(id, expr):
    return Assign([Name(id, Store())], expr)

def make_aug_assign(id, op, value):
    expr = BinOp(Name(id, Load()), op, value)
    return make_assign(id, expr)

def make_compare(id, cmp, expr):
    return Compare(Name(id, Load()), [cmp], [expr])

def desugar(tp, node):
    if tp == AugAssign:
        return make_aug_assign(node.target.id, node.op, node.value)
    elif tp == For:
        target = node.target
        id = target.id
        iter = node.iter

        assert type(iter) == Call
        top = iter.args[0]

        body = node.body
        init = make_assign(id, Constant(0))
        test = make_compare(id, Lt(), top)
        inc = make_aug_assign(id, Add(), Constant(1))
        whl = While(test, node.body + [inc], [])
        nodes = [init, whl]
        return nodes
    return node

OPS = {Add : operator.add,
       Sub : operator.sub,
       Eq : operator.eq,
       Lt : operator.lt,
       LtE : operator.le,
       Mult : operator.mul,
       USub : operator.neg}

OPS_STRINGS = {
    Add : '+',
    Sub : '-',
    Mult : '*',
    Eq : '==',
    Lt : '<',
    LtE : '<='
}

def const_eval(node):
    tp = type(node)
    if tp == BinOp:
        left = node.left
        op = node.op
        top = type(op)
        fun = OPS[top]
        right = node.right
        ltp = type(left)
        rtp = type(right)
        if ltp == Constant and left.value == 0 and top in {Add, Sub}:
            return right
        if rtp == Constant and right.value == 0 and top in {Add, Sub}:
            return left
        if ltp == rtp == Constant:
            return Constant(fun(left.value, right.value))
        return node
    elif tp == UnaryOp:
        op = node.op
        operand = node.operand
        fun = OPS[type(op)]
        return Constant(fun(operand.value))
    elif tp == Compare:
        # Only one
        left = node.left
        op = node.ops[0]
        right = node.comparators[0]
        fun = OPS[type(op)]
        if type(left) == type(right) == Constant:
            return Constant(fun(left.value, right.value))
        return node
    elif tp == Subscript:
        value = node.value
        slice = node.slice
        if type(value) == List and type(slice) == Constant:
            return node.value.elts[slice.value]
        return node
    else:
        print(tp)
        assert False

def collect_inputs(tp, node, defs, inputs):
    if tp in {Subscript, Name}:
        expr = unparse(node)
        if expr not in defs:
            inputs[expr] = node, node
    return node

def bind_subst(defs):
    def subst(tp, node):
        return subst_vars(tp, node, defs)
    return subst

def subst_vars(tp, node, defs):
    if tp == Assign:
        tgt = node.targets[0]
        setdef2(defs, node.targets[0], node.value)
    elif tp == BinOp:
        return const_eval(node)
    elif tp == Call:
        # Evaluate args
        args = [post_rewrite(a, bind_subst(defs)) for a in node.args]
        fname = node.func.id

        f = INTRINSICS.get(fname)
        if f:
            return f(args)
        params, expr = getdef(defs, fname)

        # Calling context
        defs = [{p : a for (p, a) in zip(params, args)}]
        return post_rewrite(expr, bind_subst(defs))
    elif tp == Compare:
        return const_eval(node)
    elif tp == UnaryOp:
        return const_eval(node)
    elif tp == Name and type(node.ctx) == Load:
        node = getdef(defs, node.id, node)
    elif tp == Subscript and type(node.ctx) == Load:
        return const_eval(node)
    return node

def ensure_name(node, stage, cnt, introduced):
    expr = unparse(node)
    if expr not in introduced:
        stage = Name('s%d' % stage)
        idx = Constant(cnt[0])
        v = Subscript(stage, idx, Load())
        introduced[expr] = v, node
        cnt[0] += 1
        return v
    return introduced[expr][0]

def operand_ok(node, inputs):
    tp = type(node)
    if tp == Constant:
        return True
    expr = unparse(node)
    return expr in inputs

def elim_binop(node, inputs, stage, cnt, introduced):
    l = node.left
    r = node.right
    ltp = type(l)
    rtp = type(r)
    lok = operand_ok(l, inputs)
    rok = operand_ok(r, inputs)
    if lok and rok:
        return ensure_name(node, stage, cnt, introduced)
    return node

def elim_name(node, inputs, stage, cnt, introduced):
    expr = unparse(node)
    if not expr in inputs:
        return node
    return ensure_name(node, stage, cnt, introduced)

def pipeline_exprs(defs, inputs, gen):
    introduced = {}
    cnt = [0]
    def pipeline_binops(tp, node):
        if tp == BinOp:
            return elim_binop(node, inputs, gen, cnt, introduced)
        return node
    def pipeline_names(tp, node):
        if tp in {Name, Subscript}:
            return elim_name(node, inputs, gen, cnt, introduced)
        return node

    defs = {k : post_rewrite(v, pipeline_binops)
            for (k, v) in defs.items()}
    defs = {k : post_rewrite(v, pipeline_names)
            for (k, v) in defs.items()}
    return introduced, defs

def fully_pipelined(node):
    tp = type(node)
    if tp in {Constant, Name}:
        return True
    elif tp == List:
        return all(fully_pipelined(n) for n in node.elts)
    elif tp == Subscript:
        return type(node.value) == Name and type(node.slice) == Constant
    return False

def to_vhdl(node):
    tp = type(node)
    if tp == Name:
        return node.id
    elif tp == Constant:
        return str(node.value)
    elif tp == Subscript:
        return '%s(%s)' % (to_vhdl(node.value), to_vhdl(node.slice))
    elif tp == BinOp:
        left = to_vhdl(node.left)
        right = to_vhdl(node.right)
        op = to_vhdl(node.op)
        return '%s %s %s' % (left, op, right)
    elif tp == List:
        return '(%s)' % ', '.join(to_vhdl(n) for n in node.elts)
    elif tp in OPS:
        return OPS_STRINGS[tp]
    else:
        print(tp)
        assert False

def main():
    mod = parse(PROG)
    defstack = [{}]
    def pre_subst(tp, node):
        if tp == FunctionDef:
            defstack.append({})
        elif tp == If:
            test = node.test
            trues = node.body
            falses = node.orelse
            test = rewrite(test, pre_subst, post_subst)
            assert type(test) == Constant

            nodes = trues if test.value else falses
            for node in nodes:
                rewrite(node, pre_subst, post_subst)
            return None
        elif tp == While:
            while True:
                test = rewrite(node.test, pre_subst, post_subst)
                assert type(test) == Constant
                if test.value:
                    for n in node.body:
                        rewrite(n, pre_subst, post_subst)
                else:
                    break
            return None
        return node
    def post_subst(tp, node):
        node = subst_vars(tp, node, defstack)
        if tp == FunctionDef:
            defstack.pop()
            params = [a.arg for a in node.args.args]
            ret = node.body[-1]
            setdef(defstack, node.name, (params, ret.value))
            return None
        elif tp == If:
            defstack.pop()
            return None
        return node

    print('Desugaring...')
    mod = post_rewrite(mod, desugar)

    print('Substituting...')
    mod = rewrite(mod, pre_subst, post_subst)

    targets = {'y'}
    defs = {}
    none = Constant(None)
    for t in sorted(targets):
        defs[t] = defstack[-1].get(t, none)

    print('Collecting inputs...')

    # Inputs is a dict from expr to node
    inputs = {}
    def fun(tp, n):
        return collect_inputs(tp, n, defs, inputs)
    mod = post_rewrite(mod, fun)

    stage = 1
    vars = []
    while not all(fully_pipelined(n) for n in defs.values()):
        inputs, defs = pipeline_exprs(defs, inputs, stage)
        for lv, rv in inputs.values():
            vars.append((lv, rv))
        inputs = {unparse(lv) for (lv, _) in inputs.values()}
        stage += 1

    # Add targets
    for expr, n in defs.items():
        vars.append((parse(expr).body[0].value, n))

    for lv, rv in vars:
        lv = to_vhdl(lv)
        rv = to_vhdl(rv)
        print('%-6s <= %s;' % (lv, rv))

main()
