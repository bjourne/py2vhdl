# Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
from ast import *
from copy import deepcopy
from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template
from pathlib import Path
from sys import argv
import operator

VHDL_TMPL = '''
{% for key in meta['libraries'] -%}
library {{ key }};
{% endfor -%}
{% for key in meta['uses'] -%}
use {{ key }};
{% endfor %}
entity {{ entity }} is
    port (
        clk, rstn : in std_logic;
        {% for dir, name, tp in iface -%}
        {{ name }} : {{ dir }} {{ tp }}{% if not loop.last %};{% endif %}
        {% endfor %}
    );
end {{ entity }};

architecture rtl of {{ entity }} is
    {% for sig_name, sig_vars in vars -%}
    signal {{ sig_name }} : {{ array_types[0] }}(0 to {{ sig_vars|length - 1 }});
    {% endfor %}
begin
    {% for l, r in targets -%}
    {{ l }} <= {{ r }};
    {% endfor %}
    process (clk)
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                {% for sig_name, _ in vars -%}
                {{ sig_name }} <= (others => {{ meta['neutral'] }});
                {% endfor %}
            else
                {% for _, stage_vars in vars -%}
                {% for l, r in stage_vars -%}
                {{ l }} <= {{ r }};
                {% endfor -%}
                {% endfor %}
            end if;
        end if;
    end process;
end architecture;
'''.strip()

ENV = Environment()
TMPL = ENV.from_string(VHDL_TMPL)

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
        keys = rewrite_childs(node.keys, prefun, postfun)
        values = rewrite_childs(node.values, prefun, postfun)
        node = tp(keys, values)
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

def make_assign(node, expr):
    node = deepcopy(node)
    node.ctx = Store()
    return Assign([node], expr)

def make_aug_assign(node, op, value):
    node = deepcopy(node)
    node.ctx = Load()
    expr = BinOp(node, op, value)
    return make_assign(node, expr)

def make_compare(node, cmp, expr):
    node = deepcopy(node)
    node.ctx = Load()
    return Compare(node, [cmp], [expr])

def desugar(tp, node):
    if tp == AugAssign:
        return make_aug_assign(node.target, node.op, node.value)
    elif tp == For:
        target = node.target
        # id = target.id
        iter = node.iter

        assert type(iter) == Call
        top = iter.args[0]

        body = node.body
        init = make_assign(target, Constant(0))
        test = make_compare(target, Lt(), top)
        inc = make_aug_assign(target, Add(), Constant(1))
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
IDENT_0 = {Add, Sub}

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
        if ltp == Constant and left.value == 0 and top in IDENT_0:
            return right
        if rtp == Constant and right.value == 0 and top in IDENT_0:
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

def fmt_stage(stage):
    return 's%d' % stage

def ensure_name(node, stage, cnt, introduced):
    expr = unparse(node)
    if expr not in introduced:
        stage = Name(fmt_stage(stage))
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

def node_to_vhdl(node):
    tp = type(node)
    if tp == Name:
        return node.id
    elif tp == Constant:
        return str(node.value)
    elif tp == Subscript:
        return '%s(%s)' % (node_to_vhdl(node.value), node_to_vhdl(node.slice))
    elif tp == BinOp:
        left = node_to_vhdl(node.left)
        right = node_to_vhdl(node.right)
        op = node_to_vhdl(node.op)
        return '%s %s %s' % (left, op, right)
    elif tp == List:
        return '\n(%s)' % ', '.join(node_to_vhdl(n) for n in node.elts)
    elif tp in OPS:
        return OPS_STRINGS[tp]
    else:
        print(tp)
        assert False

def vars_to_vhdl(meta, vars, targets):

    value_type = meta['value_type']
    array_types = meta['array_types'][value_type]
    def type_decl(shape):
        tp = value_type
        if shape != 0:
            arr_tp = array_types[len(shape) - 1]
            ranges = [f'(0 to {d - 1})' for d in shape]
            tp = f'{arr_tp}{"".join(ranges)}'
        return tp

    vars = [(stage_name, [(node_to_vhdl(l), node_to_vhdl(r))
                          for (l, r) in stage_vars])
            for (stage_name, stage_vars) in vars]

    targets = [(l, node_to_vhdl(r)) for (l, r) in targets.items()]
    entity = meta['entity']
    iface = [('in', n, sh) for (n, sh) in meta['inputs']]
    iface += [('out', n, sh) for (n, sh) in  meta['outputs']]
    iface = [(d, n, type_decl(sh)) for (d, n, sh) in iface]
    return TMPL.render(
        meta = meta,
        entity = entity,
        iface = iface,
        array_types = array_types,
        vars = vars,
        targets = targets)

def main():
    in_path = Path(argv[1])
    with in_path.open() as f:
        mod = parse(f.read())
    defstack = [{}]
    def pre_subst(tp, node):
        if tp == FunctionDef:
            print('  Traversing %s...' % node.name)
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


    mod = post_rewrite(mod, desugar)

    print('* Symbolic evaluation')
    mod = rewrite(mod, pre_subst, post_subst)

    print('* Get metadata')
    meta = {}
    for node in mod.body:
        tp = type(node)
        if tp == Assign and node.targets[0].id == '__meta__':
            meta = literal_eval(node.value)

    targets = {t[0] for t in meta['outputs']}
    defs = {}
    none = Constant(None)
    for t in sorted(targets):
        defs[t] = defstack[-1].get(t, none)

    # Inputs is a dict from expr to node
    inputs = {}
    def fun(tp, n):
        return collect_inputs(tp, n, defs, inputs)
    mod = post_rewrite(mod, fun)

    # Begin pipelining.
    print('* Pipelining')
    stage = 1
    vars = []
    while not all(fully_pipelined(n) for n in defs.values()):
        inputs, defs = pipeline_exprs(defs, inputs, stage)
        vars.append((fmt_stage(stage), list(inputs.values())))
        inputs = {unparse(lv) for (lv, _) in inputs.values()}
        stage += 1

    vhdl = vars_to_vhdl(meta, vars, defs)

    out_path = Path(in_path.stem + '.vhdl')
    print('* Writing VHDL to "%s".' % out_path)
    with out_path.open('w') as f:
        f.write(vhdl)


main()
