# Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
from ast import *
from collections import defaultdict
from copy import deepcopy
from itertools import product
from jinja2 import Environment, Template
from pathlib import Path
from pygraphviz import AGraph
from re import sub
from rich.box import ASCII2
from rich.console import Console
from rich.table import Table
from sys import argv
import operator


# Utils
def flatten(lst):
    return [item for sublist in lst for item in sublist]

def node_const_str(node):
    v = node.value
    return '%.3f' % v if type(v) == float else str(v)

# AST constructors
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


def make_subscript(var, indices):
    assert indices
    ctx = Load()
    if len(indices) == 1:
        return Subscript(Name(var, ctx), indices[0], ctx)
    ss = make_subscript(var, indices[:-1])
    return Subscript(ss, indices[-1], ctx)


def make_call(fun, args):
    return Call(Name(fun), args, [])

# Print schedule with Rich

console = Console(markup = False)

def node_schedule_label(node, inputs, constfun):
    tp = type(node)
    expr = unparse(node)
    if tp == BinOp:
        fun = OPS_STRINGS[type(node.op)]
        childs = [node_schedule_label(n, inputs, constfun)
                  for n in node_childs(node)]
        return (' %s ' % fun).join(childs)
    elif tp == Call:
        childs = [node_schedule_label(n, inputs, constfun)
                  for n in node_childs(node)]
        childs = ' '.join(childs)
        return '%s %s' % (node.func.id, childs)
    elif tp == Constant:
        return constfun(expr, node)
    elif tp == Subscript:
        if expr in inputs:
            return str(inputs.index(expr))
        return unparse(node.slice)
    return expr

def row_data(nodes, inputs, constfun):
    return [node_schedule_label(n, inputs, constfun) for n in nodes]

def table_data(vars, inputs, constfun):
    vars = [[n for (_, n) in vs] for vs in vars]
    n_cols = max(len(nodes) for nodes in vars)
    rows = [row_data(nodes, inputs, constfun) for nodes in vars]
    rows = [row + ['-'] * (n_cols - len(row)) for row in rows]
    return [[str(i)] + row for i, row in enumerate(rows)]

def print_schedule_rich(entity, data):
    table = Table(title = 'Schedule for %s' % entity, box = ASCII2)

    # Columns
    n_cols = max(len(row) for row in data)
    cols = [('#', 'right')] + [(str(i), 'center') for i in range(n_cols)]
    for name, just in cols:
        table.add_column(name, justify = just)

    for i, row in enumerate(data):
        table.add_row(str(i), *row)

    console.print(table)

def print_latex_table(data):
    def ss(name):
        return sub(r'(\w+)_(\d+|\?)', r'\1_{\2}', str(name))

    def fmt_row(row):
        return ' & '.join('$%s$' % ss(c) for c in row) + '\\\\'
    for row in data:
        print(fmt_row(row))
    print()

def print_constants_rich(entity, data):
    table = Table(title = 'Constants %s' % entity, box = ASCII2)
    table.add_column('Name', 'left')
    table.add_column('Value', 'right')

    for row in data:
        table.add_row(*row)
    console.print(table)

# Plotting
def setup_graph():
    G = AGraph(strict=False, directed=True)
    graph_attrs = {
        "dpi": 300,
        "ranksep": 0.22,
        "fontname": "Inconsolata",
        "bgcolor": "transparent",
    }
    G.graph_attr.update(graph_attrs)
    node_attrs = {
        "shape": "box",
        "width": 0.55,
        "style": "filled",
        "fillcolor": "white",
    }
    G.node_attr.update(node_attrs)
    edge_attrs = {"fontsize": "10pt"}
    G.edge_attr.update(edge_attrs)
    return G


NODE_FILLCOLORS = {
    Subscript: "#ffddee",
    BinOp: "#eeffdd",
    Call: "#ddeeff",
    Constant: "#d9ffff",
    List: "#eeddff",
    Name: "#ffddee",
    "input": "#fff3cc",
}

def node_label(node, use_ss):
    tp = type(node)
    if tp in {Name, Subscript}:
        s = unparse(node)
        if use_ss:
            return sub(r'(\w+)\[(\d+)\]', r'<<i>\1</i><sub>\2</sub>>', s)
        return s
    elif tp == BinOp:
        return OPS_STRINGS[type(node.op)]
    elif tp == Call:
        s = node.func.id
        if s == 'sqrt':
            s = '√'
        return s
    elif tp == List:
        return "[]"
    elif tp == Constant:
        return node_const_str(node)
    print(unparse(node))
    assert False

def node_fillcolor(node, input_vars):
    def ss_name(node):
        val = node.value
        if type(val) == Name:
            return val.id
        return ss_name(val)

    tp = type(node)
    if tp == Subscript and unparse(node) in input_vars:
        return NODE_FILLCOLORS["input"]
    return NODE_FILLCOLORS.get(tp, "white")


def plot_defs(file_name, input_vars, defs, use_ss):
    G = setup_graph()

    keyed_nodes = {}
    adj_list = defaultdict(list)

    def add_node(node):
        key = unparse(node)
        if key in keyed_nodes:
            return key
        tp = type(node)

        childs = []
        if tp == List:
            childs = node.elts
        elif tp == BinOp:
            childs = [node.left, node.right]
        elif tp == Call:
            childs = node.args

        keys = [add_node(n) for n in childs]
        if tp == List:
            return keys
        else:
            keyed_nodes[key] = node
            adj_list[key] = keys
            return key

    def add_indexed_nodes(var, path, keys):
        for i, key in enumerate(keys):
            new_path = path + [i]
            if type(key) == list:
                add_indexed_nodes(var, new_path, key)
            else:
                ss = [Constant(p) for p in new_path]
                new_node = make_subscript(var, ss)
                new_key = unparse(new_node)
                keyed_nodes[new_key] = new_node
                adj_list[new_key] = [key]

    # For scalars to scalars dataflows.
    for var, node in defs.items():
        add_indexed_nodes(var, [], add_node(node))

    indexes = {n: i for (i, n) in enumerate(adj_list)}
    for k1, k2s in adj_list.items():
        i1 = indexes[k1]
        node = keyed_nodes[k1]
        kw = {
            "label": node_label(node, use_ss),
            "fillcolor": node_fillcolor(node, input_vars),
        }
        G.add_node(i1, **kw)
        for k2 in k2s:
            i2 = indexes[k2]
            G.add_edge(i2, i1)
    G.draw(file_name, prog="dot")


# Templating
VHDL_TMPL = """
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
    signal {{ sig_name }} : {{ types[1] }}(0 to {{ sig_vars|length - 1 }});
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
                {{ sig_name }} <= (others => {{ neutral }});
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
""".strip()

ENV = Environment()
TMPL = ENV.from_string(VHDL_TMPL)
LEAF_NODES = {
    Add,
    Constant,
    Div,
    FloorDiv,
    Mod,
    Mult,
    Name,
    Pass,
    Sub,
    Eq,
    Lt,
    LtE,
    USub,
}


# The node's "proper" childs.
def node_childs(node):
    tp = type(node)
    if tp == Call:
        return node.args
    elif tp == BinOp:
        return [node.left, node.right]
    elif tp == List:
        return node.elts
    elif tp == Subscript:
        return [node.value, node.slice]
    elif tp in LEAF_NODES:
        return []
    print(unparse(node))
    assert False


def rewrite_childs(childs, prefun, postfun):
    childs = [rewrite(n, prefun, postfun) for n in childs]
    childs = [c if isinstance(c, list) else [c] for c in childs if c]
    return flatten(childs)


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
        childs = rewrite_childs(node.args, prefun, postfun)
        node = tp(node.func, childs, [])
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
        node = UnaryOp(
            rewrite(node.op, prefun, postfun),
            rewrite(node.operand, prefun, postfun),
        )
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


INTRINSICS = {"zeros": intrinsic_zeros}

# Functions that carry no logic cost and thus shouldn't have a cell in
# the pipeline.
NO_LOGICS = {"std_logic_vector", "to_float"}


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
    else:
        print(tp, unparse(node))
        assert False


def setdef2(defs, dst, src):
    tp = type(dst)
    if tp == Name:
        setdef(defs, dst.id, src)
    elif tp == Subscript:
        setelem(dst, [], src)


def getdef(defs, key, default=None):
    for d in reversed(defs):
        if key in d:
            return d[key]
    return default


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


OPS = {
    Add: operator.add,
    Sub: operator.sub,
    Div: operator.truediv,
    Eq: operator.eq,
    Lt: operator.lt,
    LtE: operator.le,
    Mult: operator.mul,
    USub: operator.neg,
}

OPS_STRINGS = {
    Add: "+",
    Div: "/",
    Sub: "-",
    Mult: "*",
    Eq: "==",
    Lt: "<",
    LtE: "<=",
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
        if top in IDENT_0:
            if ltp == Constant and left.value == 0:
                return right
            if rtp == Constant and right.value == 0:
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

def subst_vars(tp, node, defs):
    def subst_fun(tp, node):
        return subst_vars(tp, node, defs)

    if tp == Assign:
        tgt = node.targets[0]
        setdef2(defs, tgt, node.value)
    elif tp == BinOp:
        return const_eval(node)
    elif tp == Call:
        # Evaluate args
        args = [post_rewrite(a, subst_fun) for a in node.args]
        func = node.func
        name = func.id

        f = INTRINSICS.get(name)
        if f:
            return f(args)
        f = getdef(defs, name)
        if f:
            params, expr = getdef(defs, name)
            # Calling context
            defs = [{p: a for (p, a) in zip(params, args)}]
            return post_rewrite(expr, subst_fun)
        node = tp(func, args, [])
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
    return "s%d" % stage

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

# Merge these?
def fully_pipelined(node):
    tp = type(node)
    if tp in LEAF_NODES | {List, Call, Subscript}:
        childs = node_childs(node)
        return all(fully_pipelined(n) for n in childs)
    return False

def node_is_const(node, inputs):
    tp = type(node)
    if tp == Constant:
        return True
    elif tp in {Name, Subscript}:
        return unparse(node) in inputs
    elif tp == Call:
        if node.func.id not in NO_LOGICS:
            return all(node_is_const(n, inputs) for n in node.args)
    elif tp == BinOp:
        lok = node_is_const(node.left, inputs)
        rok = node_is_const(node.right, inputs)
        return lok and rok
    return False


def pipeline_exprs(defs, inputs, gen):
    introduced = {}
    cnt = [0]

    def pipeline_exprs(tp, node):
        if tp in {BinOp, Call} and node_is_const(node, inputs):
            return ensure_name(node, gen, cnt, introduced)
        return node

    def pipeline_names(tp, node):
        if tp in {Name, Subscript} and unparse(node) in inputs:
            return ensure_name(node, gen, cnt, introduced)
        return node

    defs = {k: post_rewrite(v, pipeline_exprs) for (k, v) in defs.items()}
    defs = {k: post_rewrite(v, pipeline_names) for (k, v) in defs.items()}
    return introduced, defs


def input_nodes(var, shape):
    cells = product(*[range(n) for n in shape])
    cells = [[Constant(c) for c in cell] for cell in cells]
    cells = [make_subscript(var, cell) for cell in cells]
    return cells


def node_to_vhdl(node):
    def comma_nodes(nodes):
        return ", ".join(node_to_vhdl(n) for n in nodes)

    tp = type(node)
    if tp == Name:
        return node.id
    elif tp == Call:
        return "%s(%s)" % (node.func.id, comma_nodes(node.args))
    elif tp == Constant:
        return str(node.value)
    elif tp == Subscript:
        return "%s(%s)" % (node_to_vhdl(node.value), node_to_vhdl(node.slice))
    elif tp == BinOp:
        left = node_to_vhdl(node.left)
        right = node_to_vhdl(node.right)
        op = node_to_vhdl(node.op)
        return "%s %s %s" % (left, op, right)
    elif tp == List:
        return "\n(%s)" % comma_nodes(node.elts)
    elif tp in OPS:
        return OPS_STRINGS[tp]
    else:
        print(tp)
        assert False


def vars_to_vhdl(meta, vars, targets, entity):

    types = meta["types"]

    # Neutral element
    neutrals = {'integer' : '0', 'real' : '0.0'}
    neutral = neutrals[types[0]]

    def type_decl(shape):
        idx = len(shape)
        tp = types[idx]
        if not idx:
            return tp
        ranges = [f"(0 to {d - 1})" for d in shape]
        return f'{tp}{"".join(ranges)}'

    vars = [
        (
            fmt_stage(i + 1),
            [(node_to_vhdl(l), node_to_vhdl(r)) for (l, r) in stage_vars],
        )
        for (i, stage_vars) in enumerate(vars)
    ]

    targets = [(l, node_to_vhdl(r)) for (l, r) in targets.items()]
    iface = [("in", n, sh) for (n, sh) in meta["inputs"]]
    iface += [("out", n, sh) for (n, sh) in meta["outputs"]]
    iface = [(d, n, type_decl(sh)) for (d, n, sh) in iface]
    return TMPL.render(
        meta=meta,
        entity=entity,
        iface=iface,
        types=types,
        neutral=neutral,
        vars=vars,
        targets=targets,
    )

def find_metadata_node(nodes):
    for i, node in enumerate(nodes):
        if type(node) == Assign:
            target = node.targets[0]
            tpt = type(target)
            if tpt == Name and target.id == "__meta__":
                return i, node.value
    assert False


def main():
    in_path = Path(argv[1])
    entity = in_path.stem
    with in_path.open() as f:
        mod = parse(f.read())

    defstack = [{}]

    def pre_subst(tp, node):
        if tp == FunctionDef:
            print("  Traversing %s..." % node.name)
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
            return None
        return node

    mod = post_rewrite(mod, desugar)

    # Add initialization based on the metadata:
    print("* Adding initialization")
    inits = []
    i, meta = find_metadata_node(mod.body)
    for k, v in zip(meta.keys, meta.values):
        if k.value == "outputs":
            for el in v.elts:
                var, args = el.elts
                call = make_call("zeros", args.elts)
                inits.append(make_assign(Name(var.value), call))
    mod.body = mod.body[: i + 1] + inits + mod.body[i + 1 :]

    print("* Symbolic evaluation")
    mod = rewrite(mod, pre_subst, post_subst)

    print("* Get metadata")
    meta = literal_eval(find_metadata_node(mod.body)[1])

    targets = {t[0] for t in meta["outputs"]}
    defs = {}
    none = Constant(None)
    for t in sorted(targets):
        defs[t] = defstack[-1].get(t, none)

    inputs = meta['inputs']

    # Collect inputs
    inputs = flatten(input_nodes(v, sh) for (v, sh) in inputs)
    inputs = {unparse(i): (i, i) for i in inputs}

    # Plot dataflow
    input_vars = list(inputs.keys())
    plot_defs(entity + ".png", input_vars, defs, True)

    # Begin pipelining.
    print("* Pipelining")
    stage = 1
    vars = []
    while not all(fully_pipelined(n) for n in defs.values()):
        inputs, defs = pipeline_exprs(defs, inputs, stage)
        vars.append(list(inputs.values()))
        inputs = {unparse(lv) for (lv, _) in inputs.values()}
        stage += 1

    # Write schedule
    write_latex = False
    name_constants = False
    consts = {}

    def constfun(expr, node):
        if not name_constants:
            return node_const_str(node)
        if not expr in consts:
            lbl = 'c_%d' % len(consts)
            consts[expr] = lbl, node
        return consts[expr][0]

    data = table_data(vars, input_vars, constfun)
    if write_latex:
        print_latex_table(data)
    else:
        print_schedule_rich(entity, data)
    if name_constants:
        consts = [[v0, node_const_str(v1)] for v0, v1 in consts.values()]
        if write_latex:
            print_latex_table(consts)
        else:
            print_constants_rich(entity, consts)

    # Write VHDL
    out_path = Path(entity + ".vhdl")
    print('* Writing VHDL to "%s".' % out_path)
    with out_path.open("w") as f:
        vhdl = vars_to_vhdl(meta, vars, defs, entity)
        f.write(vhdl)


main()
