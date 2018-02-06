import graphviz

YELLOW = "#fefecd"

def obj_edges(nodes, varnames=None):
    s = ""
    es = edges(nodes, varnames)
    for (p, label, q) in es:
        s += 'node%d:%s:c -> node%d [dir=both, tailclip=false, arrowtail=dot, penwidth="0.5", color="#444443", arrowsize=.4]\n' % (id(p), label, id(q))

    return s

def closure(p, varnames=None):
    """
    Find all nodes reachable from p and return a list of pointers to those reachable.
    There can't be duplicates even for cyclic graphs due to visited set. Chase ptrs
    from but don't include frame objects.
    """
    return closure_(p, varnames, set())


def closure_(p, varnames, visited):
    if p is None or isatom(p):
        return []
    if id(p) in visited:
        return []
    visited.add(id(p))
    result = []
    result.append(p)
    if hasattr(p, "__dict__"): # regular object like Tree or Node
        for k,q in p.__dict__.items():
            cl = closure_(q, varnames, visited)
            result.extend(cl)
    return result

def isatom(p):
    return isatom_(p) or type(p) == unicode # only python 2 distinguishes between str/unicode

def isatom_(p): return type(p) == int or type(p) == float or \
                       type(p) == str or \
                       p.__class__ == WrapAssoc or \
                       p.__class__ == Ellipsis
class WrapAssoc:
    def __init__(self,assoc):
        self.assoc = assoc

def gr_vtree_node(title, nodename, items, bgcolor=YELLOW, separator=None, leftfield='left', rightfield='right'):
    html = gr_vtree_html(title, items, bgcolor, separator)
    return '%s [margin="0.03", color="#444443", fontcolor="#444443", fontname="Helvetica", style=filled, fillcolor="%s", label=<%s>];\n' % (nodename,bgcolor,html)

def gr_vtree_html(title, items, bgcolor=YELLOW, separator=None, leftfield='left', rightfield='right'):
    header = '<table BORDER="0" CELLPADDING="0" CELLBORDER="1" CELLSPACING="0">\n'

    rows = []
    blankrow = '<tr><td colspan="3" cellpadding="1" border="0" bgcolor="%s"></td></tr>' % (bgcolor)

    if title is not None:
        title = '<tr><td cellspacing="0" colspan="3" cellpadding="0" bgcolor="%s" border="1" sides="b" align="center"><font color="#444443" FACE="Times-Italic" point-size="11">%s</font></td></tr>\n' % (bgcolor, title)
        rows.append(title)

    if len(items)>0:
        for label,key,value in items:
            if(label == 'leaf'):
                if(value ==0):
                    leaf_flag = False;
                else:
                    leaf_flag = True;
                continue
            font = "Helvetica"
            name = '<td port="%s_label" cellspacing="0" cellpadding="0" bgcolor="%s" border="1" sides="r" align="center"><font face="%s" color="#444443" point-size="11">AU</font></td>\n' % (label, bgcolor, font)
            sep = '<td cellspacing="0" cellpadding="0" border="0"></td>'
            if value is not None:
                if(leaf_flag):
                    if(value == 0):
                        v = 'False'
                    else:
                        v = 'True'
                else:
                    v = value
            else:
                v = "   "
            value = '<td port="%s" cellspacing="0" cellpadding="1" bgcolor="%s" border="0" align="left"><font color="#444443" point-size="11"> %s</font></td>\n' % (label, bgcolor, v)
            row = '<tr>' + name + sep + value + '</tr>\n'
            if(leaf_flag):
                row = '<tr>' + value + '</tr>\n'
            rows.append(row)
    else:
        rows.append('<tr><td cellspacing="0" cellpadding="0" border="0"><font point-size="9"> ... </font></td></tr>\n')

    if separator is not None:
        sep = '<td cellpadding="0" bgcolor="%s" border="0" valign="bottom"><font color="#444443" point-size="15">%s</font></td>' % (bgcolor,separator)
    else:
        sep = '<td cellspacing="0" cellpadding="0" bgcolor="%s" border="0"></td>' % bgcolor

    kidsep = """
    <tr><td colspan="3" cellpadding="0" border="1" sides="b" height="3"></td></tr>
    """ + blankrow

    kidnames = """
    <tr>
    <td cellspacing="0"  cellpadding="0" bgcolor="$bgcolor$" border="1" sides="r" align="center"><font color="#444443" point-size="6">0</font></td>
    %s
    <td cellspacing="0" cellpadding="1" bgcolor="$bgcolor$" border="0" align="center"><font color="#444443" point-size="6">1</font></td>
    </tr>
    """ % ( sep)

    kidptrs = """
    <tr>
    <td port="%s" cellspacing="0" cellpadding="0" bgcolor="$bgcolor$" border="1" sides="r"><font color="#444443" point-size="1"> </font></td>
    %s
    <td port="%s" cellspacing="0" cellpadding="0" bgcolor="$bgcolor$" border="0"><font color="#444443" point-size="1"> </font></td>
    </tr>
    """ % (leftfield, sep, rightfield)

    bottomsep = """
    <tr>
    <td cellspacing="0" cellpadding="0" bgcolor="$bgcolor$" border="1" sides="r"><font color="#444443" point-size="3"> </font></td>
    %s
    <td cellspacing="0" cellpadding="0" bgcolor="$bgcolor$" border="0"><font color="#444443" point-size="3"> </font></td>
    </tr>
    """ % sep

    kidsep = kidsep.replace('$bgcolor$', bgcolor)
    kidnames = kidnames.replace('$bgcolor$', bgcolor)
    kidptrs = kidptrs.replace('$bgcolor$', bgcolor)
    bottomsep = bottomsep.replace('$bgcolor$', bgcolor)

    tail = "</table>\n"
    if(leaf_flag):
        return header + blankrow.join(rows) +tail
    return header + blankrow.join(rows) + kidsep + kidnames + kidptrs + bottomsep + tail

def edges(reachable, varnames=None):
    """Return list of (p, port-in-p, q) for all p in reachable node list"""
    edges = []
    for p in reachable:
        edges.extend( node_edges(p, varnames) )
    return edges
def node_edges(p, varnames=None):
    """Return list of (p, fieldname-in-p, q) for all ptrs in p"""
    edges = []
    if hasattr(p, "__dict__"):
        for k, v in p.__dict__.items():
            if not isatom(v) and v is not None:
                edges.append((p, k, v))
    return edges
def treeviz(root, tree_name, leftfield='left', rightfield='right'):
    """
    Display a top-down visualization of a binary tree that has
    two fields pointing at the left and right subtrees. The
    type of each node is displayed, all fields, and then
    left/right pointers at the bottom.
    """
    if root is None:
        return

    s = """
    digraph G {
        nodesep=.1;
        ranksep=.3;
        rankdir=TD;
        node [penwidth="0.5", shape=box, width=.1, height=.1];
    """

    reachable = closure(root,varnames = ['value','leaf'])

    for p in reachable:
        nodename = "node%d" % id(p)
        fields = []
        # kids = [(k,k,getattr(p,k)) for k in [leftfield, rightfield]]
        for k, v in p.__dict__.items():
            if k==leftfield or k==rightfield:
                continue
            if isatom(v):
                fields.append((k, k, v))
            else:
                fields.append((k, k, None))
        s += '// %s TREE node with fields\n' % p.__class__.__name__
        s += gr_vtree_node(tree_name, nodename, fields, separator=None)
    s += obj_edges(reachable)

    s += "}\n"
    return graphviz.Source(s)


class Tree:
    def __init__(self, value, left=None, right=None, leaf = 0):
        self.value = value
        self.left = left
        self.right = right
        self.leaf = leaf



def build_tree(decision_tree):
    if(decision_tree.op == None):
        return Tree(int(decision_tree.label), leaf = 1)
    left = build_tree(decision_tree.kids[0])
    right = build_tree(decision_tree.kids[1])
    return Tree(int(decision_tree.op), left, right)
