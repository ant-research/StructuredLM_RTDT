# coding=utf-8
# Copyright (c) 2021 Ant Group

import sys

LABEL_SEP = '@'

INDENT_STRING1 = '│   '
INDENT_STRING2 = '├──'
EMPTY_TOKEN = '___EMPTY___'


def print_tree(const_tree, indent=0, out=sys.stdout):
    for i in range(indent - 1):
        out.write(INDENT_STRING1)
    if indent > 0:
        out.write(INDENT_STRING2)
    out.write(const_tree.tag)
    if not isinstance(const_tree.children[0], ConstTree):
        out.write(f' {const_tree.children[0].string}\n')
    else:
        out.write('\n')
        for child in const_tree.children:
            print_tree(child, indent + 1, out)


def _make_tree(string, make_leaf_fn, make_internal_fn):
    tokens = string.replace('(', ' ( ').replace(')', ' ) ').split()

    index, stack = 0, []
    lexicons = []

    root = None
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if token == ')':
            if not stack:
                raise ConstTreeParserError('redundant ")" at token ' + str(index))
            node = stack.pop()
            if not stack:
                root = node
            else:
                stack[-1].children.append(node)
        elif token == '(':
            tag = tokens[index]
            index += 1
            stack.append(make_internal_fn(tag))
        else:
            if not stack:
                raise ConnectionError('??? at pos ' + str(index))
            new_token = []
            while token != ')':
                if not token != '(':
                    raise Exception('bracket error')
                new_token.append(token)
                token = tokens[index]
                index += 1

            # is lexicon
            leaf_node = make_leaf_fn('_'.join(new_token))
            lexicons.append(leaf_node)

            postag_node = stack.pop()
            postag_node.children.append(leaf_node)
            if not stack:
                root = postag_node
            else:
                stack[-1].children.append(postag_node)

    if not root or stack:
        raise ConstTreeParserError('missing ")".')

    return root, lexicons


class ConstTreeParserError(Exception):
    pass


class Lexicon:
    __slots__ = ('string', 'span', 'parent')

    def __init__(self, string, span=None):
        self.string = string
        self.span = span

    def __str__(self):
        return f'<Lexicon {self.string}>'

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.string == other.string

    def __hash__(self):
        return hash(self.string) + 2

    @property
    def tag(self):
        return self.string

    def to_string(self, quote_lexicon):
        if quote_lexicon:
            return f'"{self.string}"'
        return self.string


class ConstTree:
    __slots__ = ('children', 'tag', 'span', 'index', 'parent', 'attrs')

    ROOT_LABEL = 'ROOT'

    def __init__(self, tag, children=None, span=None):
        self.tag = tag
        self.children = children if children is not None else []
        self.span = span
        self.index = None

    def __str__(self):
        child_string = ' + '.join(child.tag for child in self.children)
        return f'{self.span} {self.tag} => {child_string}'

    def __repr__(self):
        return str(self)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.children[index]
        if isinstance(index, str):
            for child in self.children:
                if isinstance(child, ConstTree) and child.tag == index.upper():
                    return child
        raise KeyError

    def to_string(self, quote_lexicon=False):
        child_string = ' '.join(child.to_string(quote_lexicon) for child in self.children)
        return f'({self.tag} {child_string})'

    @staticmethod
    def from_string(string):
        """ Construct ConstTree from parenthesis representation.

        :param string: string of parenthesis representation
        :return: ConstTree root and all leaf Lexicons
        """
        tree, lexicons = _make_tree(string, Lexicon, ConstTree)
        for index, lexicon in enumerate(lexicons):
            lexicon.span = index, index + 1
        tree.populate_spans_internal()
        return tree, lexicons

    def traverse_postorder(self):
        for child in self.children:
            if isinstance(child, ConstTree):
                yield from child.traverse_postorder()

        yield self

    def traverse_postorder_with_lexicons(self):
        for child in self.children:
            if isinstance(child, ConstTree):
                yield from child.traverse_postorder_with_lexicons()
            else:
                yield child

        yield self

    def generate_preterminals(self):
        for child in self.children:
            if isinstance(child, ConstTree):
                yield from child.generate_preterminals()

        for child in self.children:
            if isinstance(child, Lexicon):
                yield self

    def generate_lexicons(self):
        for child in self.children:
            if isinstance(child, ConstTree):
                yield from child.generate_lexicons()

        for child in self.children:
            if isinstance(child, Lexicon):
                yield child

    def is_binary_tree(self):
        if isinstance(self.children[0], Lexicon):
            return True
        return len(self.children <= 2) and all(child.is_binary_tree() for child in self.children)

    def condensed_unary_chain(self, include_preterminal=True, remove_root=None):
        if self.tag == remove_root:
            assert len(self.children) == 1
            return self.children[0].condensed_unary_chain(include_preterminal=include_preterminal)

        if len(self.children) > 1:
            return ConstTree(self.tag,
                             children=list(child.condensed_unary_chain()
                                           for child in self.children),
                             span=self.span)

        if isinstance(self.children[0], Lexicon):
            return ConstTree((self.tag if include_preterminal else EMPTY_TOKEN),
                             children=list(self.children),
                             span=self.span)

        assert isinstance(self.children[0], ConstTree)
        node = self
        new_tag = self.tag
        while len(node.children) == 1 and isinstance(node.children[0], ConstTree):
            node = node.children[0]
            if include_preterminal or isinstance(node.children[0], ConstTree):
                new_tag += LABEL_SEP + node.tag

        if len(node.children) == 1:
            children = list(node.children)
        else:
            children = list(child.condensed_unary_chain() for child in node.children)

        return ConstTree(new_tag, children=children, span=self.span)

    def expanded_unary_chain(self, add_root=None):
        if isinstance(self.children[0], Lexicon):
            children = list(self.children)
        else:
            children = list(child.expanded_unary_chain() for child in self.children)

        tags = self.tag.split(LABEL_SEP)
        for tag in reversed(tags):
            children = [ConstTree(tag, children=children, span=self.span)]

        root = children[0]
        if add_root:
            root = ConstTree(add_root, children=[root])

        return root

    def calculate_span(self):
        self.span = self.children[0].span[0], self.children[-1].span[1]

    def populate_spans_internal(self):
        for child in self.children:
            if isinstance(child, ConstTree):
                child.populate_spans_internal()

        self.calculate_span()

    def add_postorder_index(self):
        for index, node in enumerate(self.traverse_postorder()):
            node.index = index

    def add_parents(self, parent=None):
        self.parent = parent
        for child in self.children:
            if isinstance(child, ConstTree):
                child.add_parents(self)

    def is_ancestor_of(self, other):
        other = other.parent
        while other is not None and other is not self:
            other = other.parent
        return other is not None

    def generate_path_to_root(self, include_self=False):
        node = self
        if not include_self:
            node = self.parent
        while node is not None:
            yield node
            node = node.parent

    def lowest_common_ancestor(self, other):
        path = list(other.generate_path_to_root())
        for node in self.generate_path_to_root():
            try:
                return path[path.index(node)]
            except ValueError:
                pass

    def remove_nodes(self, filter):
        _children = []
        for c in self.children:
            if isinstance(c, ConstTree):
                if filter(c):
                    pass
                else:
                    filtered_node = c.remove_nodes(filter)
                    _children.append(filtered_node)
            else:
                _children.append(c)
        return ConstTree(self.tag, _children)

class SpanTree:
    def __init__(self, st, ed, subtrees=[]) -> None:
        self.st = st
        self.ed = ed
        self.subtrees = subtrees
        self.cache_id = -1