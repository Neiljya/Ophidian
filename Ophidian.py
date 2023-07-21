# Thank you CodePulse

from string_with_arrows import *
import string
import os
import math


#######################################
# CONSTANTS
#######################################

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS


#######################################
# ERRORS
#######################################

class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details

    def as_string(self):
        result = f'{self.error_name}: {self.details}\n'
        result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt,self.pos_start,self.pos_end)
        return result

class ExpectedCharError(Error):
    def __init__(self,pos_start,pos_end,details):
        super().__init__(pos_start,pos_end,'Expected Character', details)

class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal Character', details)

class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details=''):
        super().__init__(pos_start, pos_end, 'Invalid Syntax', details)

class RTError(Error):
    def __init__(self, pos_start, pos_end, details,ctx):
        super().__init__(pos_start, pos_end, 'Runtime Error', details)
        self.ctx = ctx

    def as_string(self):
        result = self.generate_traceback()
        result += f'{self.error_name}: {self.details}\n'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result

    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        ctx = self.ctx

        while ctx:
            result = f' File: {pos.fn} | line: {str(pos.ln+1)} | in {ctx.display_name}\n' + result
            pos = ctx.parent_entry_pos
            ctx = ctx.parent

        return 'Traceback (most recent call last):\n' + result



#######################################
# POSITION
#######################################

class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1

        if current_char == '\n':
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)


#######################################
# TOKENS
#######################################

# Data Types
TT_INT = 'INT'
TT_BOOLEAN = 'BOOL'
TT_FLOAT = 'FLOAT'
TT_STRING = 'STRING'

# Variables
TT_IDENTIFIER = 'IDENTIFIER'
TT_KEYWORD = 'KEYWORD'

# Binary Operations
TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_MUL = 'MUL'
TT_DIV = 'DIV'
TT_MODULO = 'MODULO'
TT_POWER = 'POWER'
TT_EQUALS = 'EQUALS'
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'

# Lists/Arrays
TT_LBRACK = 'LBR'
TT_RBRACK = 'RBR'

# Functions:
TT_COMMA = 'COMMA'
TT_COLON = ':'

# Comparisons/Logical Operators

TT_EE = 'EE' # Double equal '=='
TT_NE = 'NE' # Not equal
TT_LT = 'LT' # Less than
TT_LTE = 'LTE' # Less than or equal
TT_GT = 'GT' # Greater than
TT_GTE = 'GTE' #Greater than or equal

# Misc.
TT_NEWLINE = 'NEWLINE'
TT_EOF = 'EOF' # end of file

KEYWORDS = [
    'VAR',
    'AND',
    'OR',
    'NOT',
    'IF',
    'DO',
    'ELSEIF',
    'ELSE',
    'for',
    'TO',
    'COUNT_BY',
    'while',
    'FROM',
    'FUNCTION',
    'CREATE',
    'GET',
    'END',
    'RETURN',
    'CONTINUE',
    'BREAK',
    'named'

]



class Token:
    def __init__(self, type_, value=None, pos_start = None, pos_end = None):
        self.type = type_
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()

        if pos_end:
            self.pos_end = pos_end.copy()

    def matches(self,type_,value):
        return self.type == type_ and self.value == value

    def __repr__(self):
        if self.value: return f'{self.type}:{self.value}'
        return f'{self.type}'


#######################################
# LEXER
#######################################

class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.current_char = None
        self.advance()

    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def make_tokens(self):
        tokens = []

        while self.current_char != None:
            if self.current_char in ' \t':
                self.advance()
            elif self.current_char == '#':
                self.ignore_comment()
            elif self.current_char in ';\n':
                tokens.append(Token(TT_NEWLINE, pos_start = self.pos))
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
            elif self.current_char =='"':
                tokens.append(self.make_string())
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.advance()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()
            elif self.current_char == '%':
                tokens.append(Token(TT_MODULO, pos_start=self.pos))
                self.advance()
            elif self.current_char == '^':
                tokens.append(Token(TT_POWER,pos_start=self.pos))
                self.advance()

            # Brackets
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == '[':
                tokens.append(Token(TT_LBRACK, pos_start=self.pos))
                self.advance()
            elif self.current_char == ']':
                tokens.append(Token(TT_RBRACK, pos_start=self.pos))
                self.advance()



            elif self.current_char == '!':
                # Checks if the character after '!' is '='
                tok, error = self.make_not_equals()
                if error: return [], error
                tokens.append(tok)

            elif self.current_char == '=':
                # Check for single '=' or double '=' to invoke different methods
                tokens.append(self.make_equals())

            elif self.current_char == ',':
                tokens.append(Token(TT_COMMA, pos_start = self.pos))
                self.advance()

            elif self.current_char == '<':
                tokens.append(self.make_less_than())
            elif self.current_char == '>':
                tokens.append(self.make_greater_than())
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

        tokens.append(Token(TT_EOF,pos_start=self.pos))
        return tokens, None

    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1: break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

    def make_boolean(self):
        bool_str = ""
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char.isalpha():
            bool_str += self.current_char
            self.advance()

        bool_str_lower = bool_str.lower()

        if str(bool_str_lower) == "true":
            return Token(TT_BOOLEAN, True, pos_start, self.pos)
        elif str(bool_str_lower) == "false":
            return Token(TT_BOOLEAN, False, pos_start, self.pos)
        else:
            raise Exception(f"Invalid boolean value: {bool_str}")


    def make_string(self):
        string = ''
        pos_start = self.pos.copy()
        esc_char = False
        self.advance()

        escape_chars = {
            'n': '\n',
            't': '\t'
        }

        # grab string
        while self.current_char != None and (self.current_char != '"' or esc_char):
           if esc_char:
               #adds the escape_char into the string but if not then adds the current_char

            #implementation of escape words instead of chars (idea):
               # store escaped string into another variable
               # perform check to see if matches escape keywords
               # then add escaped string to main string

               string += escape_chars.get(self.current_char, self.current_char)
           else:
                if self.current_char == '\\':
                    esc_char = True
                else:
                    string += self.current_char
           self.advance()
           esc_char = False

        self.advance()
        return Token(TT_STRING, string, pos_start, self.pos)




    def make_identifier(self):
        id_str = ''
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in LETTERS_DIGITS + '_': #allows underscores to variable name
            id_str += self.current_char
            self.advance()

        tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
        return Token(tok_type, id_str, pos_start,self.pos)

    def make_not_equals(self):
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            return Token(TT_NE, pos_start=pos_start, pos_end=self.pos), None

        self.advance()
        return None, ExpectedCharError(pos_start, self.pos, "'=' after '!'")


    def make_equals(self):
        tok_type = TT_EQUALS
        pos_start = self.pos.copy()
        self.advance()

        # registers token
        if self.current_char == '=':
            self.advance()
            tok_type = TT_EE

        # Returns the current token
        return Token(tok_type,pos_start=pos_start,pos_end=self.pos)

    def make_less_than(self):
        tok_type = TT_LT
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = TT_LTE

        return Token(tok_type,pos_start=pos_start,pos_end=self.pos)

    def make_greater_than(self):
        tok_type = TT_GT
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = TT_GTE

        return Token(tok_type,pos_start=pos_start,pos_end=self.pos)

    def ignore_comment(self):
        # advancing past the '#'
        self.advance()

        while self.current_char != '\n':
            self.advance()

        self.advance()


#######################################
# NODES
#######################################

class String_Node:
    def __init__(self,token):
        self.token = token

        self.pos_start = self.token.pos_start
        self.pos_end = self.token.pos_end

    def __repr__(self):
        return f'{self.token}'

class List_Node:
    def __init__(self, element_nodes, pos_start, pos_end):
        self.element_nodes = element_nodes

        # much better solution than grabbing the token pos but too lazy to change it
        self.pos_start = pos_start
        self.pos_end = pos_end

class Number_Node:
    def __init__(self,token):
        self.token = token

        self.pos_start = self.token.pos_start
        self.pos_end = self.token.pos_end


    def __repr__(self):
        return f'{self.token}'

class Boolean_Node:
    def __init__(self, token):
        self.token = token
        self.pos_start = self.token.pos_start
        self.pos_end = self.token.pos_end

class VarAccess_Node:
    def __init__(self,var_name_tok):
        self.var_name_tok = var_name_tok

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end

class VarAssign_Node:
    def __init__(self,var_name_tok,value_node):
        self.var_name_tok = var_name_tok
        self.value_node = value_node

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.value_node.pos_end


class BinaryOperation:
    def __init__(self,left_node,op_tok,right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.op_tok}, {self.right_node})'

class UnaryOpNode:
    def __init__(self,op_tok,node):
        self.op_tok = op_tok
        self.node = node

        self.pos_start = self.op_tok.pos_start
        self.pos_end = node.pos_end

    def __repr__(self):
        return f'{self.op_tok},{self.node}'

class If_Node:
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case

        # Gets the tuple that's stored in the 'cases' table and then access the first element of the tuple
        self.pos_start = self.cases[0][0].pos_start
        self.pos_end = (self.else_case or self.cases[len(self.cases) - 1])[0].pos_end

class For_Node:
    def __init__(self,var_name_tok, start_value_node, end_value_node, step_value_node, body_node, return_null):
        self.var_name_tok = var_name_tok
        self.start_value_node = start_value_node
        self.end_value_node = end_value_node
        self.step_value_node = step_value_node
        self.body_node = body_node
        self.return_null = return_null

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.body_node.pos_end

class While_Node:
    def __init__(self, condition_node, body_node, return_null):
        self.condition_node = condition_node
        self.body_node = body_node
        self.return_null = return_null

        self.pos_start = self.condition_node.pos_start
        self.pos_end = self.body_node.pos_end


class FuncDef_Node:
    def __init__(self, var_name_tok, arg_name_toks, body_node, auto_return):
        self.var_name_tok = var_name_tok
        self.arg_name_toks = arg_name_toks
        self.body_node = body_node
        self.auto_return = auto_return

        if self.var_name_tok:
            self.pos_start = self.var_name_tok.pos_start
        elif len(self.arg_name_toks) > 0:
            # Looks for the first argument
            self.pos_start = self.arg_name_toks[0].pos_start
        else:
            self.pos_start = self.body_node.pos_start

        self.pos_end = self.body_node.pos_end

class Call_Node:
    def __init__(self,node_to_call, arg_nodes):
        self.node_to_call = node_to_call
        self.arg_nodes = arg_nodes

        self.pos_start = self.node_to_call.pos_start

        if len(self.arg_nodes) > 0:
            self.pos_end = self.arg_nodes[len(self.arg_nodes)-1].pos_end
        else:
            self.pos_end = self.node_to_call.pos_end

class Return_Node:
    def __init__(self, node_to_return, pos_start, pos_end):
        self.node_to_return = node_to_return
        self.pos_start = pos_start
        self.pos_end = pos_end

class Continue_Node:
    def __init__(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end

class Break_Node:
    def __init__(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end



########## PARSE #################

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.last_count = 0

        # Keeps track of number of advancements made
        self.advance_count = 0
        self.rollback_count = 0

    def register(self, res):
        self.last_count = res.advance_count
        self.advance_count += res.advance_count
        if res.error: self.error = res.error
        return res.node

    def register_advancement(self):
        self.last_count = 1
        self.advance_count += 1

    def try_register(self,res):
        if res.error:
            self.rollback_count = res.advance_count
            return None
        return self.register(res)



    def success(self, node):
        self.node = node
        return self


    def fail(self, error):
        if not self.error or self.advance_count == 0:
            self.error = error
        return self



#######################################
# PARSER
#######################################

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.advance()

    def advance(self):
        self.tok_idx += 1
        self.upd_tok()
        return self.current_tok

    def rollback(self, amount=1):
        self.tok_idx -= amount
        self.upd_tok()
        return self.current_tok

    def upd_tok(self):
        if self.tok_idx >= 0 and self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]

    def parse(self):
        res = self.statements()
        if not res.error and self.current_tok.type != TT_EOF:
            return res.fail(InvalidSyntaxError(self.current_tok.pos_start,
                                               self.current_tok.pos_end,
                                               "<Expected '+', '-','*','/','%', or '^'>"))
        return res

    ##################################################

    def power(self):
        return self.bin_op(self.call, (TT_POWER, ), self.factor)


    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_PLUS, TT_MINUS):
            res.register_advancement()
            self.advance()
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok,factor))

        """
        elif tok.type in (TT_INT,TT_FLOAT):
            res.register_advancement()
self.advance()
            return res.success(NumberNode(tok))

        elif tok.type == TT_LPAREN:
            res.register_advancement()
self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_tok.type == TT_RPAREN:
                res.register_advancement()
self.advance()
                return res.success(expr)
            else:
                return res.fail(InvalidSyntaxError(
                    self.current_tok.pos_start,
                    self.current_tok.pos_end,
                    "<Expected ')'>"
                ))
        """

        return self.power()

    def call(self):
        res = ParseResult()
        expo = res.register(self.expo())
        if res.error: return res

        if self.current_tok.type == TT_LPAREN:
            res.register_advancement()
            self.advance()
            arg_nodes = []

            # Checks if there are no arguments
            if self.current_tok.type == TT_RPAREN:
                res.register_advancement()
                self.advance()
            else:
                arg_nodes.append((res.register(self.expr())))
                if res.error: return res.fail(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ')', 'var', 'IF', 'for', 'while', 'FUNCTION', int, float, identifier, '+', '-', '[', '(', or 'NOT'"
                ))

                while self.current_tok.type == TT_COMMA:
                    res.register_advancement()
                    self.advance()

                    arg_nodes.append(res.register(self.expr()))
                    if res.error: return res

                if self.current_tok.type != TT_RPAREN:
                    return res.fail(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        f"Expected ',' or ')'"
                    ))

                res.register_advancement()
                self.advance()
            return res.success(Call_Node(expo, arg_nodes))
        return res.success(expo)


    def expo(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_INT, TT_FLOAT):
            res.register_advancement()
            self.advance()
            return res.success(Number_Node(tok))

        elif tok.type == TT_STRING:
            res.register_advancement()
            self.advance()
            return res.success(String_Node(tok))

        # Checks for identifier
        elif tok.type == TT_IDENTIFIER:
            res.register_advancement()
            self.advance()

            # If identifier then access the variable data
            return res.success(VarAccess_Node(tok))


        elif tok.type == TT_LPAREN:
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_tok.type == TT_RPAREN:
                res.register_advancement()
                self.advance()
                return res.success(expr)
            else:
                return res.fail(InvalidSyntaxError(
                    self.current_tok.pos_start,
                    self.current_tok.pos_end,
                    "<Expected ')'>"
                ))

        # list expressions
        elif tok.type == TT_LBRACK:
            list_expr = res.register(self.list_expr())
            if res.error: return res
            return res.success(list_expr)


        # Handles "IF"
        elif tok.matches(TT_KEYWORD, "IF"):
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(if_expr)

        elif tok.matches(TT_KEYWORD, "for"):
            for_expr = res.register(self.for_expr())
            if res.error: return res
            return res.success(for_expr)

        elif tok.matches(TT_KEYWORD, "while"):
            while_expr = res.register(self.while_expr())
            if res.error: return res
            return res.success(while_expr)

        elif tok.matches(TT_KEYWORD, "CREATE"):
            self.advance()
            if self.current_tok.matches(TT_KEYWORD, "FUNCTION"):
                func_def = res.register(self.func_def())
                if res.error: return res
                return res.success(func_def)
            else:
                return res.fail(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected 'FUNCTION'"
                ))

        return res.fail(InvalidSyntaxError(tok.pos_start,
                                            tok.pos_end,
                                            "<Expected int, float, identifier, '+', '-', '(', '[', 'IF', 'for', 'while', 'FUNCTION'>"))


            





    def term(self):
        return self.bin_op(self.factor, (TT_MUL, TT_DIV, TT_MODULO))

    def arith_expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    def list_expr(self):
        res = ParseResult()
        element_nodes = []
        pos_start = self.current_tok.pos_start.copy()

        if self.current_tok.type != TT_LBRACK:
            return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '['"
            ))

        res.register_advancement()
        self.advance()


        # checks for empty list
        if self.current_tok.type == TT_RBRACK:
            res.register_advancement()
            self.advance()
        else:
            element_nodes.append(res.register(self.expr()))
            if res.error: return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected ']', 'var', 'IF', 'for', 'while', 'FUNCTION', int, float, identifier, '+', '-', '[', '(', or 'NOT'"
            ))

            while self.current_tok.type == TT_COMMA:
                res.register_advancement()
                self.advance()

                element_nodes.append(res.register(self.expr()))
                if res.error: return res

            if self.current_tok.type != TT_RBRACK:
                return res.fail(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end.copy(),
                    f"Expected ',' or ']'"
                ))
            res.register_advancement()
            self.advance()

        return res.success(List_Node(
            element_nodes, pos_start, self.current_tok.pos_end.copy()
        ))



    def comp_expr(self):
        res = ParseResult()
        if self.current_tok.matches(TT_KEYWORD, "NOT"):
            op_tok = self.current_tok
            res.register_advancement()
            self.advance()

            node = res.register(self.comp_expr())
            if res.error: return res
            return res.success(UnaryOpNode(op_tok, node))

        node = res.register(self.bin_op(self.arith_expr, (TT_EE, TT_NE, TT_LT, TT_GT, TT_LTE, TT_GTE)))

        if res.error:
            return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'var', 'IF', 'for', 'while', 'FUN', int, float, identifier, '+', '-', '[', '(', 'NOT'"


            ))
        return res.success(node)

    # FOR <var> = <start_value> TO <end_value> DO <command>
    def for_expr(self):
        res = ParseResult()

        # Look for 'FOR'
        if not self.current_tok.matches(TT_KEYWORD, 'for'):
            return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'for'"
            ))

        res.register_advancement()
        self.advance()

        # Check for identifier for a variable name
        if self.current_tok.type != TT_IDENTIFIER:
            return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected identifier"
            ))


        # FOR '(var)'
        var_name = self.current_tok
        res.register_advancement()
        self.advance()

        # Checks for '=' though might change to a keyword later
        if not self.current_tok.matches(TT_KEYWORD, 'FROM'):
            return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'FROM'"
            ))

        res.register_advancement()
        self.advance()

        # The next position should be the first value which will be the starting value
        start_value = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(TT_KEYWORD, 'TO'):
            return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'TO'"
            ))

        res.register_advancement()
        self.advance()

        end_value = res.register(self.expr())
        if res.error: return res

        if self.current_tok.matches(TT_KEYWORD, 'COUNT_BY'):
            res.register_advancement()
            self.advance()

            step_value = res.register(self.expr())
            if res.error: return res
        else:
            step_value = None



        if not self.current_tok.matches(TT_KEYWORD, 'DO'):
            return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'DO'"
            ))

        res.register_advancement()
        self.advance()

        # Check for newline
        if self.current_tok.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

            body = res.register(self.statements())
            if res.error: return res

            if not self.current_tok.matches(TT_KEYWORD, 'END'):
                return res.fail(InvalidSyntaxError(
                    self.current_tok.pos_start,self.current_tok.pos_end,
                    f"Expected 'END'"
                ))

            res.register_advancement()
            self.advance()

            return res.success(For_Node(var_name, start_value, end_value, step_value, body, True))

        body = res.register(self.statement())
        if res.error: return res

        return res.success(For_Node(var_name, start_value, end_value, step_value, body, False))

    def while_expr(self):
        res = ParseResult()

        if not self.current_tok.matches(TT_KEYWORD, 'while'):
            return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'while'"
            ))

        res.register_advancement()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(TT_KEYWORD, 'DO'):
            return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'DO'"
            ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

            body = res.register(self.statements())
            if res.error: return res

            if not self.current_tok.matches(TT_KEYWORD, 'END'):
                return res.fail(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected 'END'"
                ))

            res.register_advancement()
            self.advance()

            return res.success(While_Node(condition, body, True))
        ''
        body = res.register(self.statement())
        if res.error: return res

        return res.success(While_Node(condition,body, False))

    def if_expr(self):
        res = ParseResult()
        all_cases = res.register(self.if_cases('IF'))
        if res.error: return res
        cases, else_case = all_cases

        return res.success(If_Node(cases, else_case))

    def if_expr_b(self):
        return self.if_cases('ELSEIF')

    def if_expr_c(self):
        res = ParseResult()
        else_case = None

        if self.current_tok.matches(TT_KEYWORD, 'ELSE'):
            res.register_advancement()
            self.advance()

            if self.current_tok.type == TT_NEWLINE:
                res.register_advancement()
                self.advance()

                statements = res.register(self.statements())
                if res.error: return res
                else_case = (statements, True)

                if self.current_tok.matches(TT_KEYWORD, 'END'):
                    res.register_advancement()
                    self.advance()
                else:
                    return res.fail(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected 'END'"
                    ))
            else:
                expr = res.register(self.statement())
                if res.error: return res
                else_case = (expr, False)
        return res.success(else_case)

    def choose_if(self):
        res = ParseResult()
        cases, else_case = [], None

        if self.current_tok.matches(TT_KEYWORD, 'ELSEIF'):
            all_cases = res.register(self.if_expr_b())
            if res.error: return res
            cases, else_case = all_cases
        else:
            else_case = res.register(self.if_expr_c())
            if res.error: return res
        return res.success((cases, else_case))

    def if_cases(self, keyword):
        res = ParseResult()
        cases = []
        else_case = None

        # Checking for syntax error
        if not self.current_tok.matches(TT_KEYWORD, keyword):
            return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start,self.current_tok.pos_end,
                f"Expected '{keyword}'"
            ))

        # Continue if no error
        res.register_advancement()
        self.advance()

        # Check for expression that follows 'if'
        condition = res.register(self.expr())
        if res.error: return res

        # Checking for error again
        if not self.current_tok.matches(TT_KEYWORD, 'DO'):
            return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'DO'"
            ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

            statements = res.register(self.statements())
            if res.error:
                return res
            cases.append((condition,statements,True))

            if self.current_tok.matches(TT_KEYWORD, 'END'):

                res.register_advancement()
                self.advance()
            else:
                all_cases = res.register(self.choose_if())

                if res.error: return res

                new_cases, else_case = all_cases
                cases.extend(new_cases)
        else:
            expr = res.register(self.statement())
            if res.error: return res

            cases.append((condition, expr, False))

            all_cases = res.register(self.choose_if())
            if res.error: return res
            new_cases, else_case = all_cases
            cases.extend(new_cases)

        return res.success((cases, else_case))




    def func_def(self):
        res = ParseResult()

        """
        if not self.current_tok.matches(TT_KEYWORD, 'CREATE'):
            return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'CREATE'"
            ))

        res.register_advancement()
        self.advance()
        """



        if not self.current_tok.matches(TT_KEYWORD, 'FUNCTION'):
            return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'FUNCTION'"
            ))

        res.register_advancement()
        self.advance()


        if self.current_tok.type == TT_IDENTIFIER:
            var_name_tok = self.current_tok
            res.register_advancement()
            self.advance()
            if self.current_tok.type != TT_LPAREN:
                return res.fail(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected '('"
                ))
        else:
            var_name_tok = None
            if self.current_tok.type != TT_LPAREN:
                return res.fail(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected identifier or '('"
                ))

        res.register_advancement()
        self.advance()
        arg_name_toks = []

        if self.current_tok.type == TT_IDENTIFIER:
            arg_name_toks.append(self.current_tok)
            res.register_advancement()
            self.advance()

            while self.current_tok.type == TT_COMMA:
                res.register_advancement()
                self.advance()

                if self.current_tok.type != TT_IDENTIFIER:
                    return res.fail(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        f"Expected identifier"
                    ))

                arg_name_toks.append(self.current_tok)
                res.register_advancement()
                self.advance()

            if self.current_tok.type != TT_RPAREN:
                return res.fail(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected ',' or ')'"
                ))
        else:
            if self.current_tok.type != TT_RPAREN:
                return res.fail(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected identifier or ')'"
                ))

        res.register_advancement()
        self.advance()

        if self.current_tok.matches(TT_KEYWORD, 'DO'):
            res.register_advancement()
            self.advance()

            body = res.register(self.expr())
            if res.error: return res

            return res.success(FuncDef_Node(
                var_name_tok, arg_name_toks, body, True
            ))

        if self.current_tok.type != TT_NEWLINE:
            return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'DO' or NEWLINE"
            ))

        res.register_advancement()
        self.advance()

        body = res.register(self.statements())
        if res.error: return res

        if not self.current_tok.matches(TT_KEYWORD, 'END'):
            return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'END'"
            ))

        res.register_advancement()
        self.advance()


        return res.success(FuncDef_Node(
            var_name_tok, arg_name_toks, body,
            False
        ))




    def statements(self):
        res = ParseResult()
        statements = []
        pos_start = self.current_tok.pos_start.copy()


        while self.current_tok.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()
        statement = res.register(self.statement())
        if res.error: return res
        statements.append(statement)

        more_statements = True

        while True:
            nl_count = 0
            while self.current_tok.type == TT_NEWLINE:
                res.register_advancement()
                self.advance()
                nl_count += 1
            if nl_count == 0:
                more_statements = False

            if not more_statements:
                break
            statement = res.try_register(self.statement())
            if not statement:
                self.rollback(res.rollback_count)
                more_statements = False
                continue
            statements.append(statement)

        return res.success(List_Node(
            statements, pos_start, self.current_tok.pos_end.copy()
        ))

    def statement(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()

        if self.current_tok.matches(TT_KEYWORD, 'RETURN'):
            res.register_advancement()
            self.advance()

            expr = res.try_register(self.expr())
            if not expr:
                self.rollback(res.rollback_count)
            return res.success(Return_Node(expr, pos_start, self.current_tok.pos_start.copy()))

        if self.current_tok.matches(TT_KEYWORD, 'CONTINUE'):
            res.register_advancement()
            self.advance()
            return res.success(Continue_Node(pos_start, self.current_tok.pos_start.copy()))

        if self.current_tok.matches(TT_KEYWORD, 'BREAK'):
            res.register_advancement()
            self.advance()
            return res.success(Break_Node(pos_start, self.current_tok.pos_start.copy()))

        expr = res.register(self.expr())
        if res.error:
            return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'RETURN','CREATE', 'CONTINUE', 'BREAK_LOOP', 'var', 'IF', 'for', 'while', 'FUN', int, float, identifier, '+', '-', '(', '[' or 'NOT'"

            ))
        return res.success(expr)

    def expr(self):
        res = ParseResult()

        #Checking if assignment begins with 'VAR'
        if self.current_tok.matches(TT_KEYWORD, 'VAR'):
            res.register_advancement()
            self.advance()

            # Error handling
            if self.current_tok.type != TT_IDENTIFIER:
                return res.fail(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected valid identifier"
                ))



            var_name = self.current_tok
            res.register_advancement()
            self.advance()


            if self.current_tok.type != TT_EQUALS:
                return res.fail(InvalidSyntaxError(
                    self.current_tok.pos_start,self.current_tok.pos_end,
                    "Expected '='"
                ))

            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssign_Node(var_name,expr))

        node = res.register(self.bin_op(self.comp_expr, ((TT_KEYWORD, "AND"), (TT_KEYWORD, "OR"), (TT_KEYWORD,"GET"))))

        if res.error and self.current_tok.matches(TT_KEYWORD, 'FUNCTION'):
            return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'crate' before 'FUNCTION'"
            ))
        elif res.error:
            return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'var', 'IF', 'for', 'while', 'FUNCTION', int, float, identifier, '+', '-', '(', '[' or 'NOT'"
            ))


        return res.success(node)



    def bin_op(self,func_a,ops,func_b=None):
        if func_b == None:
            func_b = func_a
        res = ParseResult()
        left = res.register(func_a()) #Takes the parsed result from the function to extract node
        if res.error: return res

        # Used for debugging purposes
        #print(self.current_tok)
        while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
            op_tok = self.current_tok
            res.register_advancement()
            self.advance()
            right = res.register(func_b())
            if res.error: return res
            left = BinaryOperation(left,op_tok,right)

        return res.success(left)




#######################################
# RUNTIME RESULT
#######################################

class RTResult:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = None
        self.error = None
        self.func_return_value = None
        self.cont_loop = False
        self.break_loop = False


    def register(self,res):
        self.error = res.error
        self.func_return_value = res.func_return_value
        self.cont_loop = res.cont_loop
        self.break_loop = res.break_loop
        return res.value

    def success(self, value):
        self.reset()
        self.value = value
        return self

    def ret_success(self, value):
        self.reset()
        self.func_return_value = value
        return self

    def cont_success(self):
        self.reset()
        self.cont_loop = True
        return self

    def break_success(self):
        self.reset()
        self.break_loop = True
        return self


    def fail(self,error):
        self.reset()
        self.error = error
        return self

    def should_return(self):
        return (
            self.error or
            self.func_return_value or
            self.cont_loop or
            self.break_loop
        )

########## PARSE #################

class Value:
    def __init__(self):
        self.set_pos()
        self.set_context()

    def set_pos(self,pos_start=None,pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context = None):
        self.ctx = context
        return self

    def add_to(self, other):
        return None, self.illegal_op(other)

    def sub_by(self, other):
        return None, self.illegal_op(other)

    def mul_by(self,other):
        return None, self.illegal_op(other)

    def div_by(self,other):
        return None, self.illegal_op(other)

    def mod_by(self,other):
        return None, self.illegal_op(other)

    def raise_to(self,other):
        return None, self.illegal_op(other)

    def compare_eq(self, other):
        return None, self.illegal_op(other)

    def compare_ne(self,other):
        return None, self.illegal_op(other)

    def compare_lt(self,other):
        return None, self.illegal_op(other)

    def compare_gt(self,other):
        return None, self.illegal_op(other)

    def compare_lte(self,other):
        return None, self.illegal_op(other)

    def compare_gte(self,other):
        return None, self.illegal_op(other)

    def and_by(self,other):
        return None, self.illegal_op(other)

    def or_by(self,other):
        return None, self.illegal_op(other)

    def not_by(self,other):
        return None, self.illegal_op(other)

    def get_by(self,other):
        return None, self.illegal_op(other)

    def execute(self, args):
        return RTResult().fail(self.illegal_op())

    def copy(self):
        raise Exception('No copy method defined')

    def is_true(self):
        return False

    def illegal_op(self, other=None):
        if not other: other = self
        return RTError(
            self.pos_start, other.pos_end,
            "Illegal Operation",
            self.ctx
        )



class Number(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value
    # Basic Arithmetic
    def add_to(self,other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.ctx), None
        else:
            return None, Value.illegal_op(self, other)
    def sub_by(self,other):
        if isinstance(other,Number):
            return Number(self.value - other.value).set_context(self.ctx), None
        else:
            return None, Value.illegal_op(self, other)

    def mul_by(self,other):
        if isinstance(other,Number):
            return Number(self.value * other.value).set_context(self.ctx), None
        else:
            return None, Value.illegal_op(self, other)

    def mod_by(self,other):
        if isinstance(other, Number):
            return Number(self.value % other.value).set_context(self.ctx), None
        else:
            return None, Value.illegal_op(self, other)

    def raise_to(self,other):
        if isinstance(other,Number):
            return Number(self.value ** other.value).set_context(self.ctx), None
        else:
            return None, Value.illegal_op(self, other)

    def compare_eq(self, other):
        if isinstance(other,Number):
            return Number(bool(self.value == other.value)).set_context(self.ctx), None
        else:
            return None, Value.illegal_op(self, other)
    def compare_ne(self, other):
        if isinstance(other, Number):
            return Number(bool(self.value != other.value)).set_context(self.ctx), None
        else:
            return None, Value.illegal_op(self, other)
    def compare_lt(self, other):
        if isinstance(other, Number):
            return Number(bool(self.value < other.value)).set_context(self.ctx), None
        else:
            return None, Value.illegal_op(self, other)
    def compare_gt(self, other):
        if isinstance(other, Number):
            return Number(bool(self.value > other.value)).set_context(self.ctx), None
        else:
            return None, Value.illegal_op(self, other)
    def compare_lte(self, other):
        if isinstance(other, Number):
            return Number(bool(self.value <= other.value)).set_context(self.ctx), None
        else:
            return None, Value.illegal_op(self, other)
    def compare_gte(self, other):
        if isinstance(other, Number):
            return Number(bool(self.value >= other.value)).set_context(self.ctx), None
        else:
            return None, Value.illegal_op(self, other)
    def and_by(self,other):
        if isinstance(other, Number):
            return Number(bool(self.value and other.value)).set_context(self.ctx), None
        else:
            return None, Value.illegal_op(self, other)
    def or_by(self,other):
        if isinstance(other,Number):
            return Number(bool(self.value or other.value)).set_context(self.ctx), None
        else:
            return None, Value.illegal_op(self, other)
    def not_by(self,other):
        if isinstance(other, Number):
            return Number(bool(-1 if self.value == 0 else 0)).set_context(self.ctx), None


    def div_by(self,other):
        if isinstance(other,Number):

            # Division by zero error
            if other.value == 0:
                return None, RTError(
                    other.pos_start, other.pos_end, "Can't divide by 0",
                    self.ctx
                )
            return Number(self.value / other.value).set_context(self.ctx), None
        else:
            return None, Value.illegal_op(self, other)

    def is_true(self):
        return self.value != False


    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start,self.pos_end)
        copy.set_context(self.ctx)
        return copy

    def __repr__(self):
        # shows the result
        return str(self.value)





class Boolean(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def copy(self):
        copy = Boolean(self.value)
        copy.set_pos(self.pos_start,self.pos_end)
        copy.set_context(self.ctx)
        return copy

    def __repr__(self):
        return str(self.value)

Number.null = Number(0)
Number.false = Boolean(False)
Number.true = Boolean(True)
Number.math_PI = Number(math.pi)

class Main_Function(Value):
    def __init__(self, name):
        super().__init__()
        self.name = name or "<no_name>"

    def create_new_context(self):
        new_ctx = Context(self.name,self.ctx,self.pos_start)
        new_ctx.symbols = Symbols(new_ctx.parent.symbols)
        return new_ctx

    # Checks for correct number of args
    def check_args(self,arg_names,args):
        res = RTResult()

        if len(args) > len(arg_names):
            return res.fail(RTError(
                self.pos_start, self.pos_end,
                f"{len(args) - len(arg_names)} too many args passed in '{self.name}",
                self.ctx
            ))

        if len(args) < len(arg_names):
            return res.fail(RTError(
                self.pos_start, self.pos_end,
                f"{len(arg_names) - len(args)} not enough args passed in '{self.name}'",
                self.ctx
            ))
        return res.success(None)

    def append_args(self, arg_names, args, ex_ctx):
        for i in range(len(args)):
            arg_name = arg_names[i]
            arg_value = args[i]
            arg_value.set_context(ex_ctx)

            ex_ctx.symbols.set(arg_name, arg_value)

    def execute_args(self, arg_names, args, ex_ctx):
        res = RTResult()
        res.register(self.check_args(arg_names, args))
        if res.should_return(): return res
        self.append_args(arg_names, args, ex_ctx)
        return res.success(None)



class Function(Main_Function):
    def __init__(self, name, body_node, arg_names, auto_return):
        super().__init__(name)
        self.body_node = body_node
        self.arg_names = arg_names
        self.auto_return = auto_return

    def execute(self, args):
        res = RTResult()
        interpreter = Interpreter()

        ex_ctx = self.create_new_context()

        # Check if number of args passed matches number of parameters
        res.register(self.execute_args(self.arg_names,args, ex_ctx))
        if res.should_return(): return res

        value = res.register(interpreter.visit(self.body_node, ex_ctx))
        if res.should_return() and res.func_return_value == None: return res

        ret_value = (value if self.auto_return else None) or res.func_return_value or Number.null
        return res.success(ret_value)

    def copy(self):
        copy = Function(self.name, self.body_node, self.arg_names, self.auto_return)
        copy.set_context(self.ctx)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy

    def __repr__(self):
        return f"<function {self.name}>"


class Built_In_Functions(Main_Function):
    def __init__(self, name):
        super().__init__(name)

    def execute(self, args):
        res = RTResult()
        ex_ctx = self.create_new_context()

        method_name = f'execute_{self.name}'
        method = getattr(self, method_name, self.no_visit_method)

        res.register(self.execute_args(method.arg_names, args, ex_ctx))
        if res.should_return(): return res

        return_value = res.register(method(ex_ctx))
        if res.should_return(): return res
        return res.success(return_value)

    def copy(self):
        copy = Built_In_Functions(self.name)
        copy.set_context(self.ctx)
        copy.set_pos(self.pos_start,self.pos_end)
        return copy

    def __repr__(self):
        return f"<built-in-function {self.name}>"

    def no_visit_method(self,node,ctx):
        raise Exception(f'No execute_{self.name} method defined')

    ######################## BUILT IN FUNCTIONS ##########################

    def execute_print(self, ex_ctx):
        print(str(ex_ctx.symbols.get('val')))
        return RTResult().success(Number.null)
    execute_print.arg_names = ["val"]



    def execute_print_return(self, ex_ctx):
        return RTResult().success(String(str(ex_ctx.symbols.get('val'))))
    execute_print_return.arg_names = ["val"]

    def execute_input(self,ex_ctx):
        text = input()
        return RTResult().success(String(text))
    execute_input.arg_names = []

    def execute_input_int(self,ex_ctx):
        while True:
            text = input()
            try:
                num = int(text)
                break
            except ValueError:
                print(f"'{text}' must be an integer value")
        return RTResult().success(Number(num))
    execute_input_int.arg_names = []

    def execute_clear(self, ex_ctx):
        # Check if platform is windows
        os.system('cls' if os.name == 'nt' else 'clear')
        return RTResult().success(Number.null)
    execute_clear.arg_names = []

    def execute_is_num(self, ex_ctx):
        is_num = isinstance(ex_ctx.symbols.get("val"), Number)
        return RTResult().success(Number.true if is_num else Number.false)
    execute_is_num.arg_names = ['val']

    def execute_is_bool(self, ex_ctx):
        is_bool = isinstance(ex_ctx.symbols.get("val"), Boolean)
        return RTResult().success(Number.true if is_bool else Number.false)
    execute_is_bool.arg_names = ['val']

    def execute_is_str(self, ex_ctx):
        is_str = isinstance(ex_ctx.symbols.get("val"), String)
        return RTResult().success(Number.true if is_str else Number.false)
    execute_is_str.arg_names = ['val']

    def execute_to_string(self, ex_ctx):
        string = String(str(ex_ctx.symbols.get('val')))
        return RTResult().success(string)
    execute_to_string.arg_names = ['val']

    def execute_to_num(self, ex_ctx):
        string = str(ex_ctx.symbols.get('val'))
        try:
            try:
                num = Number(int(string))
            except:
                num = Number(float(string))
            return RTResult().success(num)
        except:
            return RTResult().fail(RTError(
                self.pos_start, self.pos_end, "String argument must be a number", ex_ctx
            ))
    execute_to_num.arg_names = ['val']

    def execute_index_of(self, ex_ctx):
        val = ex_ctx.symbols.get('val')
        idx = ex_ctx.symbols.get('idx')

        if isinstance(val, List) or isinstance(val, String):
            if isinstance(idx, Number) and type(idx.value) == int:
                if idx.value < len(str(val)) and idx.value >= 0:
                    if isinstance(val, String):
                        return RTResult().success(String(str(val)[idx.value]))
                    else:
                        return RTResult().success(val[idx.value])
                else:
                    return RTResult().fail(RTError(
                        self.pos_start, self.pos_end, "Index is out of range", ex_ctx
                    ))
            else:
                return RTResult().fail(RTError(
                    self.pos_start, self.pos_end, "Second argument must be an integer", ex_ctx
                ))
        else:
            return RTResult().fail(RTError(
                self.pos_start, self.pos_end, "First argument must be a list or string", ex_ctx
            ))
    execute_index_of.arg_names = ['val', 'idx']

    def execute_is_list(self, ex_ctx):
        is_list = isinstance(ex_ctx.symbols.get("val"), List)
        return RTResult().success(Number.true if is_list else Number.false)
    execute_is_list.arg_names = ['val']


    def execute_is_func(self, ex_ctx):
        is_func = isinstance(ex_ctx.symbols.get("value"), Main_Function)
        return RTResult().success(Number.true if is_func else Number.false)
    execute_is_func.arg_names = ['val']

    def execute_append(self, ex_ctx):
        list_ = ex_ctx.symbols.get('list')
        value = ex_ctx.symbols.get('val')

        if not isinstance(list_, List):
            return RTResult().fail(RTError(
                self.pos_start, self.pos_end, "First argument must be list object", ex_ctx
            ))

        list_.elements.append(value)
        return RTResult().success(Number.null)
    execute_append.arg_names = ['list', 'val']

    def execute_pop(self, ex_ctx):
        list_ = ex_ctx.symbols.get('list')
        idx = ex_ctx.symbols.get('idx')

        if not isinstance(list_, List):
            return RTResult().fail(RTError(
                self.pos_start, self.pos_end, "First argument must be a list object", ex_ctx
            ))

        if not isinstance(idx, Number):
            return RTResult().fail(RTError(
                self.pos_start, self.pos_end, "Second argument must be a number", ex_ctx
            ))
        try:
            element = list_.elements.pop(idx.value)
        except:
            return RTResult().fail(RTError(
                self.pos_start, self.pos_end, "Could not remove element because index value is out of list range"

            ))

    def execute_combine(self,ex_ctx):
        list_A = ex_ctx.symbols.get("list_A")
        list_B = ex_ctx.symbols.get("list_B")

        if not isinstance(list_A, List):
            return RTResult().fail(RTError(
                self.pos_start, self.pos_end, "First argument must be a list", ex_ctx
            ))

        if not isinstance(list_B, List):
            return RTResult().fail(RTError(
                self.pos_start, self.pos_end, "Second argument must be a list", ex_ctx
            ))

        list_A.elements.extend(list_B.elements)
        return RTResult().success(Number.null)
    execute_combine.arg_names = ['list_A', 'list_B']

    def execute_len(self, ex_ctx):
        arg_ = ex_ctx.symbols.get('arg')

        if isinstance(arg_, List):
            return RTResult().success(Number(len(arg_.elements)))
        elif isinstance(arg_, String):
            str_list = []
            for i in range(arg_.length()):
                str_list.append(arg_.value[i])
            return RTResult().success(Number(len(str_list)))
        else:
            return RTResult().fail(RTError(
                self.pos_start, self.pos_end,
                "Argument must be a string or a list", ex_ctx
            ))



    execute_len.arg_names = ["arg"]

    def execute_run(self, ex_ctx):
        fn = ex_ctx.symbols.get('fn')

        if not isinstance(fn, String):
            return RTResult().fail(RTError(
                self.pos_start, self.pos_end,
                "Argument must be a string",
                ex_ctx
            ))

        fn = fn.value

        try:
            with open(fn, "r") as f:
                script = f.read()
        except Exception as e:
            return RTResult().fail(RTError(
                self.pos_start, self.pos_end,
                f"Could not load script \"{fn}\"\n"+str(e), ex_ctx
            ))

        _, error = run(fn, script)

        if error:
            return RTResult().fail(RTError(
                self.pos_start, self.pos_end,
                f"Could not finish executing script \"{fn}\"\n"+ error.as_string(), ex_ctx
            ))

        return RTResult().success(Number.null)
    execute_run.arg_names = ['fn']

Built_In_Functions.print = Built_In_Functions('print')
Built_In_Functions.print_ret = Built_In_Functions('print_ret')
Built_In_Functions.input = Built_In_Functions('input')
Built_In_Functions.input_int = Built_In_Functions('input_int')
Built_In_Functions.clear = Built_In_Functions('clear')
Built_In_Functions.is_num = Built_In_Functions('is_num')
Built_In_Functions.is_str = Built_In_Functions('is_str')
Built_In_Functions.is_list = Built_In_Functions('is_list')
Built_In_Functions.is_func = Built_In_Functions('is_func')
Built_In_Functions.append = Built_In_Functions('append')
Built_In_Functions.pop = Built_In_Functions('pop')
Built_In_Functions.combine = Built_In_Functions('combine')
Built_In_Functions.is_bool = Built_In_Functions('is_bool')
Built_In_Functions.len = Built_In_Functions('len')
Built_In_Functions.run = Built_In_Functions('run')
Built_In_Functions.to_str = Built_In_Functions('to_string')
Built_In_Functions.index_of = Built_In_Functions('index_of')
Built_In_Functions.to_num = Built_In_Functions('to_num')



class String(Value):
    def __init__(self,value):
        super().__init__()
        self.value = value

    def length(self):
        return len(self.value)

    def add_to(self,other):
        if isinstance(other, String):
            return String(self.value + other.value).set_context(self.ctx), None
        else:
            return None, Value.illegal_op(self,other)

    def mul_by(self,other):
        if isinstance(other, Number):
            return String(self.value * other.value).set_context(self.ctx), None
        else:
            return None, Value.illegal_op(self, other)

    def is_true(self):
        return len(self.value) > 0

    def copy(self):
        copy = String(self.value)
        copy.set_pos(self.pos_start,self.pos_end)
        copy.set_context(self.ctx)
        return copy

    def __str__(self):
        return self.value

    def __repr__(self):
        # returns the string
        return f'"{self.value}"'



class List(Value):
    def __init__(self, elements):
        super().__init__()
        self.elements = elements

    def add_to(self, other):
        new_list = self.copy()
        new_list.elements.append(other)
        return new_list, None

    def sub_by(self, other):
        if isinstance(other, Number):
            new_list = self.copy()
            try:
                new_list.elements.pop(other.value)
                return new_list, None
            except:
                return None, RTError(
                    other.pos_start,other.pos_end,
                    "Element could not be removed because its index value is out of bounds",self.ctx
                )

    def mul_by(self,other):
        if isinstance(other, List):
            new_list = self.copy()
            #concatenates two lists together
            new_list.elements.extend(other.elements)
            return new_list, None
        else:
            return None, Value.illegal_op(self, other)

    def get_by(self,other):
        if isinstance(other, Number):
            try:
                return self.elements[other.value], None
            except:
                return None, RTError(
                    other.pos_start,other.pos_end,
                    "Element could not be extracted because it's index value is out of bounds",self.ctx
                )

    def copy(self):
        copy = List(self.elements)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.ctx)
        return copy

    def __str__(self):
        return ", ".join([str(i) for i in self.elements])

    def __repr__(self):
        return f'[{", ".join([str(i) for i in self.elements])}]'


#######################################
# CONTEXT
#######################################

class Context:
    def __init__(self,display_name,parent=None,parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbols = None

#######################################
# SYMBOLS
#######################################

class Symbols:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent

    # Get value from variable name
    def get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value

    def set(self,name,value):

        # Uses name as the key to set value
        self.symbols[name] = value

    def remove(self,name):
        del self.symbols[name]


#######################################
# INTERPRETER
#######################################

class Interpreter:
    #takes the node and visits the node to determine what expression to evaluate
    def visit(self,node, ctx):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self,method_name,self.no_visit_method)
        return method(node, ctx)

    def no_visit_method(self,node,ctx):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    ##################################

    def visit_String_Node(self,node,ctx):
        return RTResult().success(
            String(node.token.value).set_context(ctx).set_pos(node.pos_start, node.pos_end)
        )


    def visit_Number_Node(self,node,ctx):
        return RTResult().success(
            Number(node.token.value).set_context(ctx).set_pos(node.pos_start, node.pos_end)
        )

    def visit_Boolean_Node(self,node,ctx):
        return RTResult().success(
            Boolean(node.token.value).set_context(ctx).set_pos(node.pos_start,node.pos_end)
        )

    def visit_VarAccess_Node(self,node,ctx):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = ctx.symbols.get(var_name)

        if not value:
            return res.fail(RTError(
                node.pos_start, node.pos_end,
                f"'{var_name}' is not defined",
                ctx

            ))

        value = value.copy().set_pos(node.pos_start,node.pos_end).set_context(ctx)
        return res.success(value)

    def visit_VarAssign_Node(self,node,ctx):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = res.register(self.visit(node.value_node, ctx))
        if res.should_return(): return res

        ctx.symbols.set(var_name, value)
        return res.success(value)

    def visit_BinaryOperation(self, node, ctx):
        res = RTResult()

        left = res.register(self.visit(node.left_node, ctx))
        if res.should_return():
            return res

        right = res.register(self.visit(node.right_node, ctx))
        if res.should_return():
            return res

        operations = {
            TT_PLUS: left.add_to,
            TT_MINUS: left.sub_by,
            TT_MUL: left.mul_by,
            TT_DIV: left.div_by,
            TT_POWER: left.raise_to,
            TT_MODULO: left.mod_by,
            TT_EE: left.compare_eq,
            TT_NE: left.compare_ne,
            TT_LT: left.compare_lt,
            TT_GT: left.compare_gt,
            TT_LTE: left.compare_lte,
            TT_GTE: left.compare_gte,
            TT_KEYWORD: {
                'AND': left.and_by,
                'OR': left.or_by,
                'GET': left.get_by,
            }
        }

        # Stores the op token (TT_PLUS, TT_MINUS, etc.)
        op_type = node.op_tok.type

        if op_type in operations:

            # Checks for a dict in the operations dictionary for TT_KEYWORD
            if isinstance(operations[op_type], dict):
                keyword = node.op_tok.value
                result, error = operations[op_type][keyword](right)
            else:
                result, error = operations[op_type](right)
        else:
            error = RTError(
                node.pos_start, node.pos_end,
                f"Invalid operator: {node.op_tok.value}"
            )
            return res.fail(error)

        if error:
            return res.fail(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end))




    def visit_UnaryOpNode(self,node,ctx):
        res = RTResult()
        number = res.register(self.visit(node.node,ctx))

        if res.should_return(): return res

        error = None

        if node.op_tok.type == TT_MINUS:
            number, error = number.mul_by(Number(-1))
        elif node.op_tok.type.matches(TT_KEYWORD,'NOT'):
            number, error = number.not_by()

        if error:
            return res.fail(error)
        else:
            return res.success(number.set_pos(node.pos_start,node.pos_end))

    def visit_If_Node(self,node,ctx):

        res = RTResult()

        # Loop through every tuple in cases
        for condition, expr, return_null in node.cases:
            condition_value = res.register(self.visit(condition, ctx))
            if res.should_return(): return res

            # Check if the condition is true
            if str(condition_value) == 'True':

                # Evaluate expression by visiting
                expr_value = res.register(self.visit(expr, ctx))

                # Check for another error
                if res.should_return(): return res
                # Return value if true and no error
                return res.success(Number.null if return_null else expr_value)

        # If no other condition is true then execute else_case
        if node.else_case:
            expr, return_null = node.else_case
            # Evaluate else_case expression
            else_value = res.register(self.visit(expr,ctx))
            if res.should_return(): return res
            return res.success(Number.null if return_null else else_value)

        return res.success(Number.null)

    def visit_For_Node(self,node,ctx):
        res = RTResult()
        elements = []

        start_value = res.register(self.visit(node.start_value_node, ctx))
        if res.should_return(): return res

        end_value = res.register(self.visit(node.end_value_node, ctx))
        if res.should_return(): return res

        if node.step_value_node:
            step_value = res.register(self.visit(node.step_value_node, ctx))
            if res.should_return(): return res
        else:
            # Default increase value is 1
            step_value = Number(1)

        n = start_value.value

        # implement a check for the step value
        if step_value.value >= 0:
            condition = lambda: n < end_value.value
        else:
            condition = lambda: n > end_value.value

        while condition():
            # constantly add by the step_value
            ctx.symbols.set(node.var_name_tok.value, Number(n))
            n += step_value.value

            # Execute the body node
            value = res.register(self.visit(node.body_node, ctx))
            if res.should_return() and res.cont_loop == False and res.break_loop == False:return res

            if res.cont_loop:
                continue

            if res.break_loop:
                break

            elements.append(value)

        return res.success(
            Number.null if node.return_null else
            List(elements).set_context(ctx).set_pos(node.pos_start, node.pos_end)
        )

    def visit_While_Node(self, node, ctx):
        res = RTResult()
        elements = []

        # Infinite Loop to check until condition is fulfilled
        while True:
            # Evaluate condition
            condition = res.register(self.visit(node.condition_node, ctx))
            print(condition)
            if res.should_return(): return res


            # If condition is not true anymore then break
            if str(condition) == 'False': break

            # Execute body node
            value = res.register(self.visit(node.body_node, ctx))
            if res.should_return() and res.cont_loop == False and res.break_loop == False: return res

            if res.cont_loop:
                continue
            if res.break_loop:
                break

            elements.append(value)

        return res.success(Number.null if node.return_null else List(elements).set_context(ctx).set_pos(node.pos_start, node.pos_end))

    def visit_FuncDef_Node(self, node, ctx):
        res = RTResult()

        if node.var_name_tok:
            func_name = node.var_name_tok.value
        else:
            func_name = None
        body_node = node.body_node
        arg_names = [arg_name.value for arg_name in node.arg_name_toks]
        func_val = Function(func_name, body_node, arg_names, node.auto_return).set_context(ctx).set_pos(node.pos_start, node.pos_end)

        # Checks if there's a name for the function
        if node.var_name_tok:
            ctx.symbols.set(func_name, func_val)

        return res.success(func_val)

    def visit_Call_Node(self, node, ctx):
        res = RTResult()
        args = []

        val_to_call = res.register(self.visit(node.node_to_call, ctx))
        if res.should_return(): return res
        val_to_call = val_to_call.copy().set_pos(node.pos_start, node.pos_end)

        for arg_node in node.arg_nodes:
            args.append(res.register(self.visit(arg_node, ctx)))
            if res.should_return(): return res

        return_value = res.register(val_to_call.execute(args))
        if res.should_return(): return res
        return_value = return_value.copy().set_pos(node.pos_start, node.pos_end).set_context(ctx)

        return res.success(return_value)

    def visit_List_Node(self, node, ctx):
        res = RTResult()
        elements = []

        for element in node.element_nodes:
            elements.append(res.register(self.visit(element, ctx)))
            if res.should_return(): return res

        return res.success(
            List(elements).set_context(ctx).set_pos(node.pos_start, node.pos_end)
        )

    def visit_Return_Node(self, node, ctx):
        res = RTResult()

        if node.node_to_return:
            value = res.register(self.visit(node.node_to_return, ctx))
            if res.should_return(): return res
        else:
            value = Number.null

        return res.ret_success(value)

    def visit_Continue_Node(self,node,ctx):
        return RTResult().cont_success()

    def visit_Break_Node(self,node,ctx):
        return RTResult().break_success()




#######################################
# RUN
#######################################


GLOBAL_SYMBOLS = {
    "NONE": Number.null,
    "FALSE": Number.false,
    "TRUE": Number.true,
    "MATH_PI": Number.math_PI,
    "WRITE": Built_In_Functions.print,
    "WRITE_RETURN": Built_In_Functions.print_ret,
    "input": Built_In_Functions.input,
    "input_int": Built_In_Functions.input_int,
    "CLEAR": Built_In_Functions.clear,
    "CLS": Built_In_Functions.clear,
    "IS_NUMBER": Built_In_Functions.is_num,
    "IS_BOOLEAN": Built_In_Functions.is_bool,
    "IS_STRING": Built_In_Functions.is_str,
    "IS_LIST": Built_In_Functions.is_list,
    "IS_FUNCTION": Built_In_Functions.is_func,
    "APPEND": Built_In_Functions.append,
    "REMOVE": Built_In_Functions.pop,
    "COMBINE": Built_In_Functions.combine,
    "LENGTH_OF": Built_In_Functions.len,
    "RUN": Built_In_Functions.run,
    "TO_STRING": Built_In_Functions.to_str,
    "INDEX_OF": Built_In_Functions.index_of,
    "TO_NUMBER": Built_In_Functions.to_num
}

# Set the global symbols using the dictionary
global_symbols = Symbols()
for symbol, value in GLOBAL_SYMBOLS.items():
    global_symbols.set(symbol, value)
def run(fn, text):
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()

    if error: return None, error

    # AST
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    # Run program
    interpreter = Interpreter()
    context = Context('<program>')
    context.symbols = global_symbols
    result = interpreter.visit(ast.node, context)

    return result.value, result.error
