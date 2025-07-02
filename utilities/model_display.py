# models/model_display.py

import ast
import inspect

def func_to_latex(func):
    """
    Convert a Python function to a LaTeX string.
    
    Args:
    func (callable): The function to convert
    
    Returns:
    str: A LaTeX formatted string representing the function
    """
    # Get the source code of the function
    source = inspect.getsource(func)
    # remove numpy calls
    source = source.replace('np.','')
    
    # Parse the source code into an AST
    tree = ast.parse(source)
    
    # Find the return statement
    return_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            return_node = node
            break
    
    if return_node is None:
        return "Could not parse function"
    
    # Convert the AST to LaTeX
    latex = ast_to_latex(return_node.value)
    
    return f"y = {latex}"

def ast_to_latex(node):
    """
    Recursively convert an AST node to LaTeX.
    """
    if isinstance(node, ast.BinOp):
        left = ast_to_latex(node.left)
        right = ast_to_latex(node.right)
        if isinstance(node.op, ast.Add):
            return f"{left} + {right}"
        elif isinstance(node.op, ast.Sub):
            return f"{left} - {right}"
        elif isinstance(node.op, ast.Mult):
            return f"{left} \\cdot {right}"
        elif isinstance(node.op, ast.Div):
            return f"\\frac{{{left}}}{{{right}}}"
        elif isinstance(node.op, ast.Pow):
            return f"{left}^{{{right}}}"
    if isinstance(node, ast.UnaryOp):
        operand = ast_to_latex(node.operand)
        if isinstance(node.op, ast.USub):
            return f"- {operand}"
        elif isinstance(node.op, ast.UAdd):
            return f"+ {operand}"
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in ['exp']:
                return f"e^{{{ast_to_latex(node.args[0])}}}"
            elif node.func.id in ['sin', 'cos', 'tan', 'log', 'sqrt']:
                return f"\\{node.func.id}({ast_to_latex(node.args[0])})"
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Num):
        return str(node.n)
    return str(node)

def get_human_readable_function(model_function, params=None):
    """
    Generate a human-readable LaTeX string for the model function.
    
    Args:
    model_function (callable): The model function
    params (dict): A dictionary of parameter names and values
    
    Returns:
    str: A LaTeX formatted string representing the model function
    """
    latex_func = func_to_latex(model_function)
    
    # Replace parameter names with their values
    if params is not None:
        for param, value in params.items():
            latex_func = latex_func.replace(param, f"{value:.2f}")
    
    return latex_func

