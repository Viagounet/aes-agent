import ast
import sys

from typing import TypedDict, Any


class ToolCallingResults(TypedDict):
    reasoning: str
    tool_called_name: str
    tool_called_arguments: dict
    tool_called_result: Any


def parse_function_call(call_string):
    """
    Parses a string representation of a Python function call.

    Extracts the function name, positional argument values, and keyword
    argument values. Handles various literal types including strings
    (with internal quotes, newlines), numbers, lists, dictionaries, etc.

    Args:
        call_string: A string containing a Python function call
                     (e.g., "my_func(1, 'hello', key='world')").

    Returns:
        A dictionary containing:
        - 'function_name': The name of the function (str).
        - 'positional_args': A list of evaluated positional argument values.
        - 'keyword_args': A dictionary of evaluated keyword arguments
                          (key: str, value: evaluated value).
        Returns None if the string cannot be parsed as a function call
        expression or if a SyntaxError occurs during parsing.
        Returns None if argument values are not literals (e.g. variables
        or complex expressions that ast.literal_eval cannot handle).
    """
    try:
        # Parse the string into an Abstract Syntax Tree (AST)
        # We wrap it slightly in case of Python version differences in parsing bare expressions
        tree = ast.parse(f"_{call_string.strip()}")

        # Check if the parsed tree is a simple expression containing a call
        if not (
            isinstance(tree, ast.Module)
            and len(tree.body) == 1
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Call)
        ):
            print(
                "Error: Input string does not seem to be a single function call.",
                file=sys.stderr,
            )
            return None

        call_node = tree.body[0].value

        # --- Extract Function Name ---
        function_name = None
        if isinstance(call_node.func, ast.Name):
            function_name = call_node.func.id
        # TODO: Add handling for attribute calls like 'obj.method()' if needed
        # elif isinstance(call_node.func, ast.Attribute):
        #     # This would require recursively building the name or deciding how to represent it
        #     function_name = ast.unparse(call_node.func) # Reconstructs the name potentially
        else:
            print(
                f"Error: Unsupported function call type: {type(call_node.func)}",
                file=sys.stderr,
            )
            return None  # Or handle other types like attribute access (obj.method) if needed

        # --- Extract Positional Arguments ---
        positional_args = []
        for arg_node in call_node.args:
            try:
                # ast.literal_eval safely evaluates literal expressions
                positional_args.append(ast.literal_eval(arg_node))
            except ValueError:
                # Handle cases where the argument is not a literal
                # (e.g., a variable name or complex expression)
                # You might want to store the raw string representation instead:
                # positional_args.append(ast.unparse(arg_node))
                print(
                    f"Warning: Positional argument '{ast.unparse(arg_node)}' is not a literal. Skipping.",
                    file=sys.stderr,
                )
                return None  # Or adjust behavior as needed

        # --- Extract Keyword Arguments ---
        keyword_args = {}
        for kw_node in call_node.keywords:
            arg_name = kw_node.arg
            if arg_name is None:
                print(
                    f"Error: Found '**kwargs' expansion, which is not directly supported for value extraction.",
                    file=sys.stderr,
                )
                return None  # Cannot evaluate **kwargs without context
            try:
                # ast.literal_eval safely evaluates the value node
                keyword_args[arg_name] = ast.literal_eval(kw_node.value)
            except ValueError:
                # Handle cases where the keyword argument value is not a literal
                # keyword_args[arg_name] = ast.unparse(kw_node.value)
                print(
                    f"Warning: Keyword argument '{arg_name}={ast.unparse(kw_node.value)}' value is not a literal. Skipping.",
                    file=sys.stderr,
                )
                return None  # Or adjust behavior as needed

        return {
            "function_name": function_name,
            "positional_args": positional_args,
            "keyword_args": keyword_args,
        }

    except SyntaxError as e:
        print(f"Syntax Error parsing the string: {e}", file=sys.stderr)
        return None
    except Exception as e:
        # Catch other potential errors during AST processing
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return None
