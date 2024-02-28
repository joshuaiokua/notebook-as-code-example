from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from datetime import datetime
import subprocess
import requests

from nbformat import read
import ast
import io

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_code_from_notebook(notebook_path):
    """
    Get the code from a Jupyter Notebook.

    Parameters:
    - notebook_path: Path to the input notebook (.ipynb file).

    Returns:
    - code: The code from the notebook.
    """
    code_string = ""

    # Load the notebook
    with io.open(notebook_path, 'r', encoding='utf-8') as f:
        nb = read(f, as_version=4)
    
    # Add only code cells without outputs to the code string
    for cell in nb.cells:
        if cell.cell_type == 'code' and len(cell.outputs) == 0:
            code_string += "\n" + cell.source + "\n"

    return code_string.strip()

def extract_code_elements(code):
    """
    Extracts imports, functions (with bodies), and classes (with bodies) from a given Python code string.
    
    Parameters:
    - code (str): A string containing Python code.
    
    Returns:
    - dict: A dictionary with keys 'imports', 'functions', and 'classes', each containing details about their respective elements.
    """
    tree = ast.parse(code)
    imports, functions, classes = [], {}, {}
    consolidated_imports = []

    # Function to recursively process nodes, with added context to skip class methods
    def process_node(node, inside_class=False):
        # Extract imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
                consolidated_imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module if node.module else ""
            aliases = [alias.name for alias in node.names]
            for alias in aliases:
                imports.append(f"{module}.{alias}")
            consolidated_imports.append(f"from {module} import {', '.join(aliases)}")

        # Extract functions (excluding class methods)
        elif isinstance(node, ast.FunctionDef) and not inside_class:
            function_body = ast.get_source_segment(code, node)
            if node.name not in functions:
                functions[node.name] = function_body
            else:
                raise ValueError(f"Function name '{node.name}' is already in use.")

        # Extract classes
        elif isinstance(node, ast.ClassDef):
            class_body = ast.get_source_segment(code, node)
            if node.name not in classes:
                classes[node.name] = class_body
            else:
                raise ValueError(f"Class name '{node.name}' is already in use.")
            # Process all nodes within the class, marking them as inside a class
            for child in ast.iter_child_nodes(node):
                process_node(child, inside_class=True)

    
    # Start processing from the root node
    for node in ast.walk(tree):
        if isinstance(node, ast.Module):
            for child in ast.iter_child_nodes(node):
                process_node(child)

    # Convert list of import strings to a single string
    consolidated_import_string = "\n".join(consolidated_imports)
    
    return {
        'imports': {'modules': imports, 
                    'import_string': consolidated_import_string.strip()},
        'functions': functions,
        'classes': classes
    }

def get_code_string_from_code_elements(code_elements, keys=['functions', 'classes']):
    """
    Converts a dictionary of code elements into a single code string.
    """
    code_string = ''
    for key in keys:
        for code_element in code_elements[key].values():
            code_string += code_element + '\n\n'
    return code_string.strip()

def get_class_and_function_names(code_elements: dict) -> list:
    """
    Extracts the names of classes and functions from a dictionary of code elements.
    
    Parameters:
    - code_elements (dict): A dictionary containing 'functions' and 'classes' keys, each with a list of dictionaries containing 'name' keys.
    
    Returns:
    - list: A list of class and function names.
    """
    return list(code_elements['functions'].keys()) + list(code_elements['classes'].keys())

def get_estimated_class_and_function_names(response):
    """
    Extracts the estimated names of classes and functions from an OpenAI API response.
    
    Parameters:
    - response (str or dict): The response from the OpenAI API, either as a string or a dictionary.
    
    Returns:
    - list: A list of estimated class and function names."""
    if isinstance(response, str):
        response = json.loads(response)

    estimated_output = []

    for item in response.values():
        estimated_output.extend(item['content'])
    
    return estimated_output

def organize_with_openai(client, code_string:str, full_code_string=False, seed:int=123, model:str="gpt-4-turbo-preview"):
    """
    Organizes the code into separate python files using the OpenAI API.

    """
    # Set the prompt
    base_message = "Please group the classes and functions defined in the following code into separate python files. You do not have to list out the full text to be included in each python file, but please provide a simple list of what function and or class each python file contains as well as a short description the file's basic functionality and purpose. Please try to group the functions and classes into as few files as possible. Please provide JSON output in the following format for each file: "
    example_output_format = "\'filename.py\': {\'description\': \'Two-Three sentence description\', \'content\': [\'function1\', \'class1\']}"
    message = base_message + example_output_format

    if full_code_string == True:
        pass
    else:
        code_string = get_code_string_from_code_elements(
            extract_code_elements(code_string)
            )

    # Call the API
    response = client.chat.completions.create(
        seed=seed,
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": message},
            {"role": "user", "content": code_string}
        ]
    )

    return json.loads(response.choices[0].message.content)

def write_files_from_openai_response(response, extracted_code_elements, destination='src'):
    """
    Writes Python files from the estimated class and function names provided by the OpenAI API response.
    
    Parameters:
    - response (str or dict): The response from the OpenAI API, either as a string or a dictionary.
    - extracted_code_elements (dict): A dictionary containing 'functions' and 'classes' keys, extracted from the original code using AST.
    - destination (str, optional): The destination folder where the files should be created. Defaults to 'src'.
    - custom_lines (list of str, optional): A list of custom lines to be added to the code body. Defaults to None.
    
    Returns:
    - list: A list of the paths to the created Python files or error messages.
    """
    if isinstance(response, str):
        response = json.loads(response)

    # Check if the 'src' folder exists
    if not os.path.exists(destination):
        os.makedirs(destination)

    # Check if the 'import.py' file exists in the destination folder
    if not os.path.exists(os.path.join(destination, 'imports.py')):
        write_python_file(destination, 'imports', extracted_code_elements['imports']['import_string'], description="Imports for the project.")

    # The functions and classes extracted from the original code
    code_functions_and_classes = extracted_code_elements['functions'].copy()
    code_functions_and_classes.update(extracted_code_elements['classes'])

    # Write the code for each function or class to the file
    paths = []

    for file_name, details in response.items():
        file_code_string = ""
        file_description = details['description']

        # The functions and or classes estimated to be in the file 
        estimated_code_elements = details['content']

        # Add the code for each function or class to the file
        for code_element in estimated_code_elements:
            if code_element in code_functions_and_classes:
                file_code_string += code_functions_and_classes[code_element] + "\n\n"
            else:
                return f"Error: {code_element} not found in code elements."
        
        # Write the file
        path = write_python_file(destination, file_name, file_code_string, file_description, custom_lines = f"from {destination + '.imports'} import *")
        paths.append(path)
    
    return paths

def write_python_file(destination, file_name, code, description=None, custom_lines=None):
    """
    Writes a string of code to a Python file at the specified destination, with optional custom lines added to the code body.
    
    Parameters:
    - destination (str): The destination folder where the file should be created.
    - file_name (str): The name of the file (without the .py extension).
    - code (str): The Python code to write to the file.
    - description (str, optional): A description of the file's basic functionality and purpose. Defaults to None.
    - custom_lines (list of str, optional): A list of custom lines to be added to the code body. Defaults to None.
    
    Returns:
    - str: The path to the created Python file or an error message.
    """
    # Ensure the destination ends with a slash
    if not destination.endswith(os.sep):
        destination += os.sep
    
    # Ensure the file name ends with .py
    if not file_name.endswith('.py'):
        file_name += '.py'

    # Add custom lines to the code if provided
    if custom_lines is not None:
        code = custom_lines + "\n\n" + code
    
    # Add a description to the code if provided
    if description is not None:
        description_string = f'"""\n{description}\nFile generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n"""'
        code = description_string + "\n\n" + code
    
    # Construct the full path to the file
    full_path = os.path.join(destination, file_name)
    
    # Try to write the code to the file
    try:
        with open(full_path, 'w', encoding='utf-8') as file:
            file.write(code)
        return f"File written successfully to {full_path}"
    except Exception as e:
        return f"Error writing file: {e}"
    
def check_equivalency(list1, list2):
    """
    Checks if two lists contain the same elements, regardless of order.
    
    Parameters:
    - list1 (list): The first list to compare.
    - list2 (list): The second list to compare.
    
    Returns:
    - bool: True if the lists contain the same elements, False otherwise.
    """
    return set(list1) == set(list2)

def deploy_notebook_as_code(notebook_path, destination='src'):
    """
    Deploys a Jupyter Notebook as separate Python files using the OpenAI API.
    
    Parameters:
    - notebook_path (str): The path to the input notebook (.ipynb file).
    - destination (str, optional): The destination folder where the files should be created. Defaults to 'src'.
    
    Returns:
    - list: A list of the paths to the created Python files or error messages.
    """
    # Get the code from the notebook
    code_string = get_code_from_notebook(notebook_path)
    
    # Extract the code elements from the code string
    extracted_code_elements = extract_code_elements(code_string)
    
    # Organize the code into separate python files using the OpenAI API
    response = organize_with_openai(client, code_string)

    # Check if the estimated class and function names are equivalent to the actual class and function names
    if not check_equivalency(
        get_class_and_function_names(extracted_code_elements), get_estimated_class_and_function_names(response)
        ):
        raise ValueError("The estimated class and function names are not equivalent to the actual class and function names.")
    
    # Write the files from the OpenAI response
    paths = write_files_from_openai_response(response, extracted_code_elements, destination)
    
    return paths

def run_git_commands(commit_message, token, repo_owner, repo_name):
    """
    Runs Git commands to create a new branch, add, commit, and push changes.
    Uses a Personal Access Token for authentication with GitHub.

    Parameters:
    - commit_message (str): The commit message to use.
    - github_token (str): GitHub Personal Access Token for authentication.
    """
    # Generate a unique branch name based on the current timestamp
    branch_name = f"deployed-code-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    try:
        # Create and switch to the new branch
        subprocess.run(["git", "checkout", "-b", branch_name], check=True)

        # Set Git user name and email if not already configured
        subprocess.run(["git", "config", "user.name", repo_owner], check=True)
        subprocess.run(["git", "config", "user.email", "youremail@example.com"], check=True)

        # Add changes to the staging area
        subprocess.run(["git", "add", "."], check=True)

        # Commit changes
        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        # Set the remote URL to include the token for authentication
        repo_url_with_token = f"https://{token}@github.com/{repo_owner}/{repo_name}.git"
        subprocess.run(["git", "remote", "set-url", "origin", repo_url_with_token], check=True)

        # Push the new branch to the remote repository
        subprocess.run(["git", "push", "-u", "origin", branch_name], check=True)
        print(f"Changes pushed to GitHub on branch {branch_name}.")

        # Optionally, reset the remote URL to the original to avoid leaving the token in the config
        original_repo_url = "https://github.com/{repo_owner}/{repo_name}.git"
        subprocess.run(["git", "remote", "set-url", "origin", original_repo_url], check=True)

        return branch_name  # Return the branch name for further use
    except subprocess.CalledProcessError as e:
        print(f"Error running Git command: {e}")
        return None
    
def create_github_pr(token, repo_owner, repo_name, head_branch, base_branch="main", title="Automated PR title", body="Description of the changes."):
    """
    Creates a pull request on GitHub using the GitHub API.
    """
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    data = {"title": title, "body": body, "head": head_branch, "base": base_branch}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        print("Pull request created successfully!")
        print("URL:", response.json()['html_url'])
    else:
        print("Failed to create pull request")
        print("Response:", response.json())

# Deploy the notebook as code
if __name__ == "__main__":
    notebook_path = "rag-example-notebook.ipynb"
    paths = deploy_notebook_as_code(notebook_path)
    print(paths)

    # Run Git commands
    commit_message = "Automated commit message"
    token = os.getenv("GITHUB_API_DEMO_TOKEN")
    repo_owner = "joshuaiokua"
    repo_name = "notebook-as-code-example"

    branch_name = run_git_commands(commit_message, token, repo_owner, repo_name)
    if branch_name:
        create_github_pr(token, repo_owner, repo_name, branch_name, title="Deploying a Notebook as Code", body="You've deployed a notebook as code!")