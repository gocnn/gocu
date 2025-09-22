import re
import sys

def snake_to_camel(snake_str):
    components = snake_str.lower().split('_')
    return ''.join(x.capitalize() for x in components)

def convert_enum_to_go(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with latin-1 encoding as fallback
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Last resort: read as binary and decode with error handling
            with open(file_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='ignore')

    # Find the enum block
    enum_pattern = r'typedef enum CUdevice_attribute_enum \{(.*?)\} CUdevice_attribute;'
    match = re.search(enum_pattern, content, re.DOTALL)
    if not match:
        raise ValueError("Could not find CUdevice_attribute enum in the file.")

    enum_body = match.group(1).strip()

    # Split into lines
    lines = [line.strip() for line in enum_body.split('\n') if line.strip()]

    go_consts = []
    current_group = None

    # Regex for parsing enum line: NAME = VALUE, /**< comment */
    line_pattern = r'(\w+)\s*=\s*(\d+),\s*/\*\*<\s*(.*?)\s*\*/'

    for line in lines:
        # Check if it's a group comment like /** Group description */
        if line.startswith('/**') and not '=' in line:
            group_match = re.match(r'/\*\*\s*(.*?)\s*\*/', line)
            if group_match:
                current_group = group_match.group(1).strip()
                go_consts.append(f'\t// {current_group}')
            continue

        # Parse the enum value line
        match = re.match(line_pattern, line)
        if match:
            full_name = match.group(1)
            value = match.group(2)
            comment = match.group(3)

            if full_name.startswith('CU_DEVICE_ATTRIBUTE_'):
                short_name = full_name[len('CU_DEVICE_ATTRIBUTE_'):]
                go_name = snake_to_camel(short_name)
                go_line = f'\t{go_name} DeviceAttribute = C.{full_name} // {comment}'
                go_consts.append(go_line)

    # Assemble the Go code
    go_code = '''// DeviceAttribute represents CUDA device attributes that can be queried.
type DeviceAttribute int

const ('''
    if current_group is None:
        go_code += '\n\t// Thread and block limits'  # Default group as per reference

    go_code += '\n' + '\n'.join(go_consts) + '\n)'

    return go_code

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py /path/to/cuda.h")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        go_code = convert_enum_to_go(file_path)
        print(go_code)
    except ValueError as e:
        print(e)
        sys.exit(1)