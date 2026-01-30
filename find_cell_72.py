import json

file_path = r'c:\Users\ailee\github\Datasience\scikit-learn\colab_titanic-a-beginner-friendly-approach-to-top-3.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code' and cell.get('execution_count') == 72:
        print(f"Found Cell at index {i} with execution_count 72")
        print("--- Source ---")
        print(''.join(cell['source']))
        print("\n--- End Source ---")
        break
else:
    print("Cell with execution_count 72 not found.")
