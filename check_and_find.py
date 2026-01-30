import json

file_path = r'c:\Users\ailee\github\Datasience\scikit-learn\colab_titanic-a-beginner-friendly-approach-to-top-3.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        source_lines = cell.get('source', [])
        source_text = ''.join(source_lines)
        if 'estimator__criterion' in source_text:
            print(f"Found in cell index {i} (execution count {cell.get('execution_count')})")
            print("--- Source ---")
            for line in source_lines:
                print(line, end='')
            print("\n--- End Source ---")
