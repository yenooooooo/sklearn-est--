import subprocess
import os

cwd = r'c:\Users\ailee\github\sklearn-est--'
output_file = os.path.join(cwd, 'push_result.txt')

with open(output_file, 'w', encoding='utf-8') as f:
    try:
        f.write("Starting git push...\n")
        result = subprocess.run(['git', 'push'], cwd=cwd, capture_output=True, text=True)
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\nSTDERR:\n")
        f.write(result.stderr)
        f.write(f"\nReturn Code: {result.returncode}")
    except Exception as e:
        f.write(f"Exception: {str(e)}")
