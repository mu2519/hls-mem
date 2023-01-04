import os
import shutil

shutil.rmtree('./src', ignore_errors=True)
os.mkdir('./src')
open('./src/.gitignore', 'w').close()
