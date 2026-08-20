import glob
import zipfile

wheel = glob.glob(".tmp_wheel/*.whl")[0]
print("wheel:", wheel)
names = zipfile.ZipFile(wheel).namelist()
lexicon = [n for n in names if "lexicon" in n]
env_files = [n for n in names if ".env" in n]
print("lexicon entries:", lexicon or "(missing!)")
print(".env entries:", env_files or "(missing)")
print("total files:", len(names))
