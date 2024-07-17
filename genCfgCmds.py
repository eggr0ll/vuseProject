'''
This file prints to fullFilePaths.cmd the terminal commands to run getcfg.py on .exe file paths. 
'''

# get partial file paths of .exe files
exeFilePaths = open("./partialExeFilePaths.txt")

# get front of file paths
frontFilePath = "C:/Users/mango1/Desktop/sourcecode_datasets/datasets/"
frontFilePath = frontFilePath.strip()

# get full file path for executables & print getcfg.py cmds to another file
with open("fullFilePaths.cmd", "w") as f:    # "a" vs. "w"?
    for x in range(25):
        # remove new line at end and create full path
        partialFilePath = exeFilePaths.readline().strip()
        fullFilePath = frontFilePath + partialFilePath

        # remove spaces & replace slashes w/ underscores for better output file names
        partialFilePath = partialFilePath.replace(" ", "").replace("/", "_")
        partialFilePath = partialFilePath[2:]
        
        print("python getcfg.py --input " + fullFilePath + " > results/" + partialFilePath + 
              ".output", file=f)
