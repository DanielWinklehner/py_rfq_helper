import subprocess

mpiexe = "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"
pythonexe = "E:\Anaconda3\python.exe"

print(subprocess.call([mpiexe, "/np", "4", pythonexe, "mpi_bempp_test.py"]))
