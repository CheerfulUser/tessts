# TO BE RUN ON A LOGIN NODE 

import os
import tessreduce as tr
from time import sleep
from time import time as t
import numpy as np

tStart = t()

base_path = '/fred/oz100/hroxburg'
data_path = f'{base_path}/TessData'
working_path = f'{base_path}/TessAllSky/CompleteProcess'

# -- Define Things of Interest -- #
sector = 5
cam = 4
ccd = 4
download_number = None
cut = None
n = 4

time2 = 10
time1 = '2:00'
cpus = 28
mem = 2

# -- Delete old scripts -- #
if os.path.exists(f'{working_path}/task_script1.sh'):
    os.system(f'rm {working_path}/task_script1.sh')
    os.system(f'rm {working_path}/task_script1.py')
if os.path.exists(f'{working_path}/task_script2.sh'):
    os.system(f'rm {working_path}/task_script2.sh')
    os.system(f'rm {working_path}/task_script2.py')


finished = False
message = f'Waiting for FFI Download : Sector {sector} Cam {cam} CCD {ccd}'
print('\n')
while not finished:
    if os.path.exists(f'{data_path}/Sector{sector}/Cam{cam}Ccd{ccd}/'):
        folder_length = len(os.listdir(f'{data_path}/Sector{sector}/Cam{cam}Ccd{ccd}/'))
        sleep(60)
        new_length = len(os.listdir(f'{data_path}/Sector{sector}/Cam{cam}Ccd{ccd}/'))
        if new_length - folder_length == 0:
            finished = True
        else:
            print(message, end='\r')
            message += '.'
    else:
        print(message, end='\r')
        message += '.'
        sleep(60)

# -- Create python file for cubing, cutting, reducing a cut-- # 
print('Creating Cubing/Cutting Python File')
python_text = f"\
from DataGrab import DataGrab\n\
\n\
sector = {sector}\n\
cam = {cam}\n\
ccd = {ccd}\n\
base_path = '{base_path}'\n\
data_path = f'{data_path}'\n\
download_number = {download_number}\n\
cut = {cut}\n\
n = {n}\n\
\n\
grabber = DataGrab(sector=sector,path=data_path,verbose=2)\n\
grabber.make_cube(cam=cam,ccd=ccd)\n\
grabber.make_cuts(cam=cam,ccd=ccd,n=n)"

python_file = open(f"{working_path}/task_script1.py", "w")
python_file.write(python_text)
python_file.close()

# -- Create bash file to submit job -- #
print('Creating Cubing/Cutting Batch File')
batch_text = f"\
#!/bin/bash\n\
#\n\
#SBATCH --job-name=tessreduce_attempt\n\
#SBATCH --output=/fred/oz100/hroxburg/job_logs/job_output_%A.txt\n\
#SBATCH --error=/fred/oz100/hroxburg/job_logs/errors_%A.txt\n\
#\n\
#SBATCH --ntasks=1\n\
#SBATCH --time={time1}:00\n\
#SBATCH --cpus-per-task={cpus}\n\
#SBATCH --mem-per-cpu={mem}G\n\
\n\
cd {base_path}\n\
python {working_path}/task_script1.py"

batch_file = open(f'{working_path}/task_script1.sh', "w")
batch_file.write(batch_text)
batch_file.close()

print('Submitting Cubing/Cutting Batch File')
#os.system(f'sbatch {working_path}/task_script1.sh')

if cut is None:
    cuts = range(1,n**2+1)
else:
    cuts = [cut]

# -- Generate Star Catalogue -- #
print('\n')
for i in cuts:
    message = f'Waiting for Cut {i}'
    found = False
    path = f'{data_path}/Sector{sector}/Cam{cam}Ccd{ccd}/Cut{i}of{n**2}'
    cutName = f'sector{sector}_cam{cam}_ccd{ccd}_cut{i}_of{n**2}.fits'
    while not found:
        if os.path.exists(f'{path}/{cutName}'):
            found = True
            if not os.path.exists(f'{path}/local_gaia_cat'):
                print(f'Generating Catalogue {i}')
                tpf = tr.lk.TessTargetPixelFile(f'{path}/{cutName}')
                tr.external_save_cat(tpf=tpf,path=path,maglim=19)
                del(tpf)
        else:
            print(message, end='\r')
            sleep(30)
            message += '.'

# -- Create python file for reducing a cut-- # 
print('\n')
print('Creating Reduction Python File')
python_text = f"\
from DataGrab import DataGrab\n\
from time import time as t\n\
\n\
sector = {sector}\n\
cam = {cam}\n\
ccd = {ccd}\n\
base_path = '{base_path}'\n\
data_path = f'{data_path}'\n\
download_number = {download_number}\n\
cut = {cut}\n\
n = {n}\n\
\n\
grabber = DataGrab(sector=sector,path=data_path)\n\
grabber.reduce(cam=cam,ccd=ccd,cut=cut,n=n)\n\
\n\
print(f'Total Cube,Cut,Reduce time taken = " + "{((t()-" + str(tStart) + ")/60):.2f} mins')"

python_file = open(f"{working_path}/task_script2.py", "w")
python_file.write(python_text)
python_file.close()

# -- Create bash file to submit job -- #
print('Creating Reduction Batch File')
batch_text = f"\
#!/bin/bash\n\
#\n\
#SBATCH --job-name=tessreduce_attempt\n\
#SBATCH --output=/fred/oz100/hroxburg/job_logs/job_output_%A.txt\n\
#SBATCH --error=/fred/oz100/hroxburg/job_logs/errors_%A.txt\n\
#\n\
#SBATCH --ntasks=1\n\
#SBATCH --time={time2}:00\n\
#SBATCH --cpus-per-task={cpus}\n\
#SBATCH --mem-per-cpu={mem}G\n\
\n\
cd {base_path}\n\
python {working_path}/task_script2.py"

batch_file = open(f'{working_path}/task_script2.sh', "w")
batch_file.write(batch_text)
batch_file.close()
        
print('Submitting Reduction Batch File')
os.system(f'sbatch {working_path}/task_script2.sh')