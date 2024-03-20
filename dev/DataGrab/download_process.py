# TO BE RUN ON data-mover01

from DataGrab import DataGrab

base_path = '/fred/oz100/hroxburg'
data_path = f'{base_path}/TessData'

# -- Define Things of Interest -- #
sector = 5
cam = 4
ccd = 4
download_number = None
cut = None
n = 4

# -- Download specified data -- #
grabber = DataGrab(sector=sector,path=data_path)
grabber.download(cam=cam,ccd=ccd,number=download_number)

print('Complete.')