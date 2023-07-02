import cv2
import glob
import numpy as np

path_target = './target/'
path_noise =  './noise/' 
path_mask = './mask/'

save_path  = './generated_dataset/'
if not os.path.exists('./generated_dataset/'):
        os.makedirs('./generated_dataset/')

filename_target = glob.glob(path_target+'*.png')
filename_noise = glob.glob(path_noise+'*.png')
filename_mask = glob.glob(path_mask+'*.png')

for num in range(0, len(filename_target)):

	target = cv2.resize(cv2.imread(filename_target[num]),(256,256))/255
	noise_img = cv2.resize(cv2.imread(filename_noise[num]),(256,256))/255
	mask_img = cv2.resize(cv2.imread(filename_mask[num]),(256,256))/255
	

	mask_img_bur = cv2.blur(mask_img,(10,10))

	contaminated_image = target*(1-mask_img_bur) + noise_img*mask_img_bur

	if not os.path.exists(save_path+ '/target/'):
        os.makedirs(save_path+ '/target/')

    if not os.path.exists(save_path+ '/noise/'):
        os.makedirs(save_path+ '/noise/')

    if not os.path.exists(save_path+ '/input/'):
        os.makedirs(save_path+ '/input/')

	cv2.imwrite(save_path+ '/target/'+str(num)+'.png', target*255)
	cv2.imwrite(save_path+ '/noise/'+str(num)+'.png', noise_img*255)
	cv2.imwrite(save_path+ '/input/'+str(num)+'.png', contaminated_image*255)
	print(num,end='\r')