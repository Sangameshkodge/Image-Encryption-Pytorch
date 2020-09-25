
from encryption import Cryptosys
###Test the code
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# Load Image
### Image should be Batch_size x channel x Heigh x Width
img = Image.open('./Images/test-images/lena512gray.bmp')
imgs = (TF.to_tensor(img).unsqueeze(0)).to('cuda')

fig =  plt.figure()
fig.add_subplot(1,3,1)
plt.imshow(imgs[0][0].cpu(), cmap='gray')
plt.title("Original")

# Declare crypto object
cryptosys = Cryptosys(load= False, image_size=[1,512,512], iterations=100, path='./keys/key_gray', save=True, device='cuda')

# Encrypt Image
encrypt_imgs = cryptosys(imgs)

fig.add_subplot(1,3,2)
plt.imshow(encrypt_imgs[0][0].cpu(), cmap='gray')
plt.title("Encrypt")

# Decrypt Image
decrypt_imgs = cryptosys.inverse(encrypt_imgs)

fig.add_subplot(1,3,3)
plt.imshow(decrypt_imgs[0][0].cpu(), cmap='gray')
plt.title("Decrypt")
plt.show()

### Save the Images
plt.imshow(encrypt_imgs[0][0].cpu(), cmap='gray')
plt.savefig('./Images/encrypted/lena512gray.png')
plt.imshow(decrypt_imgs[0][0].cpu(), cmap='gray')
plt.savefig('./Images/decrypted/lena512gray.png')



# Load Image
### Image should be Batch_size x channel x Heigh x Width
img = Image.open('./Images/test-images/lena512rgb.tiff')
imgs = (TF.to_tensor(img).unsqueeze(0)).to('cuda')

fig =  plt.figure()
fig.add_subplot(1,3,1)
plt.imshow(imgs[0].permute(1,2,0).contiguous().cpu())
plt.title("Original")

# Declare crypto object
cryptosys = Cryptosys(load= False, image_size=[3,512,512], iterations=100, path='./keys/key_rgb',  save=True, device='cuda')

# Encrypt Image
encrypt_imgs = cryptosys(imgs)

fig.add_subplot(1,3,2)
plt.imshow(encrypt_imgs[0].permute(1,2,0).contiguous().cpu())
plt.title("Encrypt")

# Decrypt Image
decrypt_imgs = cryptosys.inverse(encrypt_imgs)

fig.add_subplot(1,3,3)
plt.imshow(decrypt_imgs[0].permute(1,2,0).contiguous().cpu())
plt.title("Decrypt")
plt.show()

### Save the Images
plt.imshow(encrypt_imgs[0].permute(1,2,0).contiguous().cpu())
plt.savefig('./Images/encrypted/lena512rgb.png')
plt.imshow(decrypt_imgs[0].permute(1,2,0).contiguous().cpu())
plt.savefig('./Images/decrypted/lena512rgb.png')
