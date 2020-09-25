import torch
import torch.nn as nn

class Cryptosys(nn.Module):
    def __init__(self,
                iterations=10,
                alpha = 8,
                image_size= [3,32,32],
                load=False,
                save=False,
                path = None,
                device='cpu'
                ):
        """
        Credits:
        Khaled Loukhaoukha, Jean-Yves Chouinard, Abdellah Berdai, 
        "A Secure Image Encryption Algorithm Based on Rubik's Cube Principle", 
        Journal of Electrical and Computer Engineering, vol. 2012, Article ID 173931, 
        13 pages, 2012. https://doi.org/10.1155/2012/173931
        Args:
            iterations: Number of iterations for the encryption 
            alpha: bit precision of input image generally 8 
            image size: dimentions of image [channels, height, width]
            load: loads key, iterations and image_size
            save: saves key, iterations and image_size
            path: directory path to save the key to
            device: cuda device given to .to()

        """
        super(Cryptosys, self).__init__()
        self.device=device
        if load==False:
            self.image_size = image_size
            self.iterations = iterations
            self.kr = torch.randint(low=0, high=  int(2**alpha), size = [self.image_size[1]] ).to(self.device)
            self.kc = torch.randint(low= 0, high= int(2**alpha), size =[self.image_size[1]] ).to(self.device)
        self.load_save(load = load, save= save, path = path)
            

    def load_save(self, load =True, save=False, path='./keys/keys'):
        """
        Args:
            load : True for loading parameters
            save : True for saving parameters
            path : directory for load/save
        
        """
        if save:
            print("Saving Encryption Parameters")
            torch.save({'kr': self.kr, 
                        'kc':self.kc, 
                        'iterations': self.iterations, 
                        'image_size':self.image_size},
                        path+'.inkey')
        elif load:
            print("loading Encryption Parameters")
            key = torch.load(path)
            self.kr =key['kr'].to(self.device)
            self.kc =key['kc'].to(self.device)
            self.iterations = key['iterations']
            self.image_size = key['image_size']
        else:
            pass
        return
    
    def shift(self, x, M, key, dim=2):  
        """
        Args: 
            x(Tensor): Input to be shifted based on key
            M(Tensor): Direction of the shift
            key(tensor): number of index to shift the vector
            dim : operation on row or columns
        Returns:
            Tensor with shift operations based on direction and key.
        """
        M = M.repeat((x.shape[dim],1,1,1)).permute((1,2,3,0)).contiguous()  
        key = key % x.shape[dim] 
        if dim==2:
            for k in range(x.shape[2]):
                x[:,:,k] = torch.where(M[:,:,k]==0.0, torch.cat((x[:,:,k, key[k]:], x[:,:,k, :key[k]]), dim=2),  torch.cat((x[:,:,k,-key[k]:], x[:,:,k, :-key[k] ]), dim=2) )
        elif dim==3:
            for k in range(x.shape[3]):
                x[:,:,:,k] = torch.where(M[:,:,k]==1.0, torch.cat((x[:,:, key[k]:, k], x[:,:,:key[k], k]), dim=2),  torch.cat((x[:,:, -key[k]:, k], x[:,:,:-key[k], k]), dim=2) )
        else:
            raise ValueError
        return x

    def Xor_scramble(self, x, key, dim= 2):
        """
        Args:
            x (tensor) : Input for key based Xor
            key: key for Xor
            dim: dimension along which Xor has to be computed
        Returns:
            Tensor with srambled Xor of Inputs
        
        """
        reverse_key= key.flip(dims = [0])
        if dim ==2:
            for i in range(key.shape[0]):
                x[:,:,::2,i] = x[:,:,::2,i] ^ key[i] 
                x[:,:,1::2,i] = x[:,:,1::2,i] ^ reverse_key[i]

        elif dim==3:
            for i in range(key.shape[0]):
                x[:,:,i,::2] = x[:,:,i,::2] ^ key[i] 
                x[:,:,i,1::2] = x[:,:,i,1::2] ^ reverse_key[i]
        else:
            raise ValueError
        return x

    def encrypt(self, x):
        """
        Args:
            x :Input to be encrypted
        Returns:
            Encrpyted Input
        """
        for iter in range(self.iterations):
            #Step 4
            Malpha = x.sum(dim=3) % 2
            x = self.shift(x, Malpha, -self.kr, dim=2)
            #Step 5
            Mbeta = x.sum(dim=2) % 2
            x = self.shift(x, Mbeta, -self.kc, dim=3)
            #Step 6
            x = self.Xor_scramble(x, self.kc, dim=2)
            #Step 7
            x = self.Xor_scramble(x, self.kr, dim=3)
        return x

    def decrypt(self, x):
        """
        Args:
            x :Input to be decrypted
        Returns:
            Decrpyted Input
        """
        for iter in range(self.iterations):
            #Step 3
            x = self.Xor_scramble(x, self.kr, dim=3)
            #Step 4
            x = self.Xor_scramble(x, self.kc, dim=2)
            #Step 5
            Mbeta = x.sum(dim=2) % 2
            x = self.shift(x, Mbeta, self.kc, dim=3)
            #Step 6
            Malpha = x.sum(dim=3) % 2
            x = self.shift(x, Malpha, self.kr, dim=2)
        return x

    def forward(self, x):
        assert len(x.shape)==4, "Expecting a input of dimension 4 but got dimension {} instead. If using a single image for encryption use .unsqueeze(0) to make first dimention size 1".format(len(x.shape))
        return (self.encrypt((x*255.0).long())).float()/255.0
         
    def inverse(self,x):
        assert len(x.shape)==4, "Expecting a input of dimension 4 but got dimension {} instead. If using a single image for encryption use .unsqueeze(0) to make first dimention size 1".format(len(x.shape))
        return (self.decrypt((x*255.0).long())).float()/255.0
        
