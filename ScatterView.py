import numpy as np
import struct
import matplotlib.pyplot as plt


def circle():
    # Refer to: https://blog.csdn.net/yuzeyuan12/article/details/108572868
    X = np.linspace(-20, 20, 256).reshape([1, 256])
    Y = np.linspace(-20, 20, 256).reshape([256, 1])
    sample = (X - 0) ** 2 + (Y - 0) ** 2 <= 1 ** 2
    index = np.ones((256, 256), dtype='complex128') * 1.0
    index[sample] = 1.6
    return index


class planewave:
    def read(self, file, precision=8):
        
        if precision == 8:
            E0_x_r_bytes = file.read(8)
            E0_x_r = struct.unpack("d", E0_x_r_bytes)[0]
            
            E0_x_i_bytes = file.read(8)
            E0_x_i = struct.unpack("d", E0_x_i_bytes)[0]
            
            E0_y_r_bytes = file.read(8)
            E0_y_r = struct.unpack("d", E0_y_r_bytes)[0]
            
            E0_y_i_bytes = file.read(8)
            E0_y_i = struct.unpack("d", E0_y_i_bytes)[0]
            
            E0_x = complex(E0_x_r, E0_x_i)
            E0_y = complex(E0_y_r, E0_y_i)
            
            self.E0 = np.array((E0_x, E0_y))
            
            k_x_r_bytes = file.read(8)
            k_x_r = struct.unpack("d", k_x_r_bytes)[0]
            
            k_x_i_bytes = file.read(8)
            k_x_i = struct.unpack("d", k_x_i_bytes)[0]
            
            k_y_r_bytes = file.read(8)
            k_y_r = struct.unpack("d", k_y_r_bytes)[0]
            
            k_y_i_bytes = file.read(8)
            k_y_i = struct.unpack("d", k_y_i_bytes)[0]
            
            k_z_r_bytes = file.read(8)
            k_z_r = struct.unpack("d", k_z_r_bytes)[0]
            
            k_z_i_bytes = file.read(8)
            k_z_i = struct.unpack("d", k_z_i_bytes)[0]
            
            k_x = complex(k_x_r, k_x_i)
            k_y = complex(k_y_r, k_y_i)
            k_z = complex(k_z_r, k_z_i)
            
            self.k = np.array((k_x, k_y, k_z))

class heterogeneous_layer:
    def read(self, file, coefficients, precision=8):
        self.beta = np.fromfile(file, np.complex128, coefficients * 4)
        self.gamma = np.fromfile(file, np.complex128, coefficients * 4)
        self.gg = np.fromfile(file, np.complex128, (coefficients * 4)**2)
        self.ft_n2 = np.fromfile(file, np.complex128, coefficients)

class coupledwave:
    def load(self, filename):
        
        # open the file in binary mode
        f = open(filename, mode="rb")
        
        # load the flag for a heterogeneous sample
        is_volume_bytes = f.read(1)
        self.is_volume = struct.unpack("?", is_volume_bytes)[0]
        
        # load the precision for the coupled wave file
        precision_bytes = f.read(8)
        self.precision = struct.unpack("Q", precision_bytes)[0]
        
        # load the number of incident plane waves
        Pi_bytes = f.read(8)
        Pi = struct.unpack("Q", Pi_bytes)[0]
        
        # load all of the incident plane waves
        self.Pi = []
        for pi in range(Pi):
            p = planewave()
            p.read(f, self.precision)
            self.Pi.append(p)
            
        # load the number of boundaries
        num_boundaries_bytes = f.read(8)
        num_boundaries = struct.unpack("Q", num_boundaries_bytes)[0]
        
        # initialize the arrays that store boundary positions and waves
        self.z = []
        self.Ri = []
        self.Ti = []
        
        # read the data for each boundary
        for bi in range(num_boundaries):
            
            # read the boundary position
            z_bytes = f.read(8)
            z = struct.unpack("d", z_bytes)[0]
            self.z.append(z)
            
            # read the number of reflected waves
            Ri_bytes = f.read(8)
            Ri = struct.unpack("Q", Ri_bytes)[0]
            
            # read all of the reflected waves
            Ri_list = []
            for ri in range(Ri):
                p = planewave()
                p.read(f, self.precision)
                Ri_list.append(p)
            self.Ri.append(Ri_list)
            
            # read the number of transmitted waves
            Ti_bytes = f.read(8)
            Ti = struct.unpack("Q", Ti_bytes)[0]
            
            # read all of the reflected waves
            Ti_list = []
            for ti in range(Ti):
                p = planewave()
                p.read(f, self.precision)
                Ti_list.append(p)
            self.Ti.append(Ti_list)
            
        if self.is_volume is True:
            
            # load the number of Fourier coefficients
            M0_bytes = f.read(4)
            M0 = struct.unpack("i", M0_bytes)[0]
            M1_bytes = f.read(4)
            M1 = struct.unpack("i", M1_bytes)[0]
            self.M = np.array((M0, M1))
            
            # load the sample size
            D0_bytes = f.read(8)
            D0 = struct.unpack("d", D0_bytes)[0]
            D1_bytes = f.read(8)
            D1 = struct.unpack("d", D1_bytes)[0]
            D2_bytes = f.read(8)
            D2 = struct.unpack("d", D2_bytes)[0]
            self.D = np.array((D0, D1, D2))
            
            # load the number of layers
            num_layers_bytes = f.read(4)
            num_layers = struct.unpack("I", num_layers_bytes)[0]
            
            self.layers = []
            for li in range(num_layers):
                l = heterogeneous_layer()
                l.read(f, self.M[0] * self.M[1], self.precision)
                self.layers.append(l)
                
    # returns the refractive index of the volume at each point
    def getVolume(self):
        RI = np.zeros((len(self.layers), self.M[1], self.M[0]), dtype=np.complex128)
        
        for li in range(len(self.layers)):
            fourier_image = np.reshape(self.layers[li].ft_n2, (self.M[1], self.M[0]))
            image = np.fft.ifft2(np.fft.ifftshift(fourier_image)) * self.M[0] * self.M[1]
            RI[li, :, :] = 1.0 / np.sqrt(image)

        return RI

layer = coupledwave()
layer.load("C:/Users/david/Documents/build/scattersim-bld/simple_layer.cw")

volume = coupledwave()
volume.load("C:/Users/david/Documents/build/scattersim-bld/simple_volume.cw")


