import numpy as np
import scipy as sp

import spectral as spy
import spectral.io.envi as envi      

import matplotlib as mpl
from matplotlib import pyplot as plt

import os.path
from os import path

import math

import cv2 
 
## !-- need to install wxpython with : conda install -c anaconda wxpython --!
## !-- need to install pyopengl --!
## !-- need the modified spectral/clustering.py script --!


def save_envi_image(Spec, filename)  :
    md = {'lines'  : Spec.shape[0],      
          'samples': Spec.shape[1],                                             ## pixel count per line
          'bands'  : Spec.shape[2],                                             ## wavelength count
          'data type': 5 }                                                      ##5 =dble precision, 4= float, 12 = uint16
    envi.save_image(filename+'.hdr', Spec, metadata = md, force=True )
    return

def spectrum_patch(spectrum, patch):
    origin_x = patch[1]
    origin_y = patch[0]
    width    = patch[3]
    height   = patch[2]
    return spectrum[origin_x: origin_x + width, origin_y: origin_y + height,:]

def zoom_patch(image_view, patch, title=''):
    origin_x = patch[1]
    origin_y = patch[0]
    center_x     = (patch[0] + patch[2])//2
    center_y     = (patch[1] + patch[3])//2
    patch_center = np.array([center_x, center_y])
    width        = patch[3]
    height       = patch[2]
    
    sliced_data = image_view.data_rgb[origin_x:origin_x+width,origin_y:origin_y+height,:]

    #print(_patch_view)
    return sliced_data
    
def colorMatchFcn():
    cmf = np.array([400,2.214302E-02,2.452194E-03,1.096090E-01,
                    402,3.043036E-02,3.299115E-03,1.512047E-01,
                    404,4.109640E-02,4.352768E-03,2.049517E-01,
                    406,5.447394E-02,5.661014E-03,2.725123E-01,
                    408,7.070048E-02,7.250312E-03,3.547064E-01,
                    410,8.953803E-02,9.079860E-03,4.508369E-01,
                    412,1.104019E-01,1.106456E-02,5.586361E-01,
                    414,1.328741E-01,1.318014E-02,6.760982E-01,
                    416,1.566468E-01,1.545004E-02,8.013019E-01,
                    418,1.808328E-01,1.785302E-02,9.295791E-01,
                    420,2.035729E-01,2.027369E-02,1.051821E+00,
                    422,2.231348E-01,2.260041E-02,1.159527E+00,
                    424,2.403892E-01,2.491247E-02,1.256834E+00,                    
                    426,2.575896E-01,2.739923E-02,1.354758E+00,
                    428,2.753532E-01,3.016909E-02,1.456414E+00,
                    430,2.918246E-01,3.319038E-02,1.552826E+00,
                    432,3.052993E-01,3.641495E-02,1.635768E+00,
                    434,3.169047E-01,3.981843E-02,1.710604E+00,
                    436,3.288194E-01,4.337098E-02,1.787504E+00,
                    438,3.405452E-01,4.695420E-02,1.863108E+00,
                    440,3.482554E-01,5.033657E-02,1.917479E+00,
                    442,3.489075E-01,5.332218E-02,1.934819E+00,
                    444,3.446705E-01,5.606335E-02,1.926395E+00,
                    446,3.390240E-01,5.885107E-02,1.910430E+00,
                    448,3.324276E-01,6.178644E-02,1.889000E+00,
                    450,3.224637E-01,6.472352E-02,1.848545E+00,
                    452,3.078201E-01,6.757256E-02,1.781627E+00,
                    454,2.909776E-01,7.063280E-02,1.702749E+00,
                    456,2.747962E-01,7.435960E-02,1.629207E+00,
                    458,2.605847E-01,7.911436E-02,1.568896E+00,
                    460,2.485254E-01,8.514816E-02,1.522157E+00,
                    462,2.383414E-01,9.266008E-02,1.486673E+00,
                    464,2.279619E-01,1.013746E-01,1.450709E+00,
                    466,2.151735E-01,1.107377E-01,1.401587E+00,
                    468,1.992183E-01,1.203122E-01,1.334220E+00,
                    470,1.806905E-01,1.298957E-01,1.250610E+00,
                    472,1.604471E-01,1.393309E-01,1.154316E+00,
                    474,1.395705E-01,1.487372E-01,1.051347E+00,
                    476,1.189859E-01,1.583644E-01,9.473958E-01,
                    478,9.951424E-02,1.683761E-01,8.473981E-01,
                    480,8.182895E-02,1.788048E-01,7.552379E-01,
                    482,6.619477E-02,1.896559E-01,6.725198E-01,
                    484,5.234242E-02,2.008259E-01,5.972433E-01,
                    486,4.006154E-02,2.121826E-01,5.274921E-01,
                    488,2.949091E-02,2.241586E-01,4.642586E-01,
                    490,2.083981E-02,2.379160E-01,4.099313E-01,
                    492,1.407924E-02,2.546023E-01,3.650566E-01,
                    494,9.019658E-03,2.742490E-01,3.274095E-01,
                    496,5.571145E-03,2.964837E-01,2.948102E-01,
                    498,3.516303E-03,3.211393E-01,2.654100E-01,
                    500,2.461588E-03,3.483536E-01,2.376753E-01,
                    502,2.149559E-03,3.782275E-01,2.107484E-01,
                    504,2.818931E-03,4.106582E-01,1.846574E-01,
                    506,4.891359E-03,4.453993E-01,1.596918E-01,
                    508,8.942902E-03,4.821376E-01,1.369428E-01,
                    510,1.556989E-02,5.204972E-01,1.176796E-01,
                    512,2.504698E-02,5.600208E-01,1.020943E-01,
                    514,3.674999E-02,6.003172E-01,8.890075E-02,
                    516,4.978584E-02,6.409398E-01,7.700982E-02,
                    518,6.391651E-02,6.808134E-01,6.615436E-02,
                    520,7.962917E-02,7.180890E-01,5.650407E-02,
                    522,9.726978E-02,7.511821E-01,4.809566E-02,
                    524,1.166192E-01,7.807352E-01,4.079734E-02,
                    526,1.374060E-01,8.082074E-01,3.446846E-02,
                    528,1.593076E-01,8.340701E-01,2.901901E-02,
                    530,1.818026E-01,8.575799E-01,2.438164E-02,
                    532,2.045085E-01,8.783061E-01,2.046415E-02,
                    534,2.280650E-01,8.975211E-01,1.713788E-02,
                    536,2.535441E-01,9.169947E-01,1.429644E-02,
                    538,2.811351E-01,9.366731E-01,1.187897E-02,
                    540,3.098117E-01,9.544675E-01,9.846470E-03,
                    542,3.384319E-01,9.684390E-01,8.152811E-03,
                    544,3.665839E-01,9.781519E-01,6.744115E-03,
                    546,3.940988E-01,9.836669E-01,5.572778E-03,
                    548,4.213484E-01,9.863813E-01,4.599169E-03,
                    550,4.494206E-01,9.890228E-01,3.790291E-03,
                    552,4.794395E-01,9.934913E-01,3.119341E-03,
                    554,5.114395E-01,9.980205E-01,2.565722E-03,
                    556,5.448696E-01,9.999930E-01,2.111280E-03,
                    558,5.790137E-01,9.989839E-01,1.738589E-03,
                    560,6.133784E-01,9.967737E-01,1.432128E-03,
                    562,6.479223E-01,9.947115E-01,1.179667E-03,
                    564,6.833782E-01,9.921156E-01,9.718623E-04,
                    566,7.204110E-01,9.878596E-01,8.010231E-04,
                    568,7.586285E-01,9.815036E-01,6.606347E-04,
                    570,7.967750E-01,9.732611E-01,5.452416E-04,
                    572,8.337389E-01,9.631369E-01,4.503642E-04,
                    574,8.687862E-01,9.502540E-01,3.723345E-04,
                    576,9.011588E-01,9.336897E-01,3.081396E-04,
                    578,9.318245E-01,9.146707E-01,2.552996E-04,
                    580,9.638388E-01,8.963613E-01,2.117772E-04,
                    582,9.992953E-01,8.808462E-01,1.759024E-04,
                    584,1.034790E+00,8.663755E-01,1.463059E-04,
                    586,1.065522E+00,8.504295E-01,1.218660E-04,
                    588,1.089944E+00,8.320109E-01,1.016634E-04,
                    590,1.109767E+00,8.115868E-01,8.494468E-05,
                    592,1.126266E+00,7.896515E-01,7.109247E-05,
                    594,1.138952E+00,7.664733E-01,5.960061E-05,
                    596,1.147095E+00,7.422473E-01,5.005417E-05,
                    598,1.150838E+00,7.172525E-01,4.211268E-05,
                    600,1.151033E+00,6.918553E-01,3.549661E-05,
                    602,1.148061E+00,6.662846E-01,2.997643E-05,
                    604,1.140622E+00,6.402807E-01,2.536339E-05,
                    606,1.127298E+00,6.135148E-01,2.150221E-05,
                    608,1.108033E+00,5.860682E-01,1.826500E-05,
                    610,1.083928E+00,5.583746E-01,1.554631E-05,
                    612,1.055934E+00,5.307673E-01,1.325915E-05,
                    614,1.024385E+00,5.032889E-01,1.133169E-05,
                    616,9.895268E-01,4.759442E-01,0.000000E+00,
                    618,9.523257E-01,4.490154E-01,0.000000E+00,
                    620,9.142877E-01,4.229897E-01,0.000000E+00,
                    622,8.760157E-01,3.980356E-01,0.000000E+00,
                    624,8.354235E-01,3.733907E-01,0.000000E+00,
                    626,7.904565E-01,3.482860E-01,0.000000E+00,
                    628,7.418777E-01,3.228963E-01,0.000000E+00,
                    630,6.924717E-01,2.980865E-01,0.000000E+00,
                    632,6.442697E-01,2.744822E-01,0.000000E+00,
                    634,5.979243E-01,2.522628E-01,0.000000E+00,
                    636,5.537296E-01,2.314809E-01,0.000000E+00,
                    638,5.120218E-01,2.121622E-01,0.000000E+00,
                    640,4.731224E-01,1.943124E-01,0.000000E+00,
                    642,4.368719E-01,1.778274E-01,0.000000E+00,
                    644,4.018980E-01,1.622841E-01,0.000000E+00,
                    646,3.670592E-01,1.473081E-01,0.000000E+00,
                    648,3.326305E-01,1.329013E-01,0.000000E+00,
                    650,2.997374E-01,1.193120E-01,0.000000E+00,
                    652,2.691053E-01,1.067113E-01,0.000000E+00,
                    654,2.409319E-01,9.516653E-02,0.000000E+00,
                    656,2.152431E-01,8.469044E-02,0.000000E+00,
                    658,1.919276E-01,7.523372E-02,0.000000E+00,
                    660,1.707914E-01,6.671045E-02,0.000000E+00,
                    662,1.516577E-01,5.904179E-02,0.000000E+00,
                    664,1.343737E-01,5.216139E-02,0.000000E+00,
                    666,1.187979E-01,4.600578E-02,0.000000E+00,
                    668,1.047975E-01,4.050755E-02,0.000000E+00,
                    670,9.224597E-02,3.559982E-02,0.000000E+00,
                    672,8.101986E-02,3.122332E-02,0.000000E+00,
                    674,7.099633E-02,2.732601E-02,0.000000E+00,
                    676,6.206225E-02,2.386121E-02,0.000000E+00,
                    678,5.412533E-02,2.079020E-02,0.000000E+00,
                    680,4.710606E-02,1.807939E-02,0.000000E+00,
                    682,4.091411E-02,1.569188E-02,0.000000E+00,
                    684,3.543034E-02,1.358062E-02,0.000000E+00,
                    686,3.055672E-02,1.170696E-02,0.000000E+00,
                    688,2.628033E-02,1.006476E-02,0.000000E+00,
                    690,2.262306E-02,8.661284E-03,0.000000E+00,
                    692,1.954647E-02,7.481130E-03,0.000000E+00,
                    694,1.692727E-02,6.477070E-03,0.000000E+00,
                    696,1.465854E-02,5.608169E-03,0.000000E+00,
                    698,1.268205E-02,4.851785E-03,0.000000E+00,
                    700,1.096778E-02,4.195941E-03,0.000000E+00,
                    702,9.484317E-03,3.628371E-03,0.000000E+00,
                    704,8.192921E-03,3.134315E-03,0.000000E+00,
                    705,7.608750E-03,2.910864E-03,0.000000E+00,
                    706,7.061391E-03,2.701528E-03,0.000000E+00,
                    708,6.071970E-03,2.323231E-03,0.000000E+00,
                    710,5.214608E-03,1.995557E-03,0.000000E+00,
                    712,4.477579E-03,1.713976E-03,0.000000E+00,
                    714,3.847988E-03,1.473453E-03,0.000000E+00,
                    716,3.312857E-03,1.268954E-03,0.000000E+00,
                    718,2.856894E-03,1.094644E-03,0.000000E+00,
                    720,2.464821E-03,9.447269E-04,0.000000E+00,
                    722,2.125694E-03,8.150438E-04,0.000000E+00,
                    724,1.833723E-03,7.033755E-04,0.000000E+00,
                    726,1.583904E-03,6.078048E-04,0.000000E+00,
                    728,1.370151E-03,5.260046E-04,0.000000E+00,
                    730,1.186238E-03,4.555970E-04,0.000000E+00,
                    732,1.027194E-03,3.946860E-04,0.000000E+00,
                    734,8.891262E-04,3.417941E-04,0.000000E+00,
                    736,7.689351E-04,2.957441E-04,0.000000E+00,
                    738,6.648590E-04,2.558640E-04,0.000000E+00,
                    740,5.758303E-04,2.217445E-04,0.000000E+00,
                    742,5.001842E-04,1.927474E-04,0.000000E+00,
                    744,4.351386E-04,1.678023E-04,0.000000E+00,
                    746,3.783733E-04,1.460168E-04,0.000000E+00,
                    748,3.287199E-04,1.269451E-04,0.000000E+00,
                    750,2.856577E-04,1.103928E-04,0.000000E+00,
                    752,2.485462E-04,9.611836E-05,0.000000E+00,
                    754,2.165300E-04,8.379694E-05,0.000000E+00,
                    756,1.888338E-04,7.313312E-05,0.000000E+00,
                    758,1.647895E-04,6.387035E-05,0.000000E+00,
                    760,1.438270E-04,5.578862E-05,0.000000E+00,
                    762,1.255141E-04,4.872179E-05,0.000000E+00,
                    764,1.095983E-04,4.257443E-05,0.000000E+00,
                    766,9.584715E-05,3.725877E-05,0.000000E+00,
                    768,8.392734E-05,3.264765E-05,0.000000E+00,
                    770,7.347551E-05,2.860175E-05,0.000000E+00,
                    772,6.425257E-05,2.502943E-05,0.000000E+00,
                    774,5.620098E-05,2.190914E-05,0.000000E+00,
                    776,4.926279E-05,1.921902E-05,0.000000E+00,
                    778,4.328212E-05,1.689899E-05,0.000000E+00,
                    780,3.806114E-05,1.487243E-05,0.000000E+00,
                    782,3.346023E-05,1.308528E-05,0.000000E+00,
                    784,2.941371E-05,1.151233E-05,0.000000E+00,
                    786,2.586951E-05,1.013364E-05,0.000000E+00,
                    788,2.276639E-05,8.925630E-06,0.000000E+00,
                    790,2.004122E-05,7.863920E-06,0.000000E+00,
                    792,1.764358E-05,6.929096E-06,0.000000E+00,
                    794,1.553939E-05,6.108221E-06,0.000000E+00,
                    796,1.369853E-05,5.389831E-06,0.000000E+00,
                    798,1.208947E-05,4.761667E-06,0.000000E+00,
                    800,1.068141E-05,4.211597E-06,0.000000E+00])
    return cmf
    
def XYZ_to_sRGB():
    M = np.array([0.4124564,  0.3575761,  0.1804375,
                  0.2126729,  0.7151522,  0.0721750,
                  0.0193339,  0.1191920, 0.9503041  ])

    M = M.reshape(3,3)
    _M = np.linalg.inv(M) 
    return _M

def spec_to_XYZ(spectrum, patch):
    cmf        = colorMatchFcn()
    spec_patch = spectrum_patch(spectrum, patch)
    nrows      = spec_patch.shape[0]
    ncolumns   = spec_patch.shape[1]
    XYZ_patch  = np.zeros((nrows,ncolumns,3))
    delta_lambda = 2
    for i in range(nrows):
        for j in range(ncolumns):
            N = 0
            for w in range(201):
                _lambda = spec_patch[i,j,w] 
                XYZ_patch[i,j,0] += delta_lambda*_lambda*cmf[w*4+1]
                XYZ_patch[i,j,1] += delta_lambda*_lambda*cmf[w*4+2]
                XYZ_patch[i,j,2] += delta_lambda*_lambda*cmf[w*4+3]
                N += delta_lambda*cmf[w*4+2]
            XYZ_patch[i,j,0] /= N 
            XYZ_patch[i,j,1] /= N
            XYZ_patch[i,j,2] /= N            
    return XYZ_patch

def spec_to_sRGB(spectrum, patch):
    XYZ_patch = spec_to_XYZ(spectrum,patch)
    XYZ = np.zeros((3,1))
    nrows      = XYZ_patch.shape[0]
    ncolumns   = XYZ_patch.shape[1]
    image_rgb  = np.zeros((nrows,ncolumns,3))
    _M         = XYZ_to_sRGB()
    for i in range(nrows):
        for j in range(ncolumns):
          XYZ[0,0] = XYZ_patch[i,j,0] 
          XYZ[1,0] = XYZ_patch[i,j,1] 
          XYZ[2,0] = XYZ_patch[i,j,2]          
          XYZ = XYZ.reshape(3,1)
          RGB = np.dot(_M,XYZ)          
          image_rgb[i,j,0]  = RGB[0]
          image_rgb[i,j,1]  = RGB[1]
          image_rgb[i,j,2]  = RGB[2]
          ## Correction gamma
          for w in range(3):
              image_rgb[i,j,w] = pow(image_rgb[i,j,w], 1/2.2)
              if(image_rgb[i,j,w] <= 0.0031308):
                 image_rgb[i,j,w] *= 12.92
              else:
                 image_rgb[i,j,w] = pow(image_rgb[i,j,w], 1/2.4)
                 image_rgb[i,j,w] *= 1.055
                 image_rgb[i,j,w] -= 0.055
    return image_rgb

def get_white_XYZ(spectrum):
    patch = np.array([188, 455, 3, 4])
    xyz_white_patch = spec_to_XYZ(spectrum,patch)
    nrows      = xyz_white_patch.shape[0]
    ncolumns   = xyz_white_patch.shape[1]
    _x = _y = _z =0
    for i in range(nrows):
        for j in range(ncolumns):
            _x += xyz_white_patch[i,j,0]
            _y += xyz_white_patch[i,j,1]
            _z += xyz_white_patch[i,j,2]
    _x /= (ncolumns * nrows)
    _y /= (ncolumns * nrows)
    _z /= (ncolumns * nrows)
    XYZ = np.array([_x, _y, _z])
    return XYZ.reshape((3,1))

def spec_to_Lab(spectrum, patch):
    XYZ_patch = spec_to_XYZ(spectrum, patch)
    print(XYZ_patch.shape)
    XYZ       = np.zeros((3,1))
    LAB       = np.zeros((3,1))
    nrows      = XYZ_patch.shape[0]
    ncolumns   = XYZ_patch.shape[1]
    White_xyz = get_white_XYZ(spectrum)
    image_LAB  = np.zeros((nrows,ncolumns,3))
    print(image_LAB.shape)
    x = y = z = 0
    e = 216/24389
    k = 24389/27
    for i in range(nrows):
        for j in range(ncolumns):
            x = XYZ_patch[i,j,0]/0.33
            y = XYZ_patch[i,j,1]/0.33
            z = XYZ_patch[i,j,2]/0.33
            
            f_x = f_y = f_z = 0
            
            if(x>e):
                f_x = pow(x,3/2)
            else:
                f_x = (k * x + 16)/116
            
            if(y>e):
                f_y = pow(y,3/2)
            else:
                f_y = (k * y + 16)/116
                
            if(z>e):
                f_z = pow(z,3/2)
            else:
                f_z = (k * z + 16)/116
            
            L = 116 * f_y - 16
            a = 500 * (f_x - f_y)
            b = 200 * (f_y - f_z)
            
            image_LAB[i,j,0] = L
            image_LAB[i,j,1] = a
            image_LAB[i,j,2] = b
            
    return image_LAB

##------------- kubelka munk helper functions 

def saunderson_correction_inverse(r_m, k_1, k_2=0.4):
    # r_m : la reflectance mesurée 
    # k_1 : the first constant
    # k_2 : between 0.4 and 0
    r_inf = (r_m - k_1) / (1 - k_1 - k_2 + k_2 * r_m)	
    return r_inf

def saunderson_correction(r_inf, k_1, k_2=0.4):
    r_m = k_1 + ((1-k_1)*(1-k_2)*r_inf)/(1-(k_2*r_inf))
    return r_m

def k_1_from_refraction_indices(n1, n2):
    #n1		The index of refraction of the medium through which light arrives.  In
    #		most cases, this medium will be air, so n1 should be very near 1.

	#n2		The index of refraction of the medium at which light arrives.  In
	#		most cases, this medium will be a film of paint (1.5).
    k_1 = pow(((n2-n1)/(n1+n2)),2)	;
    return k_1

def k_over_s_from_masstone_r(r):
    K_over_S = 0
    
    if r == 0:		                    # No light at all is reflected (avoid a division by 0)
        K_over_S = math.inf				
    else:			                    # Generic case, where some light is reflected
        K_over_S = pow((1-r),2)/(2*r)	
    return K_over_S

def masstone_r_from_k_and_s(k,s):
    if(s==0):
        return 0
    else:
        return 1 + (k/s) - math.sqrt(pow((k/s),2) + (2 * k/s))
        

if __name__ == '__main__':    
    file_path = 'C:/Users/Asus/.spyder-py3/'                ## should set the file path
                                    
    filename  = file_path + 'cube_envi32_Reflectance_Spat_3D_BinomFilt7.hdr'
    rgb_image_filename = 'rgb_reconstruction.png'
    xyz_image_filename = 'xyz_reconstruction.png'
    lab_image_filename = 'lab_reconstruction.png'
    
    rgb_image_path     = file_path + rgb_image_filename
    xyz_image_path     = file_path + xyz_image_filename
    lab_image_path     = file_path + lab_image_filename
    
    found_rgb = path.exists(rgb_image_path)   
    found_xyz = path.exists(xyz_image_path)   
    found_lab = path.exists(lab_image_path)   
    
    img       = spy.envi.open(filename)

    if found_rgb :
        rgb_image = plt.imread(rgb_image_path)   
    if found_xyz : 
        xyz_image = plt.imread(xyz_image_path)
    if found_lab : 
        lab_image = plt.imread(lab_image_path)
       
    plt.close('all')
    
    NbLg, NbCol, NbWaves = img.nrows, img.ncols, img.nbands
    Wavelengths = np.array(img.bands.centers)
    SpecImg= np.array(img.load(), np.float32)
    
    """ ----------------------------------------------------------------------
    
    
    spy.imshow(img,title="l'image en fausses couleurs")
        
    lg=30
    col =10
    
    Spectrum = SpecImg[ lg, col , :]
    plt.figure(3)
    plt.plot(Wavelengths,Spectrum, linewidth = 3, label ="Spectrum at " + str(lg) + " x " +str(col))
    plt.xlabel("Wavelenghts (nm)")
    plt.ylabel("Radiance")
    plt.title("A spectrum")
    plt.legend()
    
    title = 'Reconstruction en rgb spectre filtrés en 700nm, 536nm et 436nm'
    
    rgb_reconstruction_view = spy.imshow(img, (150, 68, 18), title=title)
    
    #--------------- Patches selection     
    ## origin_y, origin_x, width, height
    patches = np.array([0,       0,       968,   608,
                        464,     306,     12,    12,
                        521,     203,     12,    20,     
                        74,      231,     17,    23,     
                        221,     426,     6,     6,      
                        607,     184,     27,    10,     
                        644,     495,     12,    12,     
                        750,     393,     12,    12,     
                        174,     392,     30,    10])
    
    patches_description = np.array([ "l'intégralité de l'image reconstruite",
                                     "jaune | la texture du canevas ??",
                                     "jaune coolde l'église",
                                     "2 jaune bleu des montagnes",
                                     "rouge vif tir canon",   
                                     "rouge orangé (mélange ?) toit de la porte",
                                     "rouge orangé petite maison en bas",
                                     "rouge noirci",
                                     "rouge"
                                    ])
    
    patches    = patches.reshape((len(patches)//4,4))
    
    
    #--------------- Save the image in the different spaces, save computation time 
    if found_lab==False:
        first_lab_patch = spec_to_Lab(SpecImg,patches[0])
        spy.save_rgb(lab_image_filename,first_lab_patch, format='png')
        lab_image      = plt.imread(lab_image_path)
    
    if found_xyz==False:
        _xyz_reconstruction = spec_to_XYZ(SpecImg, patches[0])
        spy.save_rgb(xyz_image_filename, _xyz_reconstruction, format='png')
        xyz_image      = plt.imread(xyz_image_path) 
       
    if found_rgb == False:
        _rgb_reconstruction = spec_to_sRGB(SpecImg, patches[0])
        spy.save_rgb(rgb_image_filename, _rgb_reconstruction, format='png')
        rgb_image      = plt.imread(rgb_image_path)
        
    
    #---------------- Reconstruction RGB,XYZ and LAB-----------------------
    _rgb_reconstruction_view = spy.imshow(rgb_image, title = patches_description[0] + ' | sRGB', interpolation='none')
    fig, (ax1,ax2) = plt.subplots(2)
    _xys_reconstruction_view = ax1.imshow(xyz_image, interpolation='none')
    ax1.set_title(patches_description[0] + ' | CIE XYZ')
    _lab_reconstruction_view = ax2.imshow(lab_image, interpolation='none')
    ax2.set_title(patches_description[0] + ' | CIE Lab')
    fig.tight_layout()
    plt.show()
    
    first_RGB_Patch = np.array([])
    first_spec_Patch = np.array([])

    #---------------- Showing the different patches -----------------------
    (fig, axes) = plt.subplots(2,(len(patches)-1)//2)
    
    j = 1
    
    for i in range(1,len(patches),1):
        patch = patches[i]
        spec_patch = spectrum_patch(SpecImg,patch) 
        #view_spec_patch = np.array(spec_patch, copy=True)  
        #spy.imshow(view_spec_patch)
        
        #XYZ_patch = spec_to_XYZ(SpecImg, patch)
        #spy.imshow(XYZ_patch, title = patches_description[i] + ' | CIE XYZ', interpolation='none')
        
        RGB_Patch_view = zoom_patch(_rgb_reconstruction_view,patch, title = patches_description[i])
        
        axes[(i-1)%2,(j-1)%4].imshow(RGB_Patch_view)
        #axes[(i-1)%2,(j-1)%4].set_title(patches_description[i])        
        
        if((i-1)%2!=0):
            j += 1
        
        if(i==1):
            first_spec_Patch = np.array(spec_patch)
            first_RGB_Patch = np.array(RGB_Patch_view)
            XYZ_patch = spec_to_XYZ(SpecImg, patch)
        #spy.save_rgb()
        #spy.imshow(RGB_Patch, title = patches_description[i] + ' | CIE RGB', interpolation='none')
    fig.tight_layout()
    plt.show()
    
    #----------------------- Clusterings of the whole image --------------
    #--------------The lab image 
    first_RGB_Patch = zoom_patch(_rgb_reconstruction_view,patches[0])
    
    first_lab_patch = lab_image
    ##spy.save_rgb(lab_image_filename,first_lab_patch, format='png')
        
    (m, c) = spy.kmeans(first_lab_patch, 5, 20, distance='L1')
    (_m, _c) = spy.kmeans(first_lab_patch, 5, 20, distance='L2')
    
    spy.imshow(first_lab_patch)
    lab_patch_view = spy.imshow(first_RGB_Patch, classes= m, interpolation='nearest', stretch_all=False)
    lab_patch_view.set_display_mode('overlay')
    lab_patch_view.class_alpha = 1
    
    rgb_patch_view = spy.imshow(first_RGB_Patch, classes= _m, interpolation='nearest', stretch_all=False)
    rgb_patch_view.set_display_mode('overlay')
    rgb_patch_view.class_alpha = 1
    
    plt.figure()
    for i in range(c.shape[0]):
        plt.plot(c[i])
    plt.grid()
    
    plt.figure()
    for i in range(_c.shape[0]):
        plt.plot(_c[i])
    plt.grid()   
    
    #--------------The rgb image 
    (m, c) = spy.kmeans(rgb_image, 4, 5, distance='L1')
    (_m, _c) = spy.kmeans(rgb_image, 4, 5, distance='L2')
    
    spy.imshow(rgb_image)
    lab_patch_view = spy.imshow(first_RGB_Patch, classes= m, interpolation='nearest', stretch_all=False)
    lab_patch_view.set_display_mode('overlay')
    lab_patch_view.class_alpha = 1
    
    rgb_patch_view = spy.imshow(first_RGB_Patch, classes= _m, interpolation='nearest', stretch_all=False)
    rgb_patch_view.set_display_mode('overlay')
    rgb_patch_view.class_alpha = 1
    
    plt.figure()
    for i in range(c.shape[0]):
        plt.plot(c[i])
    plt.grid()
    
    plt.figure()
    for i in range(_c.shape[0]):
        plt.plot(_c[i])
    plt.grid()
    
    #--------------The spectral image
    (m, c) = spy.kmeans(SpecImg[:,:,150:190], 5, 7, distance='L1')
    (_m, _c) = spy.kmeans(SpecImg[:,:,150:190], 5, 7, distance='L2')
    
    spy.imshow(rgb_image)
    lab_patch_view = spy.imshow(first_RGB_Patch, classes= m, interpolation='nearest', stretch_all=False)
    lab_patch_view.set_display_mode('overlay')
    lab_patch_view.class_alpha = 1
    
    rgb_patch_view = spy.imshow(first_RGB_Patch, classes= _m, interpolation='nearest', stretch_all=False)
    rgb_patch_view.set_display_mode('overlay')
    rgb_patch_view.class_alpha = 1
    
    
    plt.figure()
    for i in range(c.shape[0]):
        plt.plot(c[i])
    plt.grid()
    
    plt.figure()
    for i in range(_c.shape[0]):
        plt.plot(_c[i])
    plt.grid()
    
    ----------------------------------------------------------------------"""
    
    
    #--------------Kubelka Munk model
    # the white pixel on the canon is on the 456,189. We consider this pigment to be the white 
    s_lambda_white    = 1    
    
    plt.imshow(rgb_image[456:457,189:190,:])
    white_pixel_spec  = np.array(SpecImg[456,189,:]) 
    
    r_m               = np.array(white_pixel_spec) 
    k_1               = k_1_from_refraction_indices(1,1.5)
    
    r_inf             = np.zeros(NbWaves)
    k_lambda_white    = np.zeros(NbWaves)
    
    for i in range(NbWaves):
        r_inf[i]          = saunderson_correction_inverse(r_m[i],k_1)
        k_lambda_white[i] = k_over_s_from_masstone_r(r_inf[i])

    
    plt.figure()
    plt.plot(Wavelengths,np.log(k_lambda_white))
    plt.xlabel("Wavelength in nm")
    plt.ylabel("log(K/S)")
    plt.title("K/S ratio for the white paint")
    plt.show()
    
    
    ##-------------- Masstones ------------------
    
    # the yellow pixel masseton
    plt.imshow(rgb_image[316:317,261:262,:])
    yellow_pixel_spec  = np.array(SpecImg[316,261,:]) 
    k_over_s_yellow    = np.zeros(NbWaves)
    r_m                = np.array(yellow_pixel_spec)
    
    for i in range(r_m.shape[0]):
        r_inf[i]          = saunderson_correction_inverse(r_m[i],k_1)
        k_over_s_yellow[i]    = k_over_s_from_masstone_r(r_inf[i])
    
    plt.figure()
    plt.plot(Wavelengths,np.log(k_over_s_yellow))
    plt.xlabel("Wavelength in nm")
    plt.ylabel("log(K/S)")
    plt.title("K/S ratio for the yellow masstone")
    plt.show()  
    
    # the red pixel masseton
    plt.imshow(rgb_image[404:405,751:752,:])
    red_pixel_spec  = np.array(SpecImg[404,751,:]) 
    k_over_s_red       = np.zeros(NbWaves)
    r_m                = np.array(red_pixel_spec)
    
    for i in range(r_m.shape[0]):
        r_inf[i]          = saunderson_correction_inverse(r_m[i],k_1)
        k_over_s_red[i]   = k_over_s_from_masstone_r(r_inf[i])
    
    plt.figure()
    plt.plot(Wavelengths,np.log(k_over_s_red))
    plt.xlabel("Wavelength in nm")
    plt.ylabel("log(K/S)")
    plt.title("K/S ratio for the red masstone")
    plt.show()  
    
    plt.figure()
    plt.plot(Wavelengths,np.log(k_over_s_red), 'r')
    plt.plot(Wavelengths,np.log(k_lambda_white), 'w')
    plt.plot(Wavelengths,np.log(k_over_s_yellow),'y')
    ax = plt.gca()
    ax.set_facecolor((0.8,0.8,0.8))
    plt.xlabel("Wavelength in nm")
    plt.ylabel("log(K/S)")
    plt.title("K/S ratio for the red and white")
    #plt.legend()
    plt.show()

    plt.figure()
    plt.plot(Wavelengths,SpecImg[316,261,:])
    plt.xlabel("Wavelength in nm")
    plt.ylabel("reflectance")
    plt.title("K/S ratio for the yellow masstone")
    plt.show()  

    ##-------------- Mixture two pigments ------------------
    
    # Suspected yellow and white mix pigments
    # from mixture of yellow and white present on the pixel 316 478
    mixed_pixel_spec           = np.array(SpecImg[316,478,:]) 
    k_over_s_mixture           = np.zeros(NbWaves)
    s_lambda                   = np.zeros(NbWaves)
    k_lambda                   = np.zeros(NbWaves)
    predicted_r_inf            = np.zeros(NbWaves)
    predicted_r_m              = np.zeros(NbWaves)
    EMS_euclidean              = np.zeros(NbWaves)
    EMS                        = np.zeros(99)
    SPECTRAL_ANGLES            = np.zeros(99)
    c                          = np.zeros(99)
    r_m                        = mixed_pixel_spec
    lowest_EMS                 = math.inf
    lowest_SPEC_ANGLE          = math.inf
    lowest_SPEC_ANGLE_concentr = math.inf
    lowest_EMS_concentration   = math.inf
    
    for i in range(r_m.shape[0]):
        r_inf[i]               = saunderson_correction_inverse(r_m[i],k_1)
        k_over_s_mixture[i]    = k_over_s_from_masstone_r(r_inf[i])
    
    plt.figure()
    plt.plot(Wavelengths,np.log(k_over_s_mixture))
    plt.xlabel("Wavelength in nm")
    plt.ylabel("log(K/S)")
    plt.title("K/S ratio for the mixture ")
    plt.show() 
    
    # for each iteration consider a concentration c, starting from 0.1 to 0.9 
    for w in range(1,100,1):
        _E                     = 0
        _c                     = w * 0.01
        c[w-1]                 = _c
        for i in range(NbWaves):
            diff               =  (k_over_s_yellow[i] - k_over_s_mixture[i]) 
            s_lambda[i]        = ((1-_c)/_c) * (k_over_s_mixture[i] * (s_lambda_white - k_lambda_white[i])) / diff
            k_lambda[i]        = s_lambda[i]  * k_over_s_yellow[i]
            predicted_r_inf[i] = masstone_r_from_k_and_s(k_lambda[i],s_lambda[i])
            predicted_r_m[i]   = saunderson_correction(predicted_r_inf[i],k_1)
            EMS_euclidean[i]   = predicted_r_m[i] - SpecImg[316,478,i]
            _E                += pow(EMS_euclidean[i],2)
        _E  /= r_m.shape[0]
        _E  = math.sqrt(_E )
        EMS[w-1] = _E
        
        _predicted_r_m       = predicted_r_m.reshape((1,1,NbWaves))
        angle                = spy.spectral_angles(_predicted_r_m, SpecImg[316,261,:].reshape((1,NbWaves)))
        SPECTRAL_ANGLES[w-1] = angle 
        
        
        if( _E<lowest_EMS ):
            lowest_EMS = _E
            lowest_EMS_concentration = _c
        if( angle<lowest_SPEC_ANGLE):
            lowest_SPEC_ANGLE = angle
            lowest_SPEC_ANGLE_concentr = _c
            
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(c,EMS,label='RMSE')
    ax.plot(c,SPECTRAL_ANGLES,label='Spectral angle')
    plt.xlabel("concentration c")
    plt.ylabel("distance")
    plt.title("K/S ratio for the mixture ")
    plt.legend()
    plt.show() 
    
    
    """
    (fig, axes) = plt.subplots(2,(len(patches)-1)//2)
    j = 1
    
    for i in range(1,len(patches),1):
        patch = patches[i]
        
        zoom_m = zoom_patch(spy.imshow(classes=m), patch)
        
        axes[(i-1)%2,(j-1)%4].imshow(zoom_m)   
            
        if((i-1)%2!=0):
            j += 1
        
    fig.tight_layout()
    plt.show()
    """

    #print(first_lab_patch)
    #pc = spy.principal_components(first_spec_Patch)
    #xdata = pc.transform(first_spec_Patch)
    #w = spy.view_nd(xdata[:,:,:25])
    
        
