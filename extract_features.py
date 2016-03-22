from skimage import img_as_ubyte
from skimage.io import imread
from matplotlib import colors as col
import numpy as np
import pandas as pd


def extractFeatures(file_path):
    #Construct dataframe with features for output
    post_feat = pd.DataFrame(columns=['perc_bla',
                                      'perc_whi',
                                      'perc_r_y',
                                      'perc_y_g',
                                      'perc_g_c',
                                      'perc_c_b',
                                      'perc_b_m',
                                      'perc_m_r',
                                      'mean_sat',
                                      'mean_lig',
                                      'mean_lum',
                                      'contrast',
                                      'low_freq',
                                      'high_freq'])
    nans = np.empty((1,11))        
    post_feat.loc[0] = nans.fill(np.nan)
    
    #Read image file
    img = imread(file_path)
    img_RGB = img_as_ubyte(img)  

    #Transform to luminance/greyscale image
    img_lum = (img_RGB[:,:,0] * 0.2126
           + img_RGB[:,:,1] * 0.7152
           + img_RGB[:,:,2] * 0.0722)
            
    #Convert image to HSV
    img_HSV = col.rgb_to_hsv(img_RGB)
    
    #Extract features
    nr_pix = np.size(img_HSV[:,:,2])
    
    bla_pix = img_HSV[:,:,2] < 30
    perc_bla = round(bla_pix.sum() / nr_pix * 100)
    
    whi_pix = img_HSV[:,:,2] > 225
    perc_whi = round(whi_pix.sum() / nr_pix * 100)
    
    col_pix = np.invert(bla_pix + whi_pix)
    img_H = img_HSV[:,:,0]
    col_pix_H = img_H[col_pix] * 360
    nr_col_pix = np.size(col_pix_H)
    
    perc_r_y = round((col_pix_H < 60).sum() / nr_col_pix * 100)
    perc_y_g = round(((col_pix_H >= 60)*(col_pix_H < 120)).sum()/nr_col_pix * 100)
    perc_g_c = round(((col_pix_H >= 120)*(col_pix_H < 180)).sum()/nr_col_pix * 100)
    perc_c_b = round(((col_pix_H >= 180)*(col_pix_H < 240)).sum()/nr_col_pix * 100)
    perc_b_m = round(((col_pix_H >= 240)*(col_pix_H < 300)).sum()/nr_col_pix * 100)
    perc_m_r = round((col_pix_H >= 300).sum() / nr_col_pix * 100)
    
    mean_sat = round(img_HSV[:,:,1].mean(),2)
    mean_lig = round(img_HSV[:,:,2].mean())
    mean_lum = round(img_lum.mean())
    
    lig = np.reshape(img_HSV[:,:,2], nr_pix)
    lig_s = np.sort(lig)
    contrast = lig_s[round(3/4*nr_pix):].mean() - lig_s[:round(1/4*nr_pix)].mean()
    contrast = round(contrast)  
    
    #Extract frequency spectrum
    f = np.fft.fft2(img_lum)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    
    magnitude = np.hstack(magnitude_spectrum)
    
    low_freq = sum(magnitude[magnitude<200])
    high_freq = sum(magnitude[magnitude>=200])
    
    # For plotting    
    rows, cols = img_lum.shape
    crow,ccol = rows/2 , cols/2
    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)    
    
    # Collect all features and return
    post_feat.loc[0] = ([perc_bla, perc_whi, perc_r_y, perc_y_g, perc_g_c, 
                         perc_c_b, perc_b_m, perc_m_r, mean_sat, mean_lig,
                         mean_lum, contrast, low_freq, high_freq])            
    
    
    from matplotlib import pyplot as plt
    from matplotlib import cm
    
    plt.figure(1)
    plt.subplot(241)
    plt.imshow(img_RGB)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(242)
    plt.imshow(img_HSV[:,:,0], cmap = cm.hsv)
    plt.title('Hue')
    plt.axis('off')
    
    plt.subplot(245)
    plt.imshow(img_HSV[:,:,1], cmap = cm.gray_r)
    plt.title('Saturation')
    plt.axis('off')
    
    plt.subplot(246)
    plt.imshow(img_HSV[:,:,2], cmap = cm.gray)
    plt.title('Brightness')
    plt.axis('off')
    
    plt.subplot(243)
    plt.imshow(img_back, cmap= cm.gray)
    plt.title('HPF')
    plt.axis('off')
    
    #    plt.subplot(243)
    #    plt.imshow(img_ent, cmap= cm.jet)
    #    plt.title('Entropy')
    #    plt.axis('off')
    
    plt.subplot(247)
    plt.imshow(img_lum, cmap = cm.gray)
    plt.title('Luminance')
    plt.axis('off')
    
    plt.subplot(244)
    plt.imshow(whi_pix, cmap = cm.gray)
    plt.title('"White" pixels')
    plt.axis('off')
    
    plt.subplot(248)
    plt.imshow(bla_pix, cmap = cm.gray_r)
    plt.title('"Black" pixels')
    plt.axis('off')
    plt.show()
    
    return post_feat