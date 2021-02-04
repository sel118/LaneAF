import numpy as np
import cv2
import matplotlib.pyplot as plt


def tensor2image(tensor, mean, std):
    mean = mean[..., np.newaxis, np.newaxis] # (nc, 1, 1)
    mean = np.tile(mean, (1, tensor.size()[2], tensor.size()[3])) # (nc, H, W)
    std = std[..., np.newaxis, np.newaxis] # (nc, 1, 1)
    std = np.tile(std, (1, tensor.size()[2], tensor.size()[3])) # (nc, H, W)

    image = 255.0*(std*tensor[0].cpu().float().numpy() + mean) # (nc, H, W)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    image = np.transpose(image, (1, 2, 0)) # (C, H, W) to (H, W, C)
    image = image[:, :, ::-1] # RGB to BGR
    return image.astype(np.uint8) # (H, W, C)

def create_viz(img, seg, mask, vaf, haf):
    scale = 8
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    #seg_large = cv2.resize(seg, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    #seg_large_color = cv2.applyColorMap(40*seg_large, cv2.COLORMAP_JET)
    #img[seg_large > 0, :] = seg_large_color[seg_large > 0, :]
    img = np.ascontiguousarray(img, dtype=np.uint8)
    seg_color = cv2.applyColorMap(40*seg, cv2.COLORMAP_JET)
    rows, cols = np.nonzero(seg)
    for r, c in zip(rows, cols):
        img = cv2.arrowedLine(img, (c*scale, r*scale),(int(c*scale+vaf[r, c, 0]*scale*0.75), 
            int(r*scale+vaf[r, c, 1]*scale*0.5)), seg_color[r, c, :].tolist(), 1, tipLength=0.4)
    return img

def create_viz_old(img, seg, mask, vaf, haf):
    haf_dim = np.zeros((haf.shape[0], haf.shape[1]))
    haf = np.dstack((haf, haf_dim))
    down_rate = 1 # downsample visualization by this factor
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax1.imshow(img)
    ax2.imshow(mask)
    # visualize VAF
    q = ax3.quiver(np.arange(0, vaf.shape[1], down_rate), -np.arange(0, vaf.shape[0], down_rate), 
                   vaf[::down_rate, ::down_rate, 0], -vaf[::down_rate, ::down_rate, 1], scale=120, color='g')
    # visualize HAF
    q = ax4.quiver(np.arange(0, haf.shape[1], down_rate), -np.arange(0, haf.shape[0], down_rate), 
                   haf[::down_rate, ::down_rate, 0], -haf[::down_rate, ::down_rate, 1], scale=120, color='b')
    
    fig.canvas.draw()
    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    #im_out = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    fig.clear()
    plt.close(fig)
    return im_out
