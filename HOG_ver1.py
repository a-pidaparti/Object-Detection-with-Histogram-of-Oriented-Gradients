import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_differential_filter():
    filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return filter_x, filter_y

def filter_image(im, filter):
    im_padded = np.pad(im, 1)
    n, m = im_padded.shape
    im_filtered = np.zeros(im.shape)
    for i in range(0, n - 3):
        for j in range(0, m - 3):
            win_vector = im_padded[i:i+3,j:j+3].flatten()
            im_filtered[i,j] = np.dot(win_vector, filter.flatten())
    return im_filtered

def get_gradient(im_dx, im_dy):
    n, m = im_dx.shape
    grad_mag = np.zeros(im_dx.shape)
    grad_angle = np.zeros(im_dx.shape)
    for i in range(0, n):
        for j in range(0, m):
            if abs(im_dx[i,j]) > .001:
                grad_angle[i, j] = np.arctan(im_dy[i, j] / im_dx[i, j]) + (np.pi/2)
            else:
                if im_dy[i,j] < 0 and im_dx[i,j] < 0:
                    grad_angle[i,j] = 0
                else:
                    grad_angle[i,j] = np.pi
            grad_mag[i, j] = np.linalg.norm([im_dx[i,j], im_dy[i,j]])
    return grad_mag, grad_angle

def build_histogram(grad_mag, grad_angle, cell_size):
    M = int(grad_angle.shape[0] / cell_size)
    N = int(grad_angle.shape[1] / cell_size)
    ori_histo = np.zeros((M, N, 6), dtype=float)
    for i in range(M):
        for j in range(N):
            for x in range(cell_size):
                for y in range(cell_size):
                    x_cor = i*cell_size + x
                    y_cor = j*cell_size + y
                    angleInDeg = grad_angle[x_cor,y_cor] * (180 / np.pi)
                    if angleInDeg >= 0 and angleInDeg < 30:
                        ori_histo[i,j,0] += grad_mag[x_cor, y_cor]
                    elif angleInDeg >= 30 and angleInDeg < 60:
                        ori_histo[i,j,1] += grad_mag[x_cor, y_cor]
                    elif angleInDeg >= 60 and angleInDeg < 90:
                        ori_histo[i,j,2] += grad_mag[x_cor, y_cor]
                    elif angleInDeg >= 90 and angleInDeg < 120:
                        ori_histo[i,j,3] += grad_mag[x_cor, y_cor]
                    elif angleInDeg >= 120 and angleInDeg < 150:
                        ori_histo[i,j,4] += grad_mag[x_cor, y_cor]
                    else:
                        ori_histo[i,j,5] += grad_mag[x_cor, y_cor]

    return ori_histo

def get_block_descriptor(ori_histo, block_size):
    ori_histo_normalized = np.zeros((ori_histo.shape[0]-(block_size-1), ori_histo.shape[1]-(block_size-1), 6*(block_size**2)))

    x_corner = 0

    while x_corner + block_size <= ori_histo.shape[0]:
        y_corner = 0
        while y_corner + block_size <= ori_histo.shape[1]:
            hist_cat = ori_histo[x_corner:x_corner+block_size, y_corner:y_corner+block_size, :].flatten()
            hist_cat_norm = np.empty([0])
            hist_norm_constant = np.sqrt(np.sum(np.square(hist_cat)))
            for h in hist_cat:
                hist_cat_norm = np.append(hist_cat_norm, (h / hist_norm_constant + 0.001))

            ori_histo_normalized[x_corner, y_corner, :] = hist_cat_norm
            y_corner += block_size
        x_corner += block_size

    return ori_histo_normalized

def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    # To do
    im = (im - np.min(im)) / np.max(im)
    filter_x, filter_y = get_differential_filter()
    im_filtered_dx = filter_image(im, filter_x)
    im_filtered_dy = filter_image(im, filter_y)
    grad_mag, grad_angle = get_gradient(im_filtered_dx, im_filtered_dy)
    hist_mat = build_histogram(grad_mag, grad_angle, 8)
    ori_histo_normalized = get_block_descriptor(hist_mat, 2)
    hog = ori_histo_normalized.flatten()
    # visualize to verify
    visualize_hog(im, hog, 8, 2)

    return hog

# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()

def iou_rejection(box_1, box_2, box_dim):
    xa1 = box_1[0]
    xa2 = box_1[0] + box_dim[0]
    ya1 = box_1[1]
    ya2 = box_1[1] + box_dim[1]

    xb1 = box_2[0]
    xb2 = box_2[0] + box_dim[0]
    yb1 = box_2[1]
    yb2 = box_2[1] + box_dim[1]

    overlap = (min(xa2, xb2) - max(xa1, xb1)) * (min(ya2, yb2) - max(ya1, yb1))

    return overlap/((box_dim[0] * box_dim[1] * 2) - overlap)

def face_recognition(I_target, I_template):
    dim = I_template.shape
    unrejected_bounding_boxes = []

    template_hog = extract_hog(I_template)
    template_hog = template_hog - np.mean(template_hog)
    template_hog_norm = np.linalg.norm(template_hog)
    x_lim, y_lim = I_target.shape
    x_bounds, y_bounds = I_template.shape
    i,j = 0,0
    while i <= x_lim-x_bounds:
        print(i, '/', x_lim - x_bounds)
        while j <= y_lim-y_bounds:
            window_hog = extract_hog(I_target[i:i+x_bounds, j:j+y_bounds])
            window_hog = window_hog - np.mean(window_hog)
            window_hog_norm = np.linalg.norm(window_hog)
            ncc_score = float(np.dot(window_hog, template_hog)/ (template_hog_norm * window_hog_norm))
            if ncc_score >= .6:
                unrejected_bounding_boxes += [[j, i, ncc_score]]
            j += 5
        i += 5
        j = 0

    unrejected_bounding_boxes = sorted(list(unrejected_bounding_boxes),key=lambda row: (row[2]), reverse=True)
    bounding_boxes = []
    while unrejected_bounding_boxes != []:
        bounding_boxes.append(unrejected_bounding_boxes[0])
        rem = []
        for box in unrejected_bounding_boxes:
            if iou_rejection(bounding_boxes[-1], box, dim) >= 0.5:
                rem.append(box)
        for box in rem:
            unrejected_bounding_boxes.remove(box)

    return np.array(bounding_boxes)

def visualize_face_detection(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):
        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()


if __name__ == '__main__':
    im = cv2.imread('cameraman.tif', 0)
    hog = extract_hog(im)

    # I_target= cv2.imread('target.png', 0)
    # #MxN image
    #
    # I_template = cv2.imread('template.png', 0)
    # #mxn  face template
    #
    # bounding_boxes=face_recognition(I_target, I_template)
    #
    # I_target_c= cv2.imread('target.png')
    # # MxN image (just for visualization)
    # visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
    # #this is visualization code.




