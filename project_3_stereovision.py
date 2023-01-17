import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def rescale(frame, scale = 0.5):
    width = int (frame.shape[1]*scale)
    height = int (frame.shape[0]*scale)
    dimensions = (width , height)

    return cv.resize (frame, dimensions, interpolation = cv.INTER_AREA)


def normalize(ab):

    ab_dash = np.mean(ab, axis=0)
    a_dash ,b_dash = ab_dash[0], ab_dash[1]

    a_cap = ab[:,0] - a_dash
    b_cap = ab[:,1] - b_dash

    s = (2/np.mean(a_cap**2 + b_cap**2))**(0.5)
    T_scale = np.diag([s,s,1])
    T_trans = np.array([[1,0,-a_dash],[0,1,-b_dash],[0,0,1]])
    T = T_scale.dot(T_trans)

    x = np.column_stack((ab, np.ones(len(ab))))
    x_norm = (T.dot(x.T)).T

    return  x_norm, T


def drawlines(img1,img2,lines,pts1,pts2):

    r,c, _ = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def fundemental(matched_pts, iter = 1000, thresh = 0):

    x1 = matched_pts[:,0:2]
    x2 = matched_pts[:,2:4]

    x1tmp=np.array([x1[1][0], x1[1][1], 1]).T
    x2tmp=np.array([x2[1][0], x2[1][1], 1])

    for i in range(iter):
        indices = []
        r = matched_pts.shape[0]
        randm = np.random.choice(r, size = 8)
        req_8_feat = matched_pts[randm, :]


        A = np.zeros((len(x1),9))
        for i in range(0, len(x1)):
            x_1,y_1 = x1[i][0], x1[i][1]
            x_2,y_2 = x2[i][0], x2[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

        U, S, VT = np.linalg.svd(A)
        fund = VT.T[:, -1]
        fund = fund.reshape(3,3)


        U2, S2, VT2 = np.linalg.svd(fund)
        S2 = np.diag(S2)
        S2[2,2] = 0
        fund = np.dot(U2, np.dot(S2, VT2))
        for j in range(r):
            indices.append(j)
        filtered_pts = matched_pts[indices, :]

    return fund, filtered_pts

def EssentialMatrix(fund):
    Essen = cam1.T.dot(fund).dot(cam0)
    U3,S3,VT3 = np.linalg.svd(Essen)
    S3 = [1,1,0]
    Essen_rect = np.dot(U3,np.dot(np.diag(S3),VT3))
    return Essen_rect

def Disparity(block_size = 15, x_search_block_size = 50, y_search_block_size = 1):
    disp_map = np.zeros((h1,w1))

    first = True
    min_ssd = None
    min_index = None

    for y in range(block_size, h1-block_size):
        for x in range(block_size, w1-block_size):
            block_left = img_left[y:y + block_size, x:x + block_size]
            block_right = img_right[y: y+block_size, x: x+block_size]

            for k in range(max(0, y - y_search_block_size), min(img_right.shape[0]-1, y + y_search_block_size)):
                for o in range(max(0, x - x_search_block_size),  min(img_right.shape[1], x + x_search_block_size)):
                    ssd = np.sum((block_left-block_right)**2)

                    if first:
                        min_ssd = ssd
                        min_index = (y,x)
                        first = False
                    
                    else:
                        if ssd < min_ssd:
                            min_ssd = ssd
                            min_index = (y, x)

            
            disp_map[y,x] = abs(min_index[1] - x)

            disp_unscaled = disp_map.copy()

            max_pix = np.max(disp_map)
            min_pix = np.min(disp_map)

            for i in range(disp_map.shape[0]):
                for j in range(disp_map.shape[1]):
                    disp_map[i][j] = int((disp_map[i][j]*255)/(max_pix-min_pix + 1e-5))

    return disp_map, disp_unscaled


def depth(disp_unscaled, fund):
    depth_map = np.zeros((disp_unscaled.shape[0], disp_unscaled.shape[1]))
    depth_array = np.zeros((disp_unscaled.shape[0], disp_unscaled.shape[1]))
    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            depth_map[i][j] = 1/disp_unscaled[i][j]
            depth_array[i][j] = baseline*fund/disp_unscaled[i][j]

    return depth_map, depth_array


dataset = int(input('Enter the dataset number:'))

if dataset == 1:
    cam0 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    cam1 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    baseline, f = 88.39, 1758.23
    img_left = cv.imread("data/curule/im0.png")
    img_right = cv.imread("data/curule/im1.png")

elif dataset == 2:
    cam0 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    cam1 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    baseline, f = 221.76, 1742.11
    img_left = cv.imread("data/octagon/im0.png")
    img_right = cv.imread("data/octagon/im1.png")

elif dataset == 3:
    cam0 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    cam1 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    baseline, f = 537.75, 1729.05
    img_left = cv.imread("data/pendulum/im0.png")
    img_right = cv.imread("data/pendulum/im1.png")

else:

    print("Wrong Input")
    quit()

# img_left = gray = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
# img_right = gray = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

img_left = rescale(img_left)
img_right = rescale(img_right)


#Using ORB to get the features from the images

orb = cv.ORB_create(nfeatures=1000)


key_ptA, des1 = orb.detectAndCompute(img_left, None)
key_ptB, des2 = orb.detectAndCompute(img_right, None)

des1 = np.float32(des1)
des2 = np.float32(des2)

# Using FLANN algorithm and KNN matching to findout the matching features in both the images
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50) 
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

matches =  matches[0:100]

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
line_features = cv.drawMatchesKnn(img_left,key_ptA,img_right,key_ptB,matches,None)

src_pts = np.array([key_ptA[m[0].queryIdx].pt for m in matches])
dst_pts = np.array([key_ptB[m[0].trainIdx].pt for m in matches])
src_pts = np.int32(src_pts)
dst_pts = np.int32(dst_pts)


matched_pts = []

matched_pts.append([src_pts[0], dst_pts[0], src_pts[1], dst_pts[1]])
matched_pts = np.array(matched_pts).reshape(-1, 4)

fund, filtered_pts = fundemental(matched_pts, iter = 1000, thresh = 0)

Essen_rect = EssentialMatrix(fund)


lines1 = cv.computeCorrespondEpilines(dst_pts.reshape(-1,1,2), 2,fund)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img_left,img_right,lines1,src_pts,dst_pts)


lines2 = cv.computeCorrespondEpilines(src_pts.reshape(-1,1,2), 1,fund)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img_right,img_left,lines2,dst_pts,src_pts)


w1, h1, _ = img_left.shape
w2, h2, _ = img_right.shape
_, H1, H2 = cv.stereoRectifyUncalibrated(np.float64(src_pts), np.float64(dst_pts), fund, imgSize=(w1, h1))


image_Left_rectified = cv.warpPerspective(img_left, H1, (w1, h1))
image_Right_rectified = cv.warpPerspective(img_right, H2, (w2, h2))

matched_pts_left_chosen_rectified = cv.perspectiveTransform(np.array(src_pts, dtype=np.float64).reshape(-1,1,2), H1)
matched_pts_right_chosen_rectified = cv.perspectiveTransform(np.array(dst_pts, dtype=np.float64).reshape(-1,1,2), H2)

H2_T = H2.T
H2_T_inv =  np.linalg.inv(H2_T)	
H1_inv = np.linalg.inv(H1)
FM_rectified = np.dot(H2_T_inv, np.dot(fund, H1_inv))
linesL_rectified = cv.computeCorrespondEpilines(matched_pts_left_chosen_rectified,2, FM_rectified)
# print("Lrec: ",linesL_rectified)
linesL_rectified   = linesL_rectified[:,0]
# print("Lrec: ",linesL_rectified)
linesR_rectified = cv.computeCorrespondEpilines(matched_pts_right_chosen_rectified,2, FM_rectified)
linesR_rectified   = linesR_rectified[:,0]

rect_left,_ = drawlines(img_left,img_right,linesL_rectified,src_pts,dst_pts)
rect_right,_ = drawlines(img_right,img_left,linesR_rectified,dst_pts,src_pts)



  
disp_map, disp_unscaled = Disparity(block_size = 15, x_search_block_size = 50, y_search_block_size = 1)

depth_map, depth_array = depth(disp_unscaled, fund)




# cv.imshow('left image', img_left)
# cv.imshow('right image', img_right)
# cv.imshow('line features', line_features)
# cv.imshow("Lrectified",rect_left)
# cv.imshow("Rrectifed",rect_right)
# plt.subplot(121),plt.imshow(rect_left)
# plt.subplot(122),plt.imshow(rect_right)

plt.title('Disparity Map Graysacle')
plt.imshow(disp_map, cmap='gray')
plt.title('Disparity Map Hot')
plt.imshow(disp_map, cmap='hot')


plt.title('Depth Map Graysacle')
plt.imshow(depth_map, cmap='gray')
plt.title('Depth Map Hot')
plt.imshow(depth_map, cmap='hot')
plt.show()

cv.waitKey(0)








