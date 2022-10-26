# RANSAC 원근 변환 통한 나쁜 매칭 제거
import cv2 as cv

# file path
path = './'
# Text Types
# font
font = cv.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50)
# fontScale
fontScale = 1
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2

# Open Camera
def open_camera():
    # 그리드 정보
    camera = cv.VideoCapture(0)
    # camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    # camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

# 파일 존재 여부
def existsfile(filename):
    from os.path import isfile
    return isfile(filename)

# 비교 기준 받기
def takeimagefromcamera(camera):
    file_name = path + 'capture.jpg'
    win_name = 'Take Captrue Image Source'
    if existsfile(file_name):
        import os
        os.remove(file_name)

    while True:
        ret, frame = camera.read()                 # 카메라 프레임 읽기
        text = 'Press key s To Capture image!'
        if ret:
            # 화면에 인식할 영역 계산
            (left, top, right, bottom) = calc_rectangle(frame)
            cv.rectangle(frame, (left,top), (right,bottom), (255,255,255), 2)
            frame = cv.putText(frame, text, org, font, 
                   fontScale, color, thickness, cv.LINE_AA)

            cv.imshow(win_name,frame)          # 프레임 화면에 표시

            key = cv.waitKey(10)
            if key == 27:
                break
            if key == ord('s'):
                cv.imwrite(file_name,frame[top:bottom, left:right])
                break
        else:
            print('No frame!')
            break
    
    cv.destroyWindow(win_name)

    if not existsfile(file_name):
        file_name = ''
    return file_name

# 화면에 인식할 영역 계산
def calc_rectangle(frame):
    height, width = frame.shape[:2]
    left = width // 3
    right = (width // 3) * 2
    top = (height // 2) - (height // 3)
    bottom = (height // 2) + (height // 3)

    return (left, top, right, bottom)

# 동영상과 영상 근사 영역 계산
def detectAndCompute(camera, image_source):
    win_name = 'Detect And Compute'
    while True:
        ret, frame = camera.read()                 # 카메라 프레임 읽기
        if ret:
            # 화면에 인식할 영역 계산
            cv.imshow(win_name,frame)          # 프레임 화면에 표시
            
            # RANSAC 원근 변환 근사 계산으로 나쁜 매칭 제거
            findHomographywithRANSAC(image_source, frame)
            
            key = cv.waitKey(1)
            if key == 27:
                break
        else:
            print('no frame!')
            break

# RANSAC 원근 변환 근사 계산으로 나쁜 매칭 제거
def findHomographywithRANSAC(image_source, image_target):
    gray_source = cv.cvtColor(image_source, cv.COLOR_BGR2GRAY)
    gray_target = cv.cvtColor(image_target, cv.COLOR_BGR2GRAY)

    # ORB, BF-Hamming 로 knnMatch  
    detector = cv.ORB_create()
    kp1, desc1 = detector.detectAndCompute(gray_source, None)
    kp2, desc2 = detector.detectAndCompute(gray_target, None)
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)

    # 매칭 결과를 거리기준 오름차순으로 정렬 
    matches = sorted(matches, key=lambda x:x.distance)

    # 매칭점으로 원근 변환 및 영역 표시 
    import numpy as np
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])

    # RANSAC으로 변환 행렬 근사 계산 
    mtrx, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    h,w = image_source.shape[:2]
    pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
    dst = cv.perspectiveTransform(pts,mtrx)
    image_target = cv.polylines(image_target,[np.int32(dst)],True,(0,0,255),3, cv.LINE_AA)

    # 정상치 매칭만 그리기 
    matchesMask = mask.ravel().tolist()
    result_matches = cv.drawMatches(image_source, kp1, image_target, kp2, matches, None, \
                        matchesMask = matchesMask,
                        flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    # 모든 매칭점과 정상치 비율 
    accuracy = float(mask.sum()) / mask.size

    # 결과 출력 
    text = "Accuracy: %d/%d(%.2f%%)"% (mask.sum(), mask.size, accuracy)
    text = text + ", Press Key 'esc' To Exit!"
    result_matches = cv.putText(result_matches, text, org, font, 
            fontScale, color, thickness, cv.LINE_AA)
    cv.imshow('Matchings', result_matches)

if __name__ == '__main__':
    # Open Camera
    camera = open_camera()
    if not camera.isOpened:
        print('no camera!')
        cv.destroyAllWindows()
        exit()
    else:
        # 비교 기준 받기
        file_source = takeimagefromcamera(camera)
        if file_source:
            image_source = cv.imread(file_source)

            # 동영상과 영상 근사 영역 계산
            detectAndCompute(camera, image_source)

    # release camera and windows
    camera.release()
    cv.destroyAllWindows()
    