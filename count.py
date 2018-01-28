# -*- coding: utf-8

import cv2
import matplotlib.pyplot as plt


def main():
    images=[
        'images/saikoros_on_the_desk.jpg',
    ]

    for img in images:
        img_org = cv2.imread(img)
        height, width = img_org.shape[:2]
        img_org = cv2.resize(img_org, (int(width/8), int(height/8)))

        # to gray
        img = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

        # バイラテラルフィルタ（平滑化）
        img = cv2.bilateralFilter(img, 3, 24, 2)

        # Canny
        img = cv2.Canny(img, 50, 120)

        # サイコロが大部分が白色なので、それを利用してサイコロ部分だけ抜き出す。
        whiteLower = (150, 150, 150)
        whiteUpper = (255, 255, 255)
        mask = cv2.inRange(img_org, whiteLower, whiteUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # サイコロの輪郭検出
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
		                          cv2.CHAIN_APPROX_SIMPLE)

        # 輪郭を面積が大きい順に並べ替える。
        contours.sort(key=cv2.contourArea, reverse=True)

        eyes_of_saikoros = []

        for i, contour in enumerate(contours):
            # サイコロを囲む四角。傾いていても気にしない。
            x,y,width,height = cv2.boundingRect(contour)
            # cv2.rectangle(img_org,(x,y),(x+width,y+height),(0,255,0),2)

            mask_saikoro = mask[y:y+height,x:x+width]
            img_org_saikoro = img_org[y:y+height,x:x+width]

            height, width = mask_saikoro.shape[:2]
            mask_saikoro = cv2.resize(mask_saikoro, (int(width*4), int(height*4)))
            img_org_saikoro = cv2.resize(img_org_saikoro, (int(width*4), int(height*4)))
            x,y,width,height = (x*4, y*4, width*4, height*4)

            cv2.imshow('saikoro', img_org_saikoro)
            cv2.imshow('saikoro', mask_saikoro)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # サイコロを囲む四角の大きさからサイコロの目の大きさを計算
            # （傾いている場合もあるので厳密ではない）
            eye_radius_min = int((((height + width) / 2) / 100))
            eye_radius_max = int((((height + width) / 2) / 4))
            eye_min_dist = int((((height + width) / 2) / 20))

            print("height {} width {} eye raius min {} max{} min dist {}".format(height, width, eye_radius_min, eye_radius_max, eye_min_dist))

            # 球の大きさはあくまで入力の画像に合わせているのにすぎないので、別の画像を使用する場合は要調整
            circles = cv2.HoughCircles(mask_saikoro, cv2.HOUGH_GRADIENT,
                                       # param1の値は大きい方がよいっぽい
                                       dp=2, minDist=eye_min_dist,
                                       param1=200, param2=40, minRadius=eye_radius_min, maxRadius=eye_radius_max)
                                       # dp=2, minDist=5,

            if circles is None:
                raise Exception("cannot find eye of saikoro")

            for i, circle in enumerate(circles[0]):
                x, y, r = circle
                cv2.circle(img_org_saikoro, (int(x),int(y)), 3, (0, 0, 255), 3)
                cv2.rectangle(img_org_saikoro, (x-r,y-r), (x+r, y+r), (255,255,0),1)

            cv2.imshow('counters', img_org_saikoro)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            eyes_of_saikoros.append(len(circles[0]))

        print("Sum is {} {}".format(sum(eyes_of_saikoros), eyes_of_saikoros))


if __name__ == '__main__':
    main()
