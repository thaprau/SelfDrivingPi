# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import pigpio

#Functions for the frame handling for self driving

#Finds the boxes of the image
def square_up(frame, iteration, x_start, y_start):

    side = (int) (x_side/2)
    y_side = side_y

    start = x_start - side
    stop = x_start + side

    start_y = y_start - (iteration+1)*y_side 
    stop_y =  start_y + y_side

    square = frame[start_y:stop_y, start:stop]
    _ , x_indices = np.nonzero(square)
    
    if(x_indices.size == 0):
        return x_start

    else:
        x = (int) (np.mean(x_indices))
        return (int) (start+x)


# Scale the image so one the lines are left
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(5, 0), (start_x+5, IMAGE_H), (start_x_2-5, IMAGE_H), (IMAGE_W-5, 0)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

#Finds the edges in the image
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (13, 13), 0)
    canny = cv2.Canny(blur, 50, 100)
    return canny

#FUntion to display lines in picture
def display_lines(image, x_value, y_value, nr_of_points):

     #line_image = np.zeros_like(image)

    values = np.array([[x_value[0], y_value[0]]])
    for point in range(nr_of_points-1):
        values = np.append(values, np.array([[x_value[point+1], y_value[point+1]]]), axis=0)
    
    cv2.polylines(image, [values], False, (0,255,0), 2)

    return image


def display_poly(image, x1_value, y1_value, nr_of_points):

    #line_image = np.zeros_like(image)

    values = np.array([[x_value[0], y_value[0]]])
    for point in range(nr_of_points-1):
        values = np.append(values, np.array([[x_value[point+1], y_value[point+1]]]), axis=0)

    cv2.fillPoly(image, [values], False, (0,255,0), 2)

    return image
 
 
#Initalize matrices and variables for image transform
IMAGE_H = 922
IMAGE_W = 1640
start_x = 591
start_x_2 = 1047
difference = start_x_2 - start_x


print("1")

src = np.float32([[200, IMAGE_H], [1328, IMAGE_H], [0, 0], [IMAGE_W, 0]])
dst = np.float32([[start_x, IMAGE_H], [start_x_2, IMAGE_H], [0, 0], [IMAGE_W, 0]])
M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
Minv = cv2.getPerspectiveTransform(dst, src)

print("2")
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.rotation = 270
camera.resolution = (1640, 922)
camera.framerate = 2
rawCapture = PiRGBArray(camera, size=(1640, 922))

print("3")
 
# allow the camera to warmup
time.sleep(2)

# Steering values
STEERING_CENTER = 1650
STEERING_LEFT = STEERING_CENTER+300
STEERING_RIGHT = STEERING_CENTER-300

## Setup for controlling pwm signal ##

pi = pigpio.pi()
STEER = 18
DRIVE = 17

drive_value = 1600
steer_value = STEERING_CENTER

pi.set_PWM_frequency(STEER, 50)
pi.set_PWM_frequency(DRIVE, 50)

pi.set_servo_pulsewidth(STEER, steer_value)
pi.set_servo_pulsewidth(DRIVE, drive_value)


    
#Iteration to find all the boxes
x_side = 300
side_y = 160
nr_of_squares = 3
x_value = np.arange(nr_of_squares)
y_value = np.arange(nr_of_squares)
x2_value = np.arange(nr_of_squares)
y2_value = np.arange(nr_of_squares)
Y_start = IMAGE_H - 50
side = side_y
half_side = (int)(side/2)


start_time = time.time()
##Looping part


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    #Reads the image
    image = frame.array
    image = image[0:(0+IMAGE_H),0:IMAGE_W]
    
    #Analyz the image
    warped_image = cv2.warpPerspective(image, M, (IMAGE_W, IMAGE_H))
    #gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    canny_image = canny(warped_image)
    

    for square in range(nr_of_squares):
        x_value[square] = square_up(canny_image, square, start_x, Y_start)
        y_value[square] = Y_start - square*side - half_side

        x2_value[square] = square_up(canny_image, square, start_x_2, Y_start)
        y2_value[square] = Y_start - square*side - half_side

        start_x = x_value[square]
        start_x_2 = x2_value[square]
        
    if(x2_value[0] == x2_value[2]):
        x2_value = x_value + difference
        print("Second fixed")
    elif(x_value[0] == x_value[2]):
        x_value = x2_value - difference
        print("First fixed")
        
        
    for square in range(nr_of_squares):
        x1 = x_value[square] - (int)(x_side/2)
        x2 = x_value[square] + (int)(x_side/2)

        y1 = y_value[square] - half_side
        y2 = y_value[square] + half_side

        x12 = x2_value[square] - (int)(x_side/2)
        x22 = x2_value[square] + (int)(x_side/2)

        y12 = y2_value[square] - half_side
        y22 = y2_value[square] + half_side


        cv2.rectangle(canny_image,(x1,y1),(x2,y2),(255,0,0),1)
        cv2.rectangle(canny_image,(x12,y12),(x22,y22),(255,0,0),1)
        
        

        
        
        
    # If polyfit 
    #p2 = np.poly1d(np.polyfit(y2_value, x2_value, 2))
    #p1 = np.poly1d(np.polyfit(y_value, x_value, 2))
    
    #Value of the center
    start_x = 591
    start_x_2 = 1047
    center = (int)((start_x + start_x_2)/2)
    
    
    
    #Stuff for display
    
    #x2_2 = p2(y2_value).astype(int)
    #x1_2 = p1(y_value).astype(int)
    #center_x = ((x2_2+x1_2)/2).astype(int)
    #display_lines(warped_image, x1_2, y_value, nr_of_squares)
    #display_lines(warped_image, x2_2, y2_value, nr_of_squares)
    #display_lines(warped_image, center_x, y_value, nr_of_squares)
    #normal_picture = cv2.warpPerspective(warped_image, Minv, (IMAGE_W, IMAGE_H))
    
    #calculates the offset value
    box_value = (x2_value[0] + x_value[0] +
                  x2_value[1] + x_value[1] +
                  x2_value[2] +x_value[2])//6
                  #x2_value[3] +x_value[3] +
                 #x2_value[4] +x_value[4]
                
    
    #cv2.line(canny_image, (center, 0), (center, 1000), (0, 255, 0), 10)
    #cv2.line(canny_image, (box_value, 0), (box_value, 1000), (255, 0, 0), 10)
    #Steer the car
    steer_value = STEERING_CENTER - (box_value-center)*6
    print("Steer value: ", steer_value)
    #print((int)(time.time() - start_time))
    
    if(steer_value > STEERING_LEFT):
        pi.set_servo_pulsewidth(STEER, STEERING_LEFT)
        
       
    elif(steer_value < STEERING_RIGHT):
        pi.set_servo_pulsewidth(STEER, STEERING_RIGHT)
        
    else:
        pi.set_servo_pulsewidth(STEER, steer_value)
    
    print("Set steer value: ", pi.get_servo_pulsewidth(STEER))
        
        
    rawCapture.truncate(0) # Truncate frame to be able to read the next
    
    #Set True to get live feed
    if(True):
    
        # show the frame
        cv2.imshow("Frame", canny_image)
        key = cv2.waitKey(1) & 0xFF
     
        # clear the stream in preparation for the next frame
        
     
        #if the `q` key was pressed, break from the loop
        if key == ord("q"):
            cv2.imwrite("last_frame.png", canny_image)
            pi.set_servo_pulsewidth(STEER, 0)
            pi.set_servo_pulsewidth(DRIVE, 0)
            pi.stop()
            break

















