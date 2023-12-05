import argparse
from time import sleep

import cv2
import numpy as np
import yaml

COLOR_RED = ( 0, 0, 255 )
COLOR_GREEN = ( 0, 255, 0 )
COLOR_YELLOW = ( 0, 255, 255 )
COLOR_BLUE = ( 255, 0, 0 )
BIG_CIRCLE = 60
SMALL_CIRCLE = 3
COLORBOX = [ COLOR_GREEN, COLOR_YELLOW, COLOR_RED ]


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def compute_perspective_transform( corner_points, width, height, image ):
    corner_points_array = np.float32( corner_points )
    img_params = np.float32( [ [ 0, 0 ], [ width, 0 ], [ 0, height ], [ width, height ] ] )
    matrix = cv2.getPerspectiveTransform( corner_points_array, img_params )
    img_transformed = cv2.warpPerspective( image, matrix, ( width, height ) )
    return matrix, img_transformed


def compute_point_perspective_transformation( matrix, list_downoids ):
    list_points_to_detect = np.float32( list_downoids ).reshape( -1, 1, 2 )
    transformed_points = cv2.perspectiveTransform( list_points_to_detect, matrix )
    transformed_points_list = list()
    for i in range( 0, transformed_points.shape[ 0 ] ):
        transformed_points_list.append( [ transformed_points[ i ][ 0 ][ 0 ], transformed_points[ i ][ 0 ][ 1 ] ] )
    return transformed_points_list


def read_corner_factor():
    print( bcolors.WARNING + "[ Loading config file for the bird view transformation ] " + bcolors.ENDC )
    with open( "./config/config_birdview.yml", "r" ) as ymlfile:
        cfg = yaml.load( ymlfile, Loader=yaml.FullLoader )
    corner_points = []
    for section in cfg:
        corner_points.append( cfg[ "image_parameters" ][ "p1" ] )
        corner_points.append( cfg[ "image_parameters" ][ "p2" ] )
        corner_points.append( cfg[ "image_parameters" ][ "p3" ] )
        corner_points.append( cfg[ "image_parameters" ][ "p4" ] )
        width_og = int( cfg[ "image_parameters" ][ "width_og" ] )
        height_og = int( cfg[ "image_parameters" ][ "height_og" ] )
        corner_points = [ ( x / width_og, y / height_og ) for x, y in corner_points ]
        img_path = cfg[ "image_parameters" ][ "img_path" ]
        size = cfg[ "image_parameters" ][ "size" ]
    print( bcolors.OKGREEN + " Done : [ Config file loaded ] ..." + bcolors.ENDC )
    return corner_points, size


def dist( p1, p2 ):
    return ( ( p1[ 0 ] - p2[ 0 ] )**2 + ( p1[ 1 ] - p2[ 1 ] )**2 )**0.5


def draw_rectangle( frame, corner_points ):
    cv2.line( frame, tuple( map( int, corner_points[ 0 ] ) ), tuple( map( int, corner_points[ 1 ] ) ), COLOR_BLUE, thickness=1 )
    cv2.line( frame, tuple( map( int, corner_points[ 1 ] ) ), tuple( map( int, corner_points[ 3 ] ) ), COLOR_BLUE, thickness=1 )
    cv2.line( frame, tuple( map( int, corner_points[ 0 ] ) ), tuple( map( int, corner_points[ 2 ] ) ), COLOR_BLUE, thickness=1 )
    cv2.line( frame, tuple( map( int, corner_points[ 3 ] ) ), tuple( map( int, corner_points[ 2 ] ) ), COLOR_BLUE, thickness=1 )

parser = argparse.ArgumentParser( description='Script to run MobileNet-SSD object detection network ' )
parser.add_argument( "--video", help="path to video file. If empty, camera's stream will be used" )
parser.add_argument(
    "--prototxt",
    default="MobileNetSSD_deploy.prototxt",
    help='Path to text network file: '
    'MobileNetSSD_deploy.prototxt for Caffe model or ' )
parser.add_argument(
    "--weights",
    default="MobileNetSSD_deploy.caffemodel",
    help='Path to weights: '
    'MobileNetSSD_deploy.caffemodel for Caffe model or ' )
parser.add_argument( "--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections" )
parser.add_argument( "--redthr", default=100, type=int, help="distance threshold to draw red box" )
parser.add_argument( "--yelthr", default=200, type=int, help="distance threshold to draw yellow box" )
args = parser.parse_args()

# Labels of Network.
classNames = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}

if args.video:
    cap = cv2.VideoCapture( args.video )
else:
    cap = cv2.VideoCapture( 0 )

net = cv2.dnn.readNetFromCaffe( args.prototxt, args.weights )
corner_points, size = read_corner_factor()
corner_real, matrix, bird_img = [ None ] * 3

while True:
    ret, frame = cap.read()
    frame_resized = cv2.resize( frame, ( 300, 300 ) )  # resize frame for prediction
    if not corner_real:
        corner_real = [ ( x * frame.shape[ 0 ], y * frame.shape[ 1 ] ) for x, y in corner_points ]
    matrix, bird_img = compute_perspective_transform( corner_real, size, size, frame )
    blob = cv2.dnn.blobFromImage( frame_resized, 0.007843, ( 300, 300 ), ( 127.5, 127.5, 127.5 ), False )
    net.setInput( blob )
    detections = net.forward()

    cols = frame_resized.shape[ 1 ]
    rows = frame_resized.shape[ 0 ]

    dtet_obj = []

    for i in range( detections.shape[ 2 ] ):
        confidence = detections[ 0, 0, i, 2 ]  #Confidence of prediction
        class_id = int( detections[ 0, 0, i, 1 ] )  # Class label
        if confidence > args.thr and class_id in [ 15 ]:  # Filter prediction

            xLeftBottom = int( detections[ 0, 0, i, 3 ] * cols )
            yLeftBottom = int( detections[ 0, 0, i, 4 ] * rows )
            xRightTop = int( detections[ 0, 0, i, 5 ] * cols )
            yRightTop = int( detections[ 0, 0, i, 6 ] * rows )

            heightFactor = frame.shape[ 0 ] / 300.0
            widthFactor = frame.shape[ 1 ] / 300.0
            xLeftBottom = int( widthFactor * xLeftBottom )
            yLeftBottom = int( heightFactor * yLeftBottom )
            xRightTop = int( widthFactor * xRightTop )
            yRightTop = int( heightFactor * yRightTop )

            dtet_obj.append( [ ( xLeftBottom, yLeftBottom ), ( xRightTop, yRightTop ),
                               ( ( xLeftBottom + xRightTop ) // 2, yRightTop ), 0 ] )
    tnfm_obj = compute_point_perspective_transformation( matrix, np.array( [ x[ 2 ] for x in dtet_obj ] ) )

    for i in range( len( tnfm_obj ) ):
        if tnfm_obj[ i ][ 0 ] < 0 or tnfm_obj[ i ][ 1 ] < 0:  #or tnfm_obj[ i ][ 0 ] > size or tnfm_obj[ i ][ 1 ] > size:
            continue
        for j in range( i + 1, len( tnfm_obj ) ):
            if tnfm_obj[ j ][ 0 ] < 0 or tnfm_obj[ j ][ 1 ] < 0:  #or tnfm_obj[ j ][ 0 ] > size or tnfm_obj[ j ][ 1 ] > size:
                continue
            ijdist = dist( tnfm_obj[ i ], tnfm_obj[ j ] )
            if ijdist < args.yelthr:
                dtet_obj[ i ][ 3 ] = max( dtet_obj[ i ][ 3 ], 1 )
                dtet_obj[ j ][ 3 ] = max( dtet_obj[ j ][ 3 ], 1 )
                if ijdist < args.redthr:
                    dtet_obj[ i ][ 3 ] = max( dtet_obj[ i ][ 3 ], 2 )
                    dtet_obj[ j ][ 3 ] = max( dtet_obj[ j ][ 3 ], 2 )
                cv2.line(
                    bird_img,
                    tuple( map( int, tnfm_obj[ i ] ) ),
                    tuple( map( int, tnfm_obj[ j ] ) ),
                    COLOR_RED if ijdist < args.redthr else COLOR_YELLOW,
                    thickness=1 )
                cv2.line(
                    frame,
                    tuple( map( int, dtet_obj[ i ][ 2 ] ) ),
                    tuple( map( int, dtet_obj[ j ][ 2 ] ) ),
                    COLOR_RED if ijdist < args.redthr else COLOR_YELLOW,
                    thickness=1 )

        cv2.circle( bird_img, tuple( map( int, tnfm_obj[ i ] ) ), 15, COLORBOX[ dtet_obj[ i ][ 3 ] ], 2 )
        cv2.rectangle( frame, dtet_obj[ i ][ 0 ], dtet_obj[ i ][ 1 ], COLORBOX[ dtet_obj[ i ][ 3 ] ] )

    cv2.namedWindow( "frame", cv2.WINDOW_NORMAL )
    draw_rectangle( frame=frame, corner_points=corner_real )
    cv2.imshow( "frame", frame )
    cv2.imshow( "Bird view", bird_img )
    sleep( 0.1 )
    if cv2.waitKey( 1 ) == 27:  # Break with ESC
        break
