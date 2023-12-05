import cv2
import argparse
import yaml

parser = argparse.ArgumentParser( description='' )
parser.add_argument( '-V', '--video', help='path to video file. If empty, camera\'s stream will be used.' )
parser.add_argument( '-H', '--height', default=0, type=int, help='height of calibration image.' )
parser.add_argument( '-W', '--width', default=0, type=int, help='width of calibration image.' )
parser.add_argument( '-S', '--size', default=0, type=int, help='size of contour' )

list_points_g = list()
list_his_g = 0

def callBackFunc( event, x, y, flags, param ):
    if event == cv2.EVENT_LBUTTONDOWN:
        print( 'Left button of the mouse is clicked - position (', x, ', ', y, ')' )
        list_points_g.append( [ x, y ] )
    elif event == cv2.EVENT_RBUTTONDOWN:
        print( 'Right button of the mouse is clicked - position (', x, ', ', y, ')' )
        list_points_g.append( [ x, y ] )


def saveFirstFrameOfStream( videoAddr=None, resizeW=0, resizeH=0 ) -> None:
    cap = cv2.VideoCapture( videoAddr or 0 )
    _, frame = cap.read()
    if resizeH == 0:
        if resizeW == 0:
            resizeW = 540
        resizeH = frame.shape[ 0 ] / frame.shape[ 1 ] * resizeW
    elif resizeW == 0:
        resizeW = frame.shape[ 1 ] / frame.shape[ 0 ] * resizeH
    frame = cv2.resize( frame, ( int( resizeW ), int( resizeH ) ) )
    cv2.imwrite( './image/static_frame_from_video.jpg', frame )


def saveConfig( img, size ) -> None:
    config_data = dict(
        image_parameters=dict(
            p2=list_points_g[ 3 ],
            p1=list_points_g[ 2 ],
            p4=list_points_g[ 0 ],
            p3=list_points_g[ 1 ],
            width_og=img.shape[ 0 ],
            height_og=img.shape[ 1 ],
            img_path=img_path,
            size=size ) )
    # Write the result to the config file
    with open( './config/config_birdview.yml', 'w' ) as outfile:
        yaml.dump( config_data, outfile, default_flow_style=False )


if __name__ == '__main__':
    args = parser.parse_args()
    saveFirstFrameOfStream( args.video, args.width, args.height )
    windowName = 'MouseCallback'
    cv2.namedWindow( windowName )

    img_path = './image/static_frame_from_video.jpg'
    img = cv2.imread( img_path )

    cv2.setMouseCallback( windowName, callBackFunc )
    while ( True ):
        cv2.imshow( windowName, img )
        if list_his_g < len( list_points_g ):
            list_his_g += 1
            cv2.circle( img, list_points_g[ -1 ], img.shape[ 0 ] // 100, ( 0, 0, 255 ), 2 )
        if len( list_points_g ) == 4:
            saveConfig( img, args.size )
            break
        if cv2.waitKey( 20 ) == 27:
            break
    cv2.destroyAllWindows()