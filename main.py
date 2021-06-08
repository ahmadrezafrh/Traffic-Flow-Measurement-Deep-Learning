import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib
import pickle






import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import pickle

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './example_2.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.30, 'iou threshold')
flags.DEFINE_float('score', 0.35, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_float('rate_sec', 0.04, 'time delay between two frames')
flags.DEFINE_integer('num', 1300, 'num o frames for detection')
flags.DEFINE_float('start_sec', 1, 'starting second for detection')

def draw_text(img,
          text,
          x,
          y,
          font=cv2.FONT_HERSHEY_COMPLEX_SMALL,
          font_scale=0.7,
          font_thickness=1,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (x,y), (x + text_w+1, y + text_h+1), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h), font, font_scale, text_color, font_thickness)
    return text_size

def main(_argv):
    # Definition of the parameters
    dist = 30
    start = 5
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    rate_sec = FLAGS.rate_sec
    num = FLAGS.num
    sec=FLAGS.start_sec
    car_dic={}
    frame_dic={}

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)
    
    print(f"video Fps:{vid.get(cv2.CAP_PROP_FPS)}") 

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))


    def getFrame(sec):
        vid.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = vid.read()

        return hasFrames, image

    success, _ = getFrame(sec)


    model = 'classifier.pkl'
    scaler = 'scaler_loc.pkl'

    with open(model, 'rb') as f:
        clf = pickle.load(f)

    with open(scaler, 'rb') as f:
        scaler = pickle.load(f)

    frame_num = 0
  
    while True:
        ids = []
        sec = sec + rate_sec
        sec = round(sec, 2)
        success, frame= getFrame(sec)

        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        frame_num +=1
        print(f'Frame {frame_num+1}: ')
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        #allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)


        frame_dic={}
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (166,80,112), 2)
            #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            #cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            y = (int(bbox[3]) + int(bbox[1]))/2
            x = (int(bbox[2]) + int(bbox[0]))/2
            if frame_num>2 : 
                car_id = track.track_id
                frame_dic[car_id] = {'x' : x, 'y' : y}
                car_dic[frame_num-2] = frame_dic
                # print(car_dic)


        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))


        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        x =[]
        y = []
        displacement = []
        speed = []
        label = []
        inf = []
        # print(len(car_dic.values()))
        if (len(car_dic.values())-1)%(start+dist-1) == 0 and len(car_dic.values())-1 != 0:

            for i in car_dic[start]:
                for n in car_dic[start+dist]:
                    if i==n:

                        if (car_dic[start+dist][i]['x'] - car_dic[start][i]['x'])!=0:
                            x.append(car_dic[start+dist][i]['x'])
                            y.append(car_dic[start+dist][i]['y'])
                            d = ((car_dic[start+dist][i]['x']-car_dic[start][i]['x'])**2+(car_dic[start+dist][i]['y']-car_dic[start][i]['y'])**2)**0.5
                            displacement.append(((car_dic[start+dist][i]['x']-car_dic[start][i]['x'])**2+(car_dic[start+dist][i]['y']-car_dic[start][i]['y'])**2)**0.5)
                            speed.append(d/(rate_sec*(dist)))
                            ids.append(i)

                        if (car_dic[start+dist][i]['x'] - car_dic[start][i]['x'])==0 and (car_dic[start+dist][i]['y'] - car_dic[start][i]['y'])<0:
                            x.append(car_dic[start+dist][i]['x'])
                            y.append(car_dic[start+dist][i]['y'])
                            d = ((car_dic[start+dist][i]['x']-car_dic[start][i]['x'])**2+(car_dic[start+dist][i]['y']-car_dic[start][i]['y'])**2)**0.5
                            displacement.append(((car_dic[start+dist][i]['x']-car_dic[start][i]['x'])**2+(car_dic[start+dist][i]['y']-car_dic[start][i]['y'])**2)**0.5)
                            speed.append(d/rate_sec)
                            ids.append(i)

                        if (car_dic[start+dist][i]['x'] - car_dic[start][i]['x'])==0 and (car_dic[start+dist][i]['y'] - car_dic[start][i]['y'])>0:
                            d = ((car_dic[start+dist][i]['x']-car_dic[start][i]['x'])**2+(car_dic[start+dist][i]['y']-car_dic[start][i]['y'])**2)**0.5
                            displacement.append(((car_dic[start+dist][i]['x']-car_dic[start][i]['x'])**2+(car_dic[start+dist][i]['y']-car_dic[start][i]['y'])**2)**0.5)
                            speed.append(d/rate_sec)
                            x.append(car_dic[start+dist][i]['x'])
                            y.append(car_dic[start+dist][i]['y'])
                            ids.append(i)

                        if (car_dic[start+dist][i]['x'] - car_dic[start][i]['x'])==0 and (car_dic[start+dist][i]['y'] - car_dic[start][i]['y'])==0:
                            displacement.append(0)
                            speed.append(0)
                            x.append(car_dic[start+dist][i]['x'])
                            y.append(car_dic[start+dist][i]['y'])
                            ids.append(i)


            for i in car_dic[start+dist]:
                if i not in ids:
                    x.append(car_dic[start+dist][i]['x'])
                    y.append(car_dic[start+dist][i]['y'])
                    speed.append(None)
                    displacement.append(None)


            all_data = {'x' : x,
                    'y' : y,
                    'speed' : speed,
                    'displacement' : displacement,
                    }

            df_unscaled = pd.DataFrame(all_data)
            
            df = pd.DataFrame(scaler.transform(df_unscaled[['x', 'y']]))
            df = df.rename(columns={0 : 'x', 1 : 'y'})
            

            # df.to_csv(r'cars_detail.csv')


            X = np.array(df[['x','y']])
            Y = clf.predict(X)
            label = list(Y)
            all_data['class'] = label

            df_unscaled = pd.DataFrame(all_data)

            num_classes = df_unscaled.pivot_table(index=['class'], aggfunc='size')
            start = start + dist
            # for i in range(len(num_classes)):
            #     df_class = df_unscaled[df_unscaled["class"] == num_classes.index[i]]
            #     avg_speed = int(df_class['speed'].mean())
            #     avg_displacement = int(df_class['displacement'].mean()) 
            #     x_mean = int(df_class['x'].mean())
            #     y_mean = int(df_class['y'].mean())
            #     immovable = len(df_class[df_class["speed"] == 0])

            #     text_cluster = f"Count: {num_classes.iloc[i]}"
            #     text_speed = f"Average speed: {avg_speed} p/s"
            #     text_displacement = f'Average displacement: {avg_displacement} p/s'
            #     text_imovable = f'immovable cars: {immovable}'
            #     text = text_cluster + text_speed + text_displacement + text_imovable
            #     w, h = draw_text(frame, text_cluster, x_mean, y_mean, text_color=(255, 175, 0), text_color_bg=(100, 100, 100))
            #     y_mean = y_mean + h + 2
            #     draw_text(frame, text_speed, x_mean, y_mean, text_color=(255, 175, 0), text_color_bg=(100, 100, 100))
            #     y_mean = y_mean + h + 2
            #     w, h = draw_text(frame, text_displacement, x_mean, y_mean, text_color=(255, 175, 0), text_color_bg=(100, 100, 100))
            #     y_mean = y_mean + h + 2
            #     draw_text(frame, text_imovable, x_mean, y_mean, text_color=(255, 175, 0), text_color_bg=(100, 100, 100))



            # plt.scatter(X[:, 0], X[:, 1], c = list(Y))
            # plt.legend()
            # plt.ylim(max(plt.ylim()), min(plt.ylim()))
            # plt.show()
            # cv2.putText(frame, f"numbeer of cars in line {list(counts.keys())[0]}: {list(counts.values())[0]}", (00, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200, 100, 45), 1)
            # cv2.putText(frame, f"numbeer of cars in line {list(counts.keys())[2]}: {list(counts.values())[2]}", (50, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (150, 200, 60), 1)
            # cv2.putText(frame, f"numbeer of cars in line {list(counts.keys())[3]}: {list(counts.values())[3]}", (100, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (56, 150, 100), 1)
        try:
            for i in range(len(num_classes)):
                df_class = df_unscaled[df_unscaled["class"] == num_classes.index[i]]
                x_mean = int(df_class['x'].mean())
                y_mean = int(df_class['y'].mean())
                df_class = df_class.dropna()
                # print(df_class)
                try:
                    avg_displacement = int(df_class['displacement'].mean()) 
                    avg_speed = int(df_class['speed'].mean())
                except ValueError:
                    continue
                immovable = len(df_class[df_class["speed"] <= 6])
                text_cluster = f"Count: {num_classes.iloc[i]}"
                text_speed = f"Average speed: {avg_speed} p/s"
                # text_displacement = f'Average displacement in {rate_sec} seconds: {avg_displacement} pixels'
                text_imovable = f'Immovable cars: {immovable}'
                # text = text_cluster + text_speed + text_displacement + text_imovablw
                w, h = draw_text(frame, text_cluster, x_mean, y_mean, text_color=(255, 175, 0), text_color_bg=(100, 100, 100))
                y_mean = y_mean + h + 2
                draw_text(frame, text_speed, x_mean, y_mean, text_color=(255, 175, 0), text_color_bg=(100, 100, 100))
                # y_mean = y_mean + h + 2
                # w, h = draw_text(frame, text_displacement, x_mean, y_mean, text_color=(255, 175, 0), text_color_bg=(100, 100, 100))
                y_mean = y_mean + h + 2
                draw_text(frame, text_imovable, x_mean, y_mean, text_color=(255, 175, 0), text_color_bg=(100, 100, 100))
        except UnboundLocalError:
            continue

        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite('example.jpg', result)
        cv2.imshow("Output Video", result)
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cv2.destroyAllWindows()


    


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass












