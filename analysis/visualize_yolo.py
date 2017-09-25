import argparse
import itertools
import cv2
import pandas as pd
from noscope.DataUtils import get_labels

colors = [(0x33, 0x22, 0x88),
          (0x88, 0xCC, 0xEE),
          (0x44, 0xAA, 0x99),
          (0x11, 0x77, 0x33),
          (0x99, 0x99, 0x33),
          (0xDD, 0xCC, 0x77),
          (0xCC, 0x66, 0x77),
          (0x88, 0x22, 0x55),
          (0xAA, 0x44, 0x99)]

# NOTE: John's YOLO script and numpy have different representations of x,y
def draw_label(label, image, color=(0, 255, 0)):
    x, y = image.shape[0:2]
    # tl = (int(label['xmin'] * y), int(label['ymin'] * x))
    # br = (int(label['xmax'] * y), int(label['ymax'] * x))
    tl = (int(label['xmin']), int(label['ymin']))
    br = (int(label['xmax']), int(label['ymax']))

    cv2.rectangle(image, tl, br, color, 3)
    cv2.putText(image, label['object_name'] + ' ' + str(label['confidence']),
                tl, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

def objects_to_colors(objects):
    obj_to_color = {}
    color_it = itertools.cycle(colors)
    for i, obj in enumerate(objects):
        obj_to_color[obj] = color_it.next()
    return obj_to_color


def draw_video(csv_fname, video_in_fname, video_out_fname,
               nb_frames, start_frame=0,
               OBJECTS=set('person')):
    obj_to_color = objects_to_colors(OBJECTS)
    all_labels = get_labels(csv_fname, start=start_frame, limit=nb_frames, labels=OBJECTS)

    print all_labels

    print 'Processing %d frames, starting at frame %d' % (len(all_labels), start_frame)
    print 'Generating visualized file %s' % video_out_fname

    cap = cv2.VideoCapture(video_in_fname)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, frame = cap.read()

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    res = frame.shape[0:2][::-1]
    vout = cv2.VideoWriter(video_out_fname, fourcc, 30, res)

    for i in xrange(nb_frames):
        if i % 500 == 0:
            print i
        if i in all_labels.index:
            labels = all_labels.loc[i]
            try:
                label = labels
                label.object_name
                draw_label(label, frame, obj_to_color[label.object_name])
            except:
                for _, label in labels.iterrows():
                    draw_label(label, frame, obj_to_color[label['object_name']])
        # cv2.rectangle(frame, (0, 185), (700, 185), colors[1], thickness=2)
        vout.write(frame)
        ret, frame = cap.read()

    vout.release()
    cap.release()

    '''for i, labels in enumerate(all_labels):
        if i % 500 == 0:
            print i
        for label in labels:
            if label['object_name'] not in OBJECTS:
                continue
            draw_label(label, frame, obj_to_color[label['object_name']])
        vout.write(frame)
        ret, frame = cap.read()

    vout.release()
    cap.release()'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', required=True, help='CSVs with YOLO labels')
    parser.add_argument('--video_in', required=True, help='Video input filename. DO NOT confuse with video_out')
    parser.add_argument('--video_out', required=True)
    parser.add_argument('--objects', default='person,car', help='Object to draw')
    parser.add_argument('--num_frames', type=int, default=-1, help='Number of frames to label')
    parser.add_argument('--start_frame', type=int, default=0, help='Frame to start labeling')
    args = parser.parse_args()

    nb_frames = args.num_frames if args.num_frames > 0 else None
    objects = set(args.objects.split(','))
    draw_video(args.input_csv, args.video_in, args.video_out,
               nb_frames, args.start_frame, objects)


if __name__ == '__main__':
    main()
