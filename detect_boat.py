import os
from utils.settings import *
from utils.utils import *
from utils.check_night import *
from rpi_modules.commands import *

def detectBoat():
    global model, device, imgsz, names, colors
    print("STARTING DETECTION...")
    # Gather data
    if picture: 
        takePic()
        dataset = LoadImages(input_dir, img_size=imgsz)
    else:
        takeVideo()
        dataset = LoadImages(input_dir, img_size=imgsz)
    
    # Run inference
    boats = [] # boats detected per image 
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        pred = model(img, augment=False)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=None, agnostic=False)
  
        # Process detections
        for i, det in enumerate(pred):
            p, s, im0 = path, '', im0s
            save_path = str(Path(save_dir) / Path(p).name)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] # normalization gain 
           
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss' % (n, names[int(c)])  # add to string
                    print(s)
                    
                # Write results
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    if save:
                        if dataset.mode == 'images':
                            cv2.imwrite(save_path, im0)
                        else:
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  # release previous video writer

                                fourcc = 'mp4v'  # output video codec
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                            vid_writer.write(im0)
            boats.append(len(det)) # counts number of boats 
        if max(boats) >= 0:        
            notify(max(boats)) # sends to rpi-server number of boats detected
    rotate() # reorients the camera

# Initialize torch device as CPU
device = torch.device('cpu')
    
# Load model
model = attempt_load(weights)
imgsz = 640

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

while True:
    start, now, end = get_times()
    if now < start or now > end:
        with torch.no_grad():
            detectBoat()
