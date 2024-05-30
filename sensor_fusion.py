import carla
import glob
import os
import sys
import matplotlib.pyplot as plt
import math
import csv
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

import time
import numpy as np
from ultralytics import YOLO
import pickle
model = YOLO('best.pt')

# to save data
filename_cam = "cam1.csv"
filename_lidar = "lidar1.csv"
filename_gt = 'gt1.csv'
fcam = open(filename_cam, 'a', newline='')
flidar = open(filename_lidar, 'a', newline='')
fgt= open(filename_gt, 'a', newline='')
cam_writer = csv.writer(fcam)
lidar_writer = csv.writer(flidar)
gt_writer = csv.writer(fgt)

class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()

def get_world_matrix(x, y, z, roll, pitch, yaw):

    """
    This function gets the x,y,z which is the center of the new frame, for example vehicle frame and roll, pitch, yaw
    then returns the rotation + translaion matrix for later on convetring any point to the global frame. 
    
    """
    c_y = np.cos(yaw)
    s_y = np.sin(yaw)
    c_r = np.cos(roll)
    s_r = np.sin(roll)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)

    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix

class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0]/self.grid_size[1]), int(self.window_size[1]/self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return

        for s in self.sensor_list:
            s.render()

        pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display != None

class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos):
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()
        self.image = np.array([])
        self.time_processing = 0.0
        self.tics_processing = 0
        self.attached = attached

        self.display_man.add_sensor(self)

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        if sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str(disp_size[0]))
            camera_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.save_rgb_image)

            return camera
        
        if sensor_type == 'DepthCamera':
            depth_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
            disp_size = self.display_man.get_display_size()
            depth_bp.set_attribute('image_size_x', str(disp_size[0]))
            depth_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options:
                depth_bp.set_attribute(key, sensor_options[key])

            depth = self.world.spawn_actor(depth_bp, transform, attach_to=attached)
            depth.listen(self.save_depth_image)

            return depth

        elif sensor_type == 'LiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('range', '100')
            lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit', lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)

            lidar.listen(self.save_lidar_image)

            return lidar
        
        elif sensor_type == 'SemanticLiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('range', '100')
    
            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)

            lidar.listen(self.save_semanticlidar_image)

            return lidar
        
        else:
            return None

    def get_sensor(self):
        return self.sensor

    def save_rgb_image(self, image): # callback function for camera sensor  -> sensor.listen calls this fuction
        t_start = self.timer.time()
        
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.image = array

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
    
    def save_depth_image(self, depth_image): # callback function for depth camera sensor  -> sensor.listen calls this fuction
        t_start = self.timer.time()
        
        depth_image.convert(carla.ColorConverter.Depth)
        array = np.frombuffer(depth_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (depth_image.height, depth_image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.depth_image = array

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def save_lidar_image(self, image):  # callback function for lidar sensor
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def save_semanticlidar_image(self, image):  # callback function for semantic lidar
        t_start = self.timer.time()
        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.float32)
        #################################################################################################################
        X = []
        Y = []
        Z = []
        num_point = 0
        ### relative position of lidar wrt to vehicle
        tx=0
        ty=0
        tz=2.4
        xp, yp, zp = self.attached.get_location().x,self.attached.get_location().y, self.attached.get_location().z
        Rworld_lidar = get_world_matrix(xp + tx, yp + ty, zp + tz, 0, 0, 0)
        # print(Rworld_lidar)
        # once = True
        for point in image:
            # print(point)
            # print('Attached vehicle id:', self.attached.id)
            if point.object_tag ==14 and point.object_idx!=self.attached.id:
                # print('x:', point.point.x, 'y:', point.point.y, 'z:', point.point.z)
                # print('Attached vehicle id:', self.attached.id)
                # print(point)
                # if once:
                #     once = False
                    # continue
                X.append(point.point.x)
                Y.append(point.point.y)
                Z.append(point.point.z)
        if len(X)>0:
            X = sum(X)/len(X)
            Y = sum(Y)/len(Y)
            Z = sum(Z)/len(Z)
            vector = np.array([ X, Y, Z, 1 ]).reshape( (4, 1) )
            world_coordinates = Rworld_lidar@vector
            self.detected_vehicle = world_coordinates
        else:
            print('[INFO] vehicle out of range')

        
        
        # self.lidar_dist = np.sqrt(X**2 + Y**2 + Z**2)
        #################################################################################################################
        points = np.reshape(points, (int(points.shape[0] / 6), 6))

        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1 


    def render(self): # displays the data on pygame window
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()


def cam_intrinsic_odom(results,cam_sensor, depth_image, ego ):
    '''
    1- calculate middle point from Yolo bounding box
    2- Get depth information for that middle point.
    3- Convert image coordinates to world coordinates via intrinsic camera matrix
    4- 
    '''
    in_meters = None
    object_in_world_CF = np.zeros((4,1))
    for r in results:
        if r.boxes.cls.numel() > 0:
            # print(r.boxes.cls.cpu().detach())
            if r.boxes.cls.cpu().detach().any() == 0:
                # print('Vehicle detected--')
                box_loc = r.boxes.xywh.cpu().detach().numpy()
                # calculate the middle point
                x = int(box_loc[0,0]+ box_loc[0,2]/2)
                y = int(box_loc[0,1]+ box_loc[0,3]/2)

                Depth_ROI = depth_image[y-1, x-1,  :]            
                # R, G, B = np.average(Depth_ROI[:,:,0]), np.average(Depth_ROI[:,:,1]), np.average(Depth_ROI[:,:,2])
                R,G,B = Depth_ROI[0], Depth_ROI[1], Depth_ROI[2]

                normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
                in_meters = 1000 * normalized
                state = np.array([x, y, in_meters])

                object_in_world_CF = camera_2_world(cam_sensor, ego, state)
    return object_in_world_CF
    
def camera_2_world( cam_sensor, ego, state):
    # K = np.array(cam_sensor.K).reshape((3,3))
    # print(K)
    pixel_coordinates = np.array([state[0], state[1], 1])
    K = get_camera_intrinsic_matrix(cam_sensor)
    # # to have North east down NED 
    transformation_matrix = np.array([[0, 1, 0], 
                                        [0, 0, -1], 
                                        [1, 0, 0] ])
    in_cam_frame = np.dot(np.linalg.inv(transformation_matrix), np.dot(np.linalg.inv(K),pixel_coordinates*state[2]))
    print('Cam frame coordinates:', in_cam_frame)
    sensor_2_world_RT = np.array(ego.get_transform().get_matrix())
    # print('S2W:', sensor_2_world_RT)
    # print('K:',K)
    camera_vector = np.ones((4,1))
    camera_vector[:3, :] = in_cam_frame.reshape((3,1))
    return np.dot(sensor_2_world_RT,camera_vector)

def get_camera_intrinsic_matrix(camera):
    # Retrieve camera attributes
    image_width = camera.attributes.get('image_size_x')
    image_height = camera.attributes.get('image_size_y')
    fov = float(camera.attributes.get('fov'))

    # Convert FOV to focal length in pixels
    focal_length = (float(image_width) / 2.0) / math.tan(fov * math.pi / 360.0)

    # Principal point that is usually at the image center
    c_x = float(image_width) / 2.0
    c_y = float(image_height) / 2.0

    # Assuming no lens skew
    skew = 0

    # Construct the intrinsic matrix
    K = [[focal_length, skew, c_x],
         [0, focal_length, c_y],
         [0, 0, 1]]

    return K

def get_initial_carla_pos(actor):
    loc =  actor.get_location()
    return loc.x, loc.y

def get_depth_from_cam(results, depth_image):
    '''
    given the depth data and ROI calculate depth
    formula:
    normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
    in_meters = 1000 * normalized

    This cannot cover x,y,z coordinates. Just depth is calculated. 
    '''
    in_meters = 0
    for r in results:
        if r.boxes.cls.numel() > 0:
            # print(r.boxes.cls.cpu().detach())
            if r.boxes.cls.cpu().detach().any() == 0:
                print('Vehicle detected--')
                box_loc = r.boxes.xywh.cpu().detach().numpy()
                # print('Object location', box_loc)
                x = int(box_loc[0,0]+ box_loc[0,2]/4)
                y = int(box_loc[0,1]+ box_loc[0,3]/4)
                w = int(box_loc[0,2]/2)
                h = int(box_loc[0,3]/2)
                # print(x,y,w,h)
                ## calculate distance of vehicle in front from depth image
                Depth_ROI = depth_image[y:y+h, x:x+w,  :]
                R, G, B = np.average(Depth_ROI[:,:,0]), np.average(Depth_ROI[:,:,1]), np.average(Depth_ROI[:,:,2])

                normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
                in_meters = 1000 * normalized


    return in_meters

index = 0
def save_data(lidar, cam, gt):
    global cam_writer
    global lidar_writer
    global gt_writer
    global index
    index += 1
    gt = np.array([index, gt.x, gt.y, gt.z])
    cam = cam.flatten()
    cam_i = np.insert(cam, 0, index)
    lidarD = np.array([index, lidar[0,0], lidar[1,0], lidar[2,0]])
    print('Lidar:',lidarD)
    print('Cam:  ', cam_i)
    print('GT:   ', gt)
    
    lidar_writer.writerow(lidarD)
    cam_writer.writerow(cam)
    gt_writer.writerow(gt)

if __name__ == '__main__':
    client = carla.Client('localhost',2000)
    client.set_timeout(10.0)
    world = client.get_world()
    settings = world.get_settings()
    original_settings = world.get_settings()
    settings.fixed_delta_seconds = 0.1 # Set a variable time-step
    settings.synchronous_mode = True
    world.apply_settings(settings)
    actor_list = world.get_actors()
    id = 0

    for actor in actor_list.filter('*model3*'):
        id = actor.id
        vehicle = actor
    ego = actor_list.find(id)    

    for actor in actor_list.filter('*nissan*'):
        idx = actor.id
    nissan_patrol = actor_list.find(idx)

    init_x, init_y = get_initial_carla_pos(ego)

    sync_true = True # if syncronous mode is used
    display_manager = DisplayManager(grid_size=[2, 2], window_size=[1280, 720])
    cam = SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)), 
                      vehicle, {}, display_pos=[0, 0])
    
    depth_cam = SensorManager(world, display_manager, 'DepthCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)), 
                      vehicle, {}, display_pos=[0, 1])
    
    lidar = SensorManager(world, display_manager, 'SemanticLiDAR', carla.Transform(carla.Location(x=0,y=0, z=2.4)), 
                      vehicle, {'channels' : '64', 'range' : '100', 'points_per_second': '100000', 'rotation_frequency': '20'}, display_pos=[1, 0])
    
    sensors = display_manager.get_sensor_list()

    camera_rgb = sensors[0]
    lidar_semantic = sensors[1]
    call_exit = False
    fusion_model = pickle.load(open('LR_model.sav', 'rb'))
    previous_prediction = np.array([[nissan_patrol.get_location().x, nissan_patrol.get_location().y, nissan_patrol.get_location().z]])
    moving_avg_vel = 0
    alpha = 0.95
    while True:
            # Carla Tick
            if sync_true:
                world.tick()
            else:
                world.wait_for_tick()
                
            # Render received data
            display_manager.render()
            if cam.image.any():
                # print('showing image')
                # plt.imshow(cam.image)
                # plt.show(block=False)
                # plt.pause(0.01)            
                results = model.predict(cam.image, show=True, conf=0.85)
                # dist_cam = get_depth_from_cam(results, depth_cam.depth_image)
                dist_cam = cam_intrinsic_odom(results, cam.get_sensor(), depth_cam.depth_image, ego)
                print('lidar data Dist:', ego.get_location().distance(carla.Vector3D(x=lidar.detected_vehicle[0,0],
                                                                                                      y = lidar.detected_vehicle[1,0],
                                                                                                        z=lidar.detected_vehicle[2,0])))
                print('Camera dist::', dist_cam)
                print('Camera dist::', ego.get_location().distance(carla.Vector3D(x=dist_cam[0,0], y = dist_cam[1,0],
                                                                                z=dist_cam[2,0])))
                print('Ground Truth Distance: ', ego.get_location().distance(nissan_patrol.get_location()))

                ### Sensor Fusion prediction
                how_cam_off = np.array([dist_cam[0,0] - previous_prediction[0,0] , dist_cam[1,0] - previous_prediction[0,1],
                                                     dist_cam[2,0] - previous_prediction[0,2]])
                
                print('How of camera is:', how_cam_off)
                if (dist_cam[0,0] == 0 and dist_cam[1,0] == 0) or np.absolute(how_cam_off/0.1).any() > 30:
                     print('***************** camera zero')
                     X = np.array([[lidar.detected_vehicle[0,0], lidar.detected_vehicle[1,0], lidar.detected_vehicle[2,0],
                              lidar.detected_vehicle[0,0], lidar.detected_vehicle[1,0],lidar.detected_vehicle[2,0]]])
                else:
                    X = np.array([[lidar.detected_vehicle[0,0], lidar.detected_vehicle[1,0], lidar.detected_vehicle[2,0],
                              dist_cam[0,0], dist_cam[1,0], dist_cam[2,0]]])
                
                prediction = fusion_model.predict(X)
                
                print('Prediction:', prediction)
                print('GT:', nissan_patrol.get_location())
                print('Predicted Distance:', ego.get_location().distance(carla.Vector3D(x=prediction[0,0], y=prediction[0,1], z=prediction[0,2])))
                vel = (prediction -  previous_prediction)/0.1

                eucl_vel = np.linalg.norm(vel)
                moving_avg_vel = alpha*eucl_vel + (1-alpha)*moving_avg_vel
                print('Euclidian Vel', moving_avg_vel)
                print('Velocity:', vel)
                previous_prediction = prediction
                # save_data(lidar.detected_vehicle, dist_cam, nissan_patrol.get_location())
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        call_exit = True
                        break

            if call_exit:
                break

    world.apply_settings(original_settings)
    fcam.close()
    flidar.close()
    fgt.close()