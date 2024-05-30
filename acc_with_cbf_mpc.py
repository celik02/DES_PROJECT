#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard.

# pylint: disable=protected-access

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    TAB          : change sensor position
    `            : next sensor
    [1-9]        : change to sensor [1-9]
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function
from navigation.simple_agent import SimpleAgent
from control import mpc_exec
from scipy.optimize import least_squares
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenarios.follow_leading_vehicle import FollowLeadingVehicle
# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
from collections import deque
# import os
# import sys
# os.environ["CARLA_ROOT"] = "/home/rastic-admin/Carla_v12/CARLA_0.9.12"
# carla_root = os.getenv('CARLA_ROOT')
#
# if carla_root:
#     carla_python_api_path = os.path.join(carla_root, 'PythonAPI')
#     if carla_python_api_path not in sys.path:
#         sys.path.append(carla_python_api_path)


import carla
from carla import ColorConverter as cc
import csv
import argparse
import collections
import datetime
import logging
import math
import time
import random
from srunner.utilities.utils import (find_weather_presets, get_actor_display_name,
                             calculate_boundaries,throttle_brake_mapping1)
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q

except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')
import timeit

# ==============================
from casadi import *
import queue
from control import motion, dynamics

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================
def wrap_to_pi(x: float) -> float:
    """Wrap the input radian to (-pi, pi]. Note that -pi is exclusive and +pi is inclusive.

    Args:
        x (float): radian.

    Returns:
        The radian in range (-pi, pi].
    """
    angles = x
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


def circle_residuals(params, points):
    xc, yc, r = params
    return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2) - r


def fit_circle(points):
    # Initial guess for parameters: center at mean of points, radius as half of maximum distance from center
    xc_initial = np.mean(points[:, 0])
    yc_initial = np.mean(points[:, 1])
    r_initial = np.max(np.sqrt((points[:, 0] - xc_initial) ** 2 + (points[:, 1] - yc_initial) ** 2)) / 2

    params_initial = [xc_initial, yc_initial, r_initial]

    result = least_squares(circle_residuals, params_initial, args=(points,))
    xc, yc, r = result.x
    return xc, yc, r

class World(object):
    def __init__(self, carla_world, hud):
        self.world = carla_world
        self.map = carla_world.get_map()
        self.mapname = carla_world.get_map().name
        self.hud = hud
        self.world.on_tick(hud.on_world_tick)
        self.world.wait_for_tick(10.0)
        self.player = None
        while self.player is None:
            print("Scenario not yet ready")
            time.sleep(1)
            possible_vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == "hero":
                    self.player = vehicle
                else:
                    self.player1 = vehicle
        self.vehicle_name = self.player.type_id
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.set_sensor(
            0, notify=False)  # Change sensor type
        self.controller = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        # self.predicted_dist = self.camera_manager._prediciton

    def restart(self):
        cam_index = self.camera_manager._index
        cam_pos_index = self.camera_manager._transform_index
        start_pose = self.player.get_transform()
        start_pose.location.z += 2.0
        start_pose.rotation.roll = 0.0
        start_pose.rotation.pitch = 0.0
        # blueprint = self._get_random_blueprint()
        blueprint = self.world.get_blueprint_library().find("vehicle.audi.etron")

        self.destroy()
        self.player = self.world.spawn_actor(blueprint, start_pose)
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager._transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):

        if len(self.world.get_actors().filter(self.vehicle_name)) < 1:
            print("Scenario ended -- Terminating")
            return False

        self.hud.tick(self, self.mapname, clock)
        return True

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, mapname, clock):
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame_number - 200]
                     for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        ################################################################
        speed_ = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        time_stamp_ = datetime.timedelta(seconds=float(self.simulation_time))
        # write2csv(speed_, time_stamp_)
        ######################

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(
                world.player, truncate=20),
            'Map:     % 20s' % mapname,
            'Simulation time: % 12s' % datetime.timedelta(
                seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 *
                                       math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' %
                                (t.location.x, t.location.y)),
            'Height:  % 18.0f m' % t.location.z,
            '',
            ('Throttle:', c.throttle, 0.0, 1.0),
            ('Steer:', c.steer, -1.0, 1.0),
            ('Brake:', c.brake, 0.0, 1.0),
            ('Reverse:', c.reverse),
            ('Hand brake:', c.hand_brake),
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)
        ]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

            def distance(l):
                return math.sqrt((l.x - t.location.x) **
                                 2 + (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2)

            vehicles = [(distance(x.get_location()), x)
                        for x in vehicles if x.id != world.player.id]
            for d, vehicle in vehicles:  # sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
        self._notifications.tick(world, clock)

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30)
                                  for x, y in enumerate(item)]
                        pygame.draw.lines(
                            display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255),
                                         rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(
                        item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 *
                    self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._history = []
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self._history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self._hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self._history.append((event.frame_number, intensity))
        if len(self._history) > 4000:
            self._history.pop(0)



# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================
class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        text = ['%r' % str(x).split()[-1]
                for x in set(event.crossed_lane_markings)]
        # self._hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._surface = None
        self._parent = parent_actor
        self._hud = hud
        self._recording = False
        # self._camera_transforms = [
        #     carla.Transform(carla.Location(x=1.6, z=1.7)),
        #     carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))]
        self._camera_transforms = [
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            carla.Transform(carla.Location(x=0, y=0.0, z=1.7), carla.Rotation(yaw=0.0))]

        self._transform_index = 1
        self._sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth,
             'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw,
             'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self._sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            item.append(bp)
        self._index = None
        self.image_queue = queue.Queue()
        self._prediction = None

    def toggle_camera(self):
        self._transform_index = (
                                        self._transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(
            self._camera_transforms[self._transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self._sensors)
        needs_respawn = True if self._index is None \
            else self._sensors[index][0] != self._sensors[self._index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self._surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self._sensors[index][-1],
                self._camera_transforms[self._transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self._hud.notification(self._sensors[index][2])
        self._index = index

    def next_sensor(self):
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        self._recording = not self._recording
        self._hud.notification('Recording %s' %
                               ('On' if self._recording else 'Off'))

    def render(self, display):
        if self._surface is not None:
            display.blit(self._surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self._sensors[self._index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self._hud.dim) / 100.0
            lidar_data += (0.5 * self._hud.dim[0], 0.5 * self._hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self._hud.dim[0], self._hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self._surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self._sensors[self._index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self._recording:
            image.save_to_disk('_out/%08d' % image.frame_number)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    T = 15  # 52 with default speed # clash at 106s
    dt = 0.01
    time_steps = int(np.ceil(T / dt))
    i = 0
    data = {"index" : [], "throttle" : [], "brake": [], "model_acc" : [], "ego_vel": [],  "target_vel": [],
            "model_vel": [],"actual_acc": [], "actual_x": [], "actual_y": [], "actual_heading":[],
            "model_x": [], "model_y": [], "model_heading":[], "x_ip" : [], "y_ip": [], "psi_ip" : [],
            "v_ip" : [], "acc_ip" :[]}
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud)

        agent = SimpleAgent(world.player)
        physics_control = world.player.get_physics_control()
        physics_control.mass = 1650

        spawn_points = world.map.get_spawn_points()
        random.shuffle(spawn_points)

        clock = pygame.time.Clock()
        pre_target_yaw = 3.14
        last_psi_ref_path = 3.14
        pre_yaw = 3.14
        pre_phi_d = 0

        ######## celik
        from sensor_fusion import DisplayManager, SensorManager, cam_intrinsic_odom
        import pickle 
        from ultralytics import YOLO
        model = YOLO('best.pt')

        display_manager = DisplayManager(grid_size=[2, 2], window_size=[1280, 720])
        cam = SensorManager(world.world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)), 
                        world.player, {}, display_pos=[0, 0])
        
        depth_cam = SensorManager(world.world, display_manager, 'DepthCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)), 
                        world.player, {}, display_pos=[0, 1])
        
        lidar = SensorManager(world.world, display_manager, 'SemanticLiDAR', carla.Transform(carla.Location(x=0,y=0, z=2.4)), 
                        world.player, {'channels' : '64', 'range' : '150', 'points_per_second': '100000', 'rotation_frequency': '100'}, display_pos=[1, 0])

        sensors = display_manager.get_sensor_list()

        camera_rgb = sensors[0]
        lidar_semantic = sensors[1]
        call_exit = False
        fusion_model = pickle.load(open('LR_model.sav', 'rb'))
        previous_prediction = np.array([[world.player1.get_location().x, world.player1.get_location().y, world.player1.get_location().z]])
        moving_avg_vel = 0
        alpha = 0.95
        
        settings = world.world.get_settings()
        original_settings = world.world.get_settings()
        settings.fixed_delta_seconds = 0.1 # Set a variable time-step
        settings.synchronous_mode = True
        sync_true = True
        world.world.apply_settings(settings)
        moving_avg_vel = 0
        alpha = 0.95
        ##########################################################################3
        while True:
            ########celik #######################################################################################
            #####################################################################################################
            display_manager.render()
            results = model.predict(cam.image, show=True, conf=0.75)
            dist_cam = cam_intrinsic_odom(results, cam.get_sensor(), depth_cam.depth_image, world.player)
            how_cam_off = np.array([dist_cam[0,0] - previous_prediction[0,0] , dist_cam[1,0] - previous_prediction[0,1],
                                                     dist_cam[2,0] - previous_prediction[0,2]])
            if (dist_cam[0,0] == 0 and dist_cam[1,0] == 0) or np.absolute(how_cam_off/0.1).any() > 30:
                print('***************** camera zero')
                X = np.array([[lidar.detected_vehicle[0,0], lidar.detected_vehicle[1,0], lidar.detected_vehicle[2,0],
                            lidar.detected_vehicle[0,0], lidar.detected_vehicle[1,0],lidar.detected_vehicle[2,0]]])
            else:
                X = np.array([[lidar.detected_vehicle[0,0], lidar.detected_vehicle[1,0], lidar.detected_vehicle[2,0],
                            dist_cam[0,0], dist_cam[1,0], dist_cam[2,0]]])
                
            prediction = fusion_model.predict(X)
                
            print('Prediction:', prediction)
            print('GT:', world.player1.get_location())
            if sync_true:
                world.world.tick()
            else:
                world.world.wait_for_tick()

            vel = (prediction -  previous_prediction)/0.1

            eucl_vel = np.linalg.norm(vel)
            moving_avg_vel = alpha*eucl_vel + (1-alpha)*moving_avg_vel
            print('Euclidian Vel', moving_avg_vel)
            print('Velocity:', vel)
            previous_prediction = prediction
            # pygame.display.flip()  ### renders the original pygame window
            #######################################################################################################
            ##########################################################################################################3

            start_time = time.time()
            # clock.tick_busy_loop(60)
            # if not world.world.wait_for_tick(10.0):
            #     continue

            # if (not world.tick(clock)):
            #     print("Failure: scenario terminated")
            #     break

            # world.render(display)

            

            control, target_vehicle = agent.run_step()
            waypoint = world.map.get_waypoint(world.player.get_location(), project_to_road=True,
                                              lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))

            ego_vel = world.player.get_velocity()
            ego_vel_transform = np.sqrt(
                ego_vel.x ** 2 + ego_vel.y ** 2 + ego_vel.z ** 2)
            ego_loc = world.player.get_location()
            ego_yaw = np.deg2rad(world.player.get_transform().rotation.yaw)

            ego_yaw = (ego_yaw - pre_yaw + np.pi) % (2 * np.pi) - np.pi + pre_yaw
            states = [ego_loc.x, ego_loc.y, ego_yaw, ego_vel_transform]
            ego_acceleration = world.player.get_acceleration()
            ego_acceleration_transform = ego_acceleration.x
            pre_yaw = ego_yaw

            x_centerline = []
            y_centerline = []
            desired_phi = []

            for _ in range(0, 5):
                waypoint = waypoint.next(1)[0]
            phi_d = np.arctan2(waypoint.transform.location.y - ego_loc.y, waypoint.transform.location.x - ego_loc.x)
            phi_d = (phi_d - pre_phi_d + np.pi) % (2 * np.pi) - np.pi + pre_phi_d
            pre_phi_d = phi_d
            waypoint = world.map.get_waypoint(world.player.get_location(), project_to_road=True,
                                          lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))


            for _ in range(0, 20):
                x_centerline.append(waypoint.transform.location.x)
                y_centerline.append(waypoint.transform.location.y)
                phi = np.deg2rad(waypoint.transform.rotation.yaw)
                phi = (phi - pre_yaw + np.pi) % (2 * np.pi) - np.pi + pre_yaw
                desired_phi.append(phi)
                waypoint = waypoint.next(1)[0]

            desired_phi[0] = 0.5 * phi_d + 0.5*desired_phi[0]

            centerline = np.column_stack((x_centerline, y_centerline))

            xc_fit, yc_fit, r_fit = fit_circle(centerline)
            target_vel = world.player1.get_velocity()
            target_vel_transform = np.sqrt(
                target_vel.x ** 2 + target_vel.y ** 2 + target_vel.z ** 2)
            target_loc = world.player1.get_location()
            target_yaw = np.deg2rad(world.player1.get_transform().rotation.yaw)
            target_yaw = (target_yaw - pre_target_yaw + np.pi) % (2 * np.pi) - np.pi + pre_target_yaw


            # states_ip = [target_loc.x, target_loc.y, target_yaw, target_vel_transform]
            #####################################################################################################################################celik
            states_ip = [prediction[0,0], prediction[0,1], target_yaw, moving_avg_vel]
            ##########################################################################################################################################
            pre_target_yaw = target_yaw


            status, action, next_states, model_acc = mpc_exec(desired_phi, states,
                                      states_ip, i, r_fit)
            last_psi_ref_path = desired_phi[0]


            a = float(action[0])
            throttle, brake = throttle_brake_mapping1(a)
            control.throttle = throttle
            control.brake = brake
            control.steer = action[1]
            data["index"].append(i)
            data["throttle"].append(throttle)
            data["brake"].append(brake)
            data["model_acc"].append(model_acc)
            data["ego_vel"].append(ego_vel_transform)
            data["target_vel"].append(target_vel_transform)
            data["model_vel"].append(next_states[3])
            data["actual_acc"].append(ego_acceleration_transform)
            data["actual_x"].append(states[0])
            data["actual_y"].append(states[1])
            data["actual_heading"].append(states[2])
            data["model_x"].append(next_states[0])
            data["model_y"].append(next_states[1])
            data["model_heading"].append(next_states[2])
            data["x_ip"].append(states_ip[0])
            data["y_ip"].append(states_ip[1])
            data["psi_ip"].append(states_ip[2])
            data["v_ip"].append(states_ip[3])
            data["acc_ip"].append(world.player1.get_acceleration().x)
            world.player.apply_control(control)

            i = i + 1
            if i == time_steps - 1:
                print('Exceed itertation')
                break

    finally:
        with open('data.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header row
            writer.writerow(data.keys())
            # Write values row by row
            for row in zip(*data.values()):
                writer.writerow(row)
        world.world.apply_settings(original_settings)
        if world is not None:
            world.destroy()
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',  # '127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as error:
        logging.exception(error)


if __name__ == '__main__':
    main()
