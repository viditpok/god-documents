import threading
from tkinter import *
import time
# from setting import *
import setting
import random
# random.seed(setting.RANDOM_SEED)
import copy
import math
import queue

from environment import Environment
from utils import *
from setting import *
import json



# GUI class
class GUIWindow():
    def __init__(self, config_file_path: str):
        with open(config_file_path, "r") as file:
            configs = json.load(file)
        self.queue = queue.Queue()
        self.width = configs["gui_width"]
        self.height = configs["gui_height"]
        self.x_range = configs["x_range"]
        self.y_range = configs["y_range"]
        self.margin = 0.15 * self.width
        self.update_cnt = 0

        self.running = threading.Event()
        self.updated = threading.Event()
        self.updated.clear()
        self.lock = threading.Lock()
        # grid info
        world_file = os.path.join(WORLD_PATH, configs["world_file"])
        self.markers = read_marker_positions(world_file)
        self.wall_poses, self.wall_dimensions = read_walls(world_file)
        self.T_r_c = SE2(*configs["camera_pose"])
        self.T_r_l = SE2(*configs["lidar_pose"])

        self.particles = []
        self.robot = None
        self.weights = None
        self.marker_measures = None

    def scale_x(self, x):
        old_min, old_max = self.x_range
        return (x - old_min) / (old_max - old_min) * (self.width) + self.margin
    
    def scale_y(self, y):
        old_min, old_max = self.y_range
        return (y - old_min) / (old_max - old_min) * (self.height) - self.margin
    
    def scale(self, coord):
        x, y = coord
        return (self.scale_x(x), self.scale_y(y))

    # start GUI thread
    def start(self):
        master = Tk()
        master.wm_title("Particle Filter: Grey/Green - estimated, Red - ground truth")

        self.canvas = Canvas(master, width = self.width + 2 * self.margin, height = self.height + 2 * self.margin, bd = 0, bg = '#FFFFFF')
        self.canvas.pack()

        # self.drawGrid()
        self.drawMarkers()
        self.drawWalls()

        # Start mainloop and indicate that it is running
        self.running.set()
        while True:
            self.updated.wait()
            if self.updated.is_set():
                self.update()
            try:
                while not self.queue.empty():
                    func, args = self.queue.get()
                    func(*args)
                master.update_idletasks()
                master.update()
            except TclError:
                break

        # Indicate that main loop has finished
        self.running.clear()


    """
    plot
    """
    def drawGrid(self):
        for y in range(1,self.height):
            self.canvas.create_line(0, y, int(self.canvas.cget("height")) - 1, y)
        for x in range(1,self.width):
            self.canvas.create_line(x, 0, x, int(self.canvas.cget("width")) - 1)

    def drawMarkers(self):
        for marker in self.markers:
            self.colorCircle(self.scale((marker.x, marker.y)), color='red', dot_size= 6)

    def drawWalls(self):
        for wall_pose, wall_dim in zip(self.wall_poses, self.wall_dimensions):
            corner1_local = Point(-wall_dim[0]/2, -wall_dim[1]/2)
            corner2_local = Point(wall_dim[0]/2, wall_dim[1]/2)
            corner1_w = wall_pose.transform_point(corner1_local)
            corner2_w = wall_pose.transform_point(corner2_local)
            corner1 = self.scale((corner1_w.x, corner1_w.y))
            corner2 = self.scale((corner2_w.x, corner2_w.y))
            self.colorRectangle(corner1, corner2, "#616161")

    def weight_to_color(self, weight):
        return "#%02x00%02x" % (int(weight * 255), int((1 - weight) * 255))

    def _show_mean(self, mean_pose, confident=False):
        if confident:
            color = "#00AA00"
        else:
            color = "#CCCCCC"
        coord = self.scale((mean_pose.x,mean_pose.y))
        self.colorTriangle(coord, mean_pose.h, color=color, tri_size=18, outline='#000000')


    def _show_particles(self, particles, weights):
        idx = 0
        color = '#757575'
        while idx < setting.PARTICLE_MAX_SHOW:
            p = copy.deepcopy(particles[int(idx)])
            coord = self.scale((p.x,p.y))
            self.colorTriangle(coord, p.h, color=color, tri_size=4)
            idx += 1

    def _show_lidar_array(self, robot_pose, lidar_array):
        T_w_l = robot_pose.compose(self.T_r_l)
        for angle, dist, in enumerate(lidar_array):
            theta_rad = angle * math.pi / 180
            point_l = Point(dist * math.cos(theta_rad), dist * math.sin(theta_rad))
            point_w = T_w_l.transform_point(point_l)
            self.colorCircle(self.scale((point_w.x, point_w.y)), color='#4db6ac', dot_size = 2.5)

    def _show_robot(self, robot_pose, marker_measures):
        coord = self.scale((robot_pose.x, robot_pose.y))
        T_w_c = robot_pose.compose(self.T_r_c)
        T_w_l = robot_pose.compose(self.T_r_l)
        coord_c = self.scale((T_w_c.x, T_w_c.y))
        h = robot_pose.h
        self.colorTriangle(coord, h, '#FF0000', tri_size=12, outline='#000000')
        # plot fov
        fov_lx, fov_ly = rotate_point(0.5 * self.width, 0, h + setting.ROBOT_CAMERA_FOV / 2)
        fov_rx, fov_ry = rotate_point(0.5 * self.width, 0, h - setting.ROBOT_CAMERA_FOV / 2)
        self.colorLine(coord_c, (coord_c[0]+fov_lx, coord_c[1]+fov_ly), color='#9e9e9e', linewidth=1, dashed=True)
        self.colorLine(coord_c, (coord_c[0]+fov_rx, coord_c[1]+fov_ry), color='#9e9e9e', linewidth=1, dashed=True)
        
        if marker_measures is not None:
            for marker_measure in marker_measures:
                depth, angle, dist = marker_measure.depth, marker_measure.angle, marker_measure.lidar_range
                marker_c = Point(depth, depth * math.tan(angle))
                marker_w = T_w_c.transform_point(marker_c)
                point_l = Point(dist * math.cos(angle), dist * math.sin(angle))
                point_w = T_w_l.transform_point(point_l)
                self.colorLine(self.scale((T_w_c.x, T_w_c.y)), self.scale((marker_w.x, marker_w.y)), color='#ff8f00', linewidth=1.5, dashed=False)
                self.colorLine(self.scale((T_w_l.x, T_w_l.y)), self.scale((point_w.x, point_w.y)), color='#5e35b1', linewidth=1.5, dashed=False)

    def clean_world(self):
        self.canvas.delete("all")
        self.drawMarkers()
        self.drawWalls()

    """
    plot utils
    """
    # Draw a colored square at the specified grid coordinates
    def colorCell(self, location, color):
        coords = (location[0]*self.grid.scale, (self.height-location[1]-1)*self.grid.scale)
        self.canvas.create_rectangle(coords[0], coords[1], coords[0] + self.grid.scale, coords[1] + self.grid.scale, fill=color)

    def colorRectangle(self, corner1, corner2, color):
        coords1 =  (corner1[0], (self.height-corner1[1]))
        coords2 =  (corner2[0], (self.height-corner2[1]))
        self.canvas.create_rectangle(coords1[0], coords1[1], coords2[0], coords2[1], fill=color)

    def colorCircle(self,location, color, dot_size = 5):
        x0, y0 = location[0] - dot_size, (self.height-location[1]) - dot_size
        x1, y1 = location[0] + dot_size, (self.height-location[1]) + dot_size
        return self.canvas.create_oval(x0, y0, x1, y1, fill=color, width=0)

    def colorLine(self, coord1, coord2, color='black', linewidth=1, dashed=False):
        if dashed:
            self.canvas.create_line(coord1[0], (self.height-coord1[1]), \
                coord2[0], (self.height-coord2[1]),  \
                fill=color, width=linewidth, dash=(5,3))
        else:
            self.canvas.create_line(coord1[0], (self.height-coord1[1]), \
                coord2[0], (self.height-coord2[1]),  \
                fill=color, width=linewidth)

    def colorTriangle(self, location, heading, color, tri_size, outline=''):
        if outline is None:
            outline = '#000000'
        hx, hy = rotate_point(tri_size * 1.5, 0, heading)
        lx, ly = rotate_point(-tri_size, tri_size, heading)
        rx, ry = rotate_point(-tri_size, -tri_size, heading)
        # reverse Y here since input to row, not Y
        hrot = (hx + location[0], -hy + (self.height-location[1]))
        lrot = (lx + location[0], -ly + (self.height-location[1]))
        rrot = (rx + location[0], -ry + (self.height-location[1]))
        return self.canvas.create_polygon(hrot[0], hrot[1], lrot[0], lrot[1], rrot[0], rrot[1], \
            fill=color, outline=outline,width=1)

    """
    Sync data to plot from other thread
    """
    def show_mean(self, mean_pose, confident=False):
        self.lock.acquire()
        self.mean_pose = copy.deepcopy(mean_pose)
        self.mean_confident = confident
        self.lock.release()

    def show_particles(self, particles, weights=None):
        self.lock.acquire()
        self.particles = copy.deepcopy(particles)
        self.weights = weights
        self.lock.release()

    def show_lidar_array(self, robot, lidar_range):
        self.lock.acquire()
        self.robot = copy.deepcopy(robot)
        self.lidar_range = copy.deepcopy(lidar_range)
        self.lock.release()

    def show_robot(self, robot, marker_measures):
        self.lock.acquire()
        self.robot = copy.deepcopy(robot)
        self.marker_measures = copy.deepcopy(marker_measures)
        self.lock.release()

    def setupdate(self):
        self.updateflag = True

    def update(self):
        self.lock.acquire()
        self.queue.put((self.clean_world, []))
        self.queue.put((self._show_particles, [self.particles, self.weights]))
        self.queue.put((self._show_mean, [self.mean_pose, self.mean_confident]))
        # # self.queue.put((self._show_mean, [self.mean_x, self.mean_y, self.mean_heading, self.mean_confident]))
        
        if self.robot != None:
            self.queue.put((self._show_robot, [self.robot, self.marker_measures]))
            self.queue.put((self._show_lidar_array, [self.robot, self.lidar_range]))
        self.updated.clear()
        self.lock.release()




