import threading
from tkinter import *
import time
import random
import copy
import math

from grid import *
from utils import *

# GUI class
class GUIWindow():
    def __init__(self, grid, program_state, stop_event: threading.Event):
        self.width = grid.width
        self.height = grid.height
        self.update_cnt = 0

        self.program_state = program_state
        self.grid = grid
        self.running = threading.Event()
        self.updated = threading.Event()
        self.updated.clear()
        self.lock = threading.Lock()
        # grid info
        self.occupied = grid.occupied
        self.markers = grid.markers
        self.stop_event = stop_event
        self.empty = grid.empty

        self.robot = None


    """
    plot
    """
    def drawGrid(self):
        for y in range(1,self.grid.height):
            self.canvas.create_line(0, y * self.grid.scale, int(self.canvas.cget("width")) - 1, y * self.grid.scale)
        for x in range(1,self.grid.width):
            self.canvas.create_line(x * self.grid.scale, 0, x * self.grid.scale, int(self.canvas.cget("height")) - 1)
    def drawFreeCells(self):
        if self.robot:
            for block in self.robot.explored_cells:
                self.colorCell(block, '#FFFFFF')

            if (self.program_state == 'tasks'):
                for block in self.empty:
                    self.colorCell(block, '#FFFFFF')

    def drawOccupied(self):
        for block in self.occupied:
            if (self.robot and block in self.robot.explored_cells) or (self.program_state == 'tasks'):
                self.colorCell(block, '#222222')

    def drawMarkers(self):
        if not self.robot:
            return
        for marker in self.markers:
            if marker[0:2] not in self.robot.explored_cells and self.program_state == 'exploration':
                continue
            marker_x, marker_y, marker_h = self.grid.parse_marker_info(marker[0], marker[1], marker[2])

            if self.program_state == 'tasks' and marker not in self.robot.markers_found_or_picked:
                self.colorCell((marker[0],marker[1]), '#bebebe')        
          
            if marker[2] == 'U':
                marker_x += 0.5
                arrow_head_x, arrow_head_y = rotate_point(0.6, 0, marker_h)
                self.colorLine((marker_x, marker_y), (marker_x + arrow_head_x, marker_y + arrow_head_y), \
                linewidth=2, color='#222222')
            elif marker[2] == 'D':
                marker_x += 0.5
                marker_y += 1
                arrow_head_x, arrow_head_y = rotate_point(0.6, 0, marker_h)
                self.colorLine((marker_x, marker_y), (marker_x + arrow_head_x, marker_y + arrow_head_y), \
                linewidth=2, color='#222222')
            elif marker[2] == 'L':
                marker_x += 1
                marker_y += 0.5
                self.colorLine((marker_x-1, marker_y), (marker_x-0.4, marker_y), \
                linewidth=2, color='#222222')
            elif marker[2] == 'R':                
                marker_y += 0.5
                self.colorLine((marker_x+0.6, marker_y), (marker_x + 1, marker_y), \
                linewidth=2, color='#222222')

            c1x, c1y = rotate_point(0.4, -0.5, marker_h)
            c2x, c2y = rotate_point(0.6, 0.5, marker_h)
            self.colorRectangle((marker_x+c1x, marker_y+c1y), (marker_x+c2x, marker_y+c2y), '#008c8c')

    def drawCentroid(self):
        if not self.grid.centroid:
            return
        centroid_color = "#0091EA"
        self.colorCircle(self.grid.centroid, centroid_color)

    def weight_to_color(self, weight):
        return "#%02x00%02x" % (int(weight * 255), int((1 - weight) * 255))

    def _show_mean(self, x, y, heading_deg, confident=False):
        if confident:
            color = "#00AA00"
        else:
            color = "#CCCCCC"
        location = (x,y)
        self.colorTriangle(location, heading_deg, color,tri_size=20)

    def _show_robot(self, robot):
        
        coord = (robot.x, robot.y)
        self.colorTriangle(coord, robot.h, '#FF0000', tri_size=15)
        self.colorLine((robot.x, robot.y), (robot.next_coord[0], robot.next_coord[1]), color='#222222', linewidth=2, dashed=True)

    def show_image(self):
        global img
        img = PhotoImage(file="images/cs3630.gif")
        self.canvas.create_image(10,10, anchor=NW, image=img)

    def clean_world(self):
        self.canvas.delete("all")
        self.drawFreeCells()
        self.drawOccupied()
        self.drawMarkers()
        self.drawCentroid()

    """
    plot utils
    """

    # Draw a colored square at the specified grid coordinates
    def colorCell(self, location, color):
        coords = (location[0]*self.grid.scale, (location[1])*self.grid.scale)
        self.canvas.create_rectangle(coords[0], coords[1], coords[0] + self.grid.scale, coords[1] + self.grid.scale, width=1, fill=color)

    def colorRectangle(self, corner1, corner2, color):
        coords1 =  (corner1[0]*self.grid.scale, (corner1[1])*self.grid.scale)
        coords2 =  (corner2[0]*self.grid.scale, (corner2[1])*self.grid.scale)
        self.canvas.create_rectangle(coords1[0], coords1[1], coords2[0], coords2[1], width=0, fill=color)

    def colorCircle(self,location, color, dot_size = 5):
        x0, y0 = location[0]*self.grid.scale - dot_size, (location[1])*self.grid.scale - dot_size
        x1, y1 = location[0]*self.grid.scale + dot_size, (location[1])*self.grid.scale + dot_size
        return self.canvas.create_oval(x0, y0, x1, y1, fill=color)

    def colorLine(self, coord1, coord2, color='black', linewidth=1, dashed=False):
        if dashed:
            self.canvas.create_line(coord1[0] * self.grid.scale, (coord1[1])* self.grid.scale, \
                coord2[0] * self.grid.scale, (coord2[1]) * self.grid.scale,  \
                fill=color, width=linewidth, dash=(5,3))
        else:
            self.canvas.create_line(coord1[0] * self.grid.scale, (coord1[1])* self.grid.scale, \
                coord2[0] * self.grid.scale, (coord2[1]) * self.grid.scale,  \
                fill=color, width=linewidth)

    def colorTriangle(self, location, heading_deg, color, tri_size):
        hx, hy = rotate_point(tri_size, 0, heading_deg)
        lx, ly = rotate_point(-tri_size, tri_size, heading_deg)
        rx, ry = rotate_point(-tri_size, -tri_size, heading_deg)
        hrot = (hx + location[0]*self.grid.scale, -hy + (location[1])*self.grid.scale)
        lrot = (lx + location[0]*self.grid.scale, -ly + (location[1])*self.grid.scale)
        rrot = (rx + location[0]*self.grid.scale, -ry + (location[1])*self.grid.scale)
        return self.canvas.create_polygon(hrot[0], hrot[1], lrot[0], lrot[1], rrot[0], rrot[1], \
            fill=color, outline='#000000',width=1)

    """
    Sync data to plot from other thread
    """
    def show_robot(self, robot):
        self.lock.acquire()
        self.robot = copy.deepcopy(robot)
        self.lock.release()

    def setupdate(self):
        self.updateflag = True

    def update(self):
        self.lock.acquire()
        self.clean_world()
        if self.robot != None:
            self._show_robot(self.robot)

            time.sleep(0.001)
        self.updated.clear()
        self.lock.release()

    # start GUI thread
    def start(self):
        master = Tk()
        master.wm_title("Warehouse: Red - robot")

        bg_color = '#616161'
        self.canvas = Canvas(master, width = self.grid.width * self.grid.scale, height = self.grid.height * self.grid.scale, bd = 0, bg = bg_color)

        self.canvas.pack()

        self.drawFreeCells()
        self.drawOccupied()
        self.drawMarkers()

        # Start mainloop and indicate that it is running
        self.running.set()
        while not self.stop_event.is_set():
            if self.updated.is_set():
                self.update()
            try:
                master.update_idletasks()
                master.update()
            except TclError:
                break
        # Indicate that main loop has finished
        self.running.clear()