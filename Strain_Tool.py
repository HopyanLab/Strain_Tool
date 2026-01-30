#!/usr/bin/env python3

import sys
import time
import copy
import subprocess
import numpy as np
import mahotas as mh
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
#from matplotlib import pyplot as plt
from scipy import ndimage as ndi # ndi.fourier_shift
from scipy.signal import windows, correlate2d
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy import fftpack
from functools import partial
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qt5agg import (
					FigureCanvasQTAgg as FigureCanvas,
					NavigationToolbar2QT as NavigationToolbar
							)
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib import colors, ticker, cm
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from PyQt5.QtCore import (
					Qt, QPoint, QRect, QSize,
					QObject, QThread, pyqtSignal
							)
from PyQt5.QtGui import (
					QIntValidator, QDoubleValidator, QMouseEvent,
					QPalette, QColor
							)
from PyQt5.QtWidgets import (
					QApplication, QLabel, QWidget, QFrame,
					QPushButton, QHBoxLayout, QVBoxLayout,
					QComboBox, QCheckBox, QSlider, QProgressBar,
					QFormLayout, QLineEdit, QTabWidget,
					QSizePolicy, QFileDialog, QMessageBox
							)
from skimage import data
from skimage.registration import phase_cross_correlation
from pathlib import Path
from zipfile import ZipFile
from PIL import Image
from bioio import BioImage
#from aicsimageio import AICSImage
#from aicspylibczi import CziFile
#from nd2reader import ND2Reader
#from imaris_ims_file_reader.ims import ims

################################################################################
# colormaps for matplotlib #
############################

red_cdict = {
		'red':   ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		'green': ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		'blue':  ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		'alpha': ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
			}
red_cmap = LinearSegmentedColormap('red_cmap', red_cdict)
#cm.register_cmap(cmap=red_cmap)
try:
	matplotlib.colormaps.register(red_cmap)
except:
	pass

green_cdict = {
		'red':   ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		'green': ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		'blue':  ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		'alpha': ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
			}
green_cmap = LinearSegmentedColormap('green_cmap', green_cdict)
#cm.register_cmap(cmap=green_cmap)
try:
	matplotlib.colormaps.register(green_cmap)
except:
	pass

blue_cdict = {
		'red':   ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		'green': ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		'blue':  ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		'alpha': ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
			}
blue_cmap = LinearSegmentedColormap('blue_cmap', blue_cdict)
#cm.register_cmap(cmap=blue_cmap)
try:
	matplotlib.colormaps.register(blue_cmap)
except:
	pass

transparent_cdict = {
		'red':   ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		'green': ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		'blue':  ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		'alpha': ((0, 1.0, 1.0),
				  (1, 0.0, 0.0)),
			}
transparent_cmap = LinearSegmentedColormap('transparent_cmap',
											transparent_cdict)
#cm.register_cmap(cmap=transparent_cmap)
try:
	matplotlib.colormaps.register(transparent_cmap)
except:
	pass

################################################################################
# remove tree method for pathlib #
##################################

def rm_tree(pth):
	pth = Path(pth)
	for child in pth.glob('*'):
		if child.is_file():
			child.unlink()
		else:
			rm_tree(child)
	pth.rmdir()

################################################################################
# read image arrays from data files #
#####################################

#def get_image (image_stack,
#				t_value = 0, z_value = 0, channel = None):
#	image = None
#	if image_stack is not None:
#		if channel is None:
#			image = image_stack.get_image_data('CYX',
#										T = t_value,
#										Z = z_value)
#		else:
#			image = image_stack.get_image_data('YX',
#										C = channel,
#										T = t_value,
#										Z = z_value)
#	else:
#		display_error('No data file loaded!')
#	return image

def get_image (image_stack,
				t_value = 0, z_value = 0, channel = None):
	image = None
	if image_stack is not None:
		if channel is None:
			image = image_stack.get_image_data('YXC',
									#	C = channel,
										T = t_value,
										Z = z_value)
	#		image = np.sum(image, axis = -1)
		else:
			image = image_stack.get_image_data('YX',
										C = channel,
										T = t_value,
										Z = z_value)
	else:
		display_error('No data file loaded!')
	return image

################################################################################
# class to open single tif image. Not used. #
#############################################

class single_tif_stack:
	def __init__ (self, tif_file_path):
		self.t_values = np.array([0])
		self.z_values = np.array([0])
		self.tif_image = Image.open(tif_file_path)
		self.image_array = np.array(self.tif_image)
		self.shape = self.image_array.shape
		if len(self.shape) == 3 and self.shape[2]>2:
			self.channel_names = np.array(['red','green','blue'])
		else:
			self.channel_names = np.array(['grey'])
		self.shape = np.array([len(self.t_values),
							   len(self.channel_names),
							   len(self.z_values),
							   self.image_array.shape[0],
							   self.image_array.shape[1]])
		self.physical_pixel_sizes = [1,1,1]
	def get_image_data(self, string, C = 0, T = 0, Z = 0):
		if len(self.shape) == 2:
			return self.image_array
		elif string == 'YX':
			return self.image_array[:,:,C]
		else:
			return self.image_array

################################################################################
# class to open tif stack from imageJ #
#######################################

class zip_tif_stack:
	def __init__ (self, zip_file_path):
		self.zip_file = ZipFile(zip_file_path)
		self.tif_dir = zip_file_path.parent/'temp'
		if self.tif_dir.exists():
			rm_tree(self.tif_dir)
		self.zip_file.extractall(self.tif_dir)
		self.name_list = self.zip_file.namelist()
		self.t_list = np.array([(element.split('.')[-2]).split('_')[-2].lstrip(
							'T') for element in self.name_list]).astype(int)
		self.z_list = np.array([(element.split('.')[-2]).split('_')[-1].lstrip(
							'Z') for element in self.name_list]).astype(int)
		self.t_values = np.unique(self.t_list)
		self.z_values = np.unique(self.z_list)
		image = Image.open(self.tif_dir/self.name_list[0])
		image_array = np.array(image)
		if len(image_array.shape) == 2:
			image_array = image_array[:,:,np.newaxis]
		test = np.sum(image_array, axis=(0,1))
		self.mask = (test != 0)
		if len(test) == 3:
			self.channel_names = np.array(['red','green','blue'])
		elif len(test) == 1:
			self.channel_names = np.array(['grey'])
		self.shape = np.array([len(self.t_values),
							   len(self.channel_names),
							   len(self.z_values),
							   image_array.shape[0],
							   image_array.shape[1]])
	#	print(self.shape)
	#	print(self.channel_names)
		self.physical_pixel_sizes = [1,1,1]
	def get_image_data(self, string, C = 0, T = 0, Z = 0):
		file_index = np.argmax(np.logical_and((self.t_list == T+1),
											  (self.z_list == Z+1)))
	#	print((self.tif_dir/self.name_list[file_index]))
		image_array = np.array(
						Image.open(self.tif_dir/self.name_list[file_index]))
		if len(image_array.shape) == 2:
			image_array = image_array[:,:,np.newaxis]
		if string == 'YX':
			return image_array[:,:,C]
		else:
			return image_array

################################################################################
# function to compute strains for triangle element #
####################################################

def compute_strain (tri_0, tri_1):
	if np.cross(tri_0[1]-tri_0[0], tri_0[2]-tri_0[1]) < 0:
		tri_0 = tri_0[::-1]
		tri_1 = tri_1[::-1]
	delta = tri_1 - tri_0
	x_13 = tri_0[0,0]-tri_0[2,0]
	x_23 = tri_0[1,0]-tri_0[2,0]
	y_13 = tri_0[0,1]-tri_0[2,1]
	y_23 = tri_0[1,1]-tri_0[2,1]
	dx_13 = delta[0,0]-delta[2,0]
	dx_23 = delta[1,0]-delta[2,0]
	dy_13 = delta[0,1]-delta[2,1]
	dy_23 = delta[1,1]-delta[2,1]
	det_j = x_13 * y_23 - y_13 * x_23
	eps_xx = (y_23 * dx_13 - y_13 * dx_23) / det_j
	eps_xy = (x_13 * dx_23 - x_23 * dx_13 +
				y_23 * dy_13 - y_13 * dy_23) / det_j / 2
	eps_yy = (x_13 * dy_23 - x_23 * dy_13) / det_j
	return eps_xx, eps_yy, eps_xy

################################################################################
# digital image correlation functions #
#######################################

def prepare_image (image):
	intensity_mean = np.mean(image)
	image = image - intensity_mean
	#pixel_count = image.shape[0] * image.shape[1]
	#image = image * pixel_count
	intensity_norm = np.linalg.norm(image)
	image = image / intensity_norm
	return image

def get_shift (image_1, image_2, tracking_method = 'ski',
			   search_distance = 12, window = None):
	if tracking_method == 'ski':
		return get_shift_ski(image_1, image_2)
	elif tracking_method == 'fft':
		return get_shift_fft(image_1, image_2, window)
	elif tracking_method == 'con':
		return get_shift_con(image_1, image_2, search_distance)
	elif tracking_method == 'bpf':
		return get_shift_bpf(image_1, image_2, search_distance)
	else:
		return np.array([0,0])

def get_shift_ski (image_1, image_2, window = None):
	shift, error, phase = phase_cross_correlation(image_1, image_2,
												  upsample_factor=100,
												  normalization=None)
	return -shift[::-1]

def get_shift_fft (image_1, image_2, window = None):
	image_1 = prepare_image(image_1.astype(float))
	image_2 = prepare_image(image_2.astype(float))
	if window is not None:
		image_1 = image_1 * window
		image_2 = image_2 * window
	I1 = np.fft.fft2(image_1)
	I2 = np.fft.fft2(image_2)
	#I1 = np.fft.fftshift(I1)
	#I2 = np.fft.fftshift(I2)
	R = I1*np.conjugate(I2)
	#R = np.divide(R, np.absolute(I1)*np.absolute(I2))
	#R = np.fft.ifftshift(R)
	r = np.fft.ifft2(R)
	r = np.fft.fftshift(r)
	max_index = np.argmax(np.absolute(r))
	shift = np.floor(np.array(r.shape)/2).astype(int) - \
			np.array(np.unravel_index(max_index, r.shape))
	return shift[::-1]

def get_shift_con(image_1, image_2, search_distance = 12):
	correlation = np.zeros((2*search_distance, 2*search_distance),
						dtype = float)
	for y_shift in range(-search_distance,search_distance):
		for x_shift in range(-search_distance,search_distance):
			I_1 = image_1[search_distance:-search_distance,
						  search_distance:-search_distance]
			I_2 = image_2[
					search_distance+y_shift:-search_distance+y_shift,
					search_distance+x_shift:-search_distance+x_shift]
			I_1 = prepare_image(I_1)
			I_2 = prepare_image(I_2)
			correlation[search_distance+y_shift,
						search_distance+x_shift] = np.sum(I_1 * I_2)
	max_index = np.argmax(correlation)
	shift = [search_distance,search_distance] - \
				np.array(np.unravel_index(max_index, correlation.shape))
	return shift[::-1]

def get_shift_bpf(image_1, image_2, search_distance = 12):
	pass

################################################################################
# find bright points in image array #
#####################################

def find_centres (frame, neighbourhood_size,
				  threshold_difference, gauss_deviation, channel = 0):
	x_size, y_size = frame.shape[0:2]
	if len(frame.shape) == 3:
		frame = frame[:,:,channel]
	frame = ndi.gaussian_filter(frame, gauss_deviation)
	frame_max = ndi.maximum_filter(frame, neighbourhood_size)
	maxima = (frame == frame_max)
	frame_min = ndi.minimum_filter(frame, neighbourhood_size)
	differences = ((frame_max - frame_min) > threshold_difference)
	maxima[differences == 0] = 0
	maximum = np.amax(frame)
	minimum = np.amin(frame)
	outside_filter = (frame_max > (maximum-minimum)*0.1 + minimum)
	maxima[outside_filter == 0] = 0
	labeled, num_objects = ndi.label(maxima)
	slices = ndi.find_objects(labeled)
	centres = np.zeros((len(slices),2), dtype = int)
	good_centres = 0
	for (dy,dx) in slices:
		centres[good_centres,0] = int((dx.start + dx.stop - 1)/2)
		centres[good_centres,1] = int((dy.start + dy.stop - 1)/2)
		if centres[good_centres,0] < neighbourhood_size/2 or \
		   centres[good_centres,0] > y_size - neighbourhood_size/2 or \
		   centres[good_centres,1] < neighbourhood_size/2 or \
		   centres[good_centres,1] > x_size - neighbourhood_size/2:
			good_centres -= 1
		good_centres += 1
	centres = centres[:good_centres]
	# TODO: more elegant way to do this!
	to_remove = np.zeros(centres.shape[0])
	for i, centre_i in enumerate(centres):
		if to_remove[i]:
			continue
		for j, centre_j in enumerate(centres):
			if i == j:
				continue
			if to_remove[j]:
				continue
			if np.linalg.norm(centre_i-centre_j) < neighbourhood_size/2:
				centre_i = (centre_i + centre_j)/2
				to_remove[j] = 1
	centres = centres[to_remove == 0]
	#
	return centres

################################################################################
# function to triangulate point cloud #
#######################################

def triangulate (points, max_length = None):
	triangles = Delaunay(points).simplices
	num_triangles = len(triangles)
	edges_1 = triangles[:,0:2]
	edges_2 = triangles[:,1:]
	edges_3 = triangles[:,0::2]
	edges = np.vstack([edges_1, edges_2, edges_3])
	lengths = np.linalg.norm(points[edges][:,0,:] -
							 points[edges][:,1,:], axis=-1)
	mean_length = np.mean(lengths)
	if max_length is not None:
		if max_length > 0:
			edge_mask = (lengths < max_length)
		else:
			edge_mask = (lengths > 0)
	else:
		edge_mask = (lengths > 0)
	triangle_mask = np.logical_and(np.logical_and(edge_mask[:num_triangles],
								edge_mask[num_triangles:2*num_triangles]),
								edge_mask[2*num_triangles:3*num_triangles])
	edges = np.sort(edges[edge_mask], axis=1)
	edges = np.unique(edges, axis=0)
	triangles = triangles[triangle_mask]
	return edges, triangles

################################################################################
# helper functions for GUI elements #
#####################################

def display_error (error_text = 'Something went wrong!'):
	msg = QMessageBox()
	msg.setIcon(QMessageBox.Critical)
	msg.setText("Error")
	msg.setInformativeText(error_text)
	msg.setWindowTitle("Error")
	msg.exec_()

def setup_textbox (function, layout, label_text,
				   initial_value = 0, is_int = True):
	textbox = QLineEdit()
	need_inner = not isinstance(layout, QHBoxLayout)
	if need_inner:
		inner_layout = QHBoxLayout()
	label = QLabel(label_text)
	label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
	if need_inner:
		inner_layout.addWidget(label)
	else:
		layout.addWidget(label)
	textbox.setMaxLength(6)
	textbox.setFixedWidth(100)
	textbox.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
	if is_int:
		textbox.setValidator(QIntValidator())
	else:
		textbox.setValidator(QDoubleValidator())
	textbox.setText(str(initial_value))
	textbox.editingFinished.connect(function)
	if need_inner:
		inner_layout.addWidget(textbox)
		layout.addLayout(inner_layout)
	else:
		layout.addWidget(textbox)
	return textbox

def get_textbox (textbox,
				 minimum_value = None,
				 maximum_value = None,
				 is_int = False):
	if is_int:
		value = int(np.floor(float(textbox.text())))
	else:
		value = float(textbox.text())
	if maximum_value is not None:
		if value > maximum_value:
			value = maximum_value
	if minimum_value is not None:
		if value < minimum_value:
			value = minimum_value
	textbox.setText(str(value))
	return value

def setup_button (function, layout, label_text, toggle = False):
	button = QPushButton()
	if toggle:
		button.setCheckable(True)
	button.setText(label_text)
	button.clicked.connect(function)
	layout.addWidget(button)
	return button

def setup_checkbox (function, layout, label_text,
					is_checked = False):
		checkbox = QCheckBox()
		checkbox.setText(label_text)
		checkbox.setChecked(is_checked)
		checkbox.stateChanged.connect(function)
		layout.addWidget(checkbox)
		return checkbox

def setup_tab (tabs, tab_layout, label):
	tab = QWidget()
	tab.layout = QVBoxLayout()
	tab.setLayout(tab.layout)
	tab.layout.addLayout(tab_layout)
	tabs.addTab(tab, label)

def horizontal_separator (layout, palette):
	separator = QFrame()
	separator.setFrameShape(QFrame.HLine)
	#separator.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Expanding)
	separator.setLineWidth(1)
	palette.setColor(QPalette.WindowText, QColor('lightgrey'))
	separator.setPalette(palette)
	layout.addWidget(separator)

def setup_progress_bar (layout):
	progress_bar = QProgressBar()
	clear_progress_bar(progress_bar)
	layout.addWidget(progress_bar)
	return progress_bar

def clear_progress_bar (progress_bar):
	progress_bar.setMinimum(0)
	progress_bar.setFormat('')
	progress_bar.setMaximum(1)
	progress_bar.setValue(0)

def update_progress_bar (progress_bar, value = None,
						 minimum_value = None,
						 maximum_value = None,
						 text = None):
	if minimum_value is not None:
		progress_bar.setMinimum(minimum_value)
	if maximum_value is not None:
		progress_bar.setMaximum(maximum_value)
	if value is not None:
		progress_bar.setValue(value)
	if text is not None:
		progress_bar.setFormat(text)

def setup_slider (layout, function, maximum_value = 1,
				  direction = Qt.Horizontal):
		slider = QSlider(direction)
		slider.setMinimum(0)
		slider.setMaximum(maximum_value)
		slider.setSingleStep(1)
		slider.setValue(0)
		slider.valueChanged.connect(function)
		return slider

def update_slider (slider, value = None,
				   maximum_value = None):
	if value is not None:
		slider.setValue(value)
	if maximum_value is not None:
		slider.setMaximum(maximum_value)

def setup_combobox (function, layout, label_text):
	combobox = QComboBox()
	need_inner = not isinstance(layout, QHBoxLayout)
	if need_inner:
		inner_layout = QHBoxLayout()
	label = QLabel(label_text)
	label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
	if need_inner:
		inner_layout.addWidget(label)
	else:
		layout.addWidget(label)
	combobox.currentIndexChanged.connect(function)
	if need_inner:
		inner_layout.addWidget(combobox)
		layout.addLayout(inner_layout)
	else:
		layout.addWidget(combobox)
	return combobox

def clear_layout (layout):
	for i in reversed(range(layout.count())): 
		widgetToRemove = layout.takeAt(i).widget()
		layout.removeWidget(widgetToRemove)
		widgetToRemove.deleteLater()

################################################################################
# matplotlib canvas widget #
############################

class MPLCanvas(FigureCanvas):
	def __init__ (self, parent=None, width=8, height=8, dpi=100):
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.ax = self.fig.add_subplot(111)
		FigureCanvas.__init__(self, self.fig)
		self.setParent(parent)
		FigureCanvas.setSizePolicy(self,
				QSizePolicy.Expanding,
				QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		self.fig.tight_layout()
		# stuff to plot
		self.image_array = np.array([[0]], dtype = int)
		self.track_points = np.zeros((0,2), dtype = int)
		self.bad_points = np.zeros((0), dtype = int)
		self.points_adjust = np.zeros((0,2), dtype = int)
		self.drift_adjust = np.array([0,0], dtype = int)
		self.drift_changed = False
		self.focus_box = None # np.array([[x_0, x_1], [y_0, y_1]])
		self.edges = np.zeros((0,2), dtype = int)
		self.triangles = np.zeros((0,3), dtype = int)
		self.strains = np.zeros((0), dtype = float)
		self.strain_max = 0
		self.selected_point = None
		# plot objects
		self.image_plot = None
		self.box_plot = None
		self.points_plot = None
		self.edge_plot = None
		self.strain_plot = None
		self.bad_points_plot = None
		self.select_box = None
		self.selected_point_plot = None
		# boolean flags
		self.show_box = False
		self.show_points = False
		self.show_edges = True
		self.show_strain = False
		# colour choices
		self.colormap = 'Greys_r' #'afmhot'
		self.marker_color = 'white' # 'blue'
		self.marker_alpha = 0.8
		#
		self.zoomed = False
		self.flip_vertical = False
	
	def clear_canvas (self):
		# stuff to plot
		self.image_array = np.array([[0]], dtype = int)
		self.track_points = np.zeros((0,2), dtype = int)
		self.bad_points = np.zeros((0), dtype = int)
		self.points_adjust = np.zeros((0,2), dtype = int)
		self.drift_adjust = np.array([0,0], dtype = int)
		self.drift_changed = False
		self.focus_box = None # np.array([[x_0, x_1], [y_0, y_1]])
		self.edges = np.zeros((0,2), dtype = int)
		self.triangles = np.zeros((0,3), dtype = int)
		self.strains = np.zeros((0), dtype = float)
		self.strain_min = 0
		self.strain_max = 0
		self.selected_point = None
		# plot objects
		self.remove_plot_element(self.image_plot)
		self.remove_plot_element(self.box_plot)
		self.remove_plot_element(self.points_plot)
		self.remove_plot_element(self.edge_plot)
		self.remove_plot_element(self.strain_plot)
		self.remove_plot_element(self.bad_points_plot)
		self.remove_plot_element(self.selected_point_plot)
		self.select_box = None
		# boolean flags
		self.show_box = False
		self.show_points = False
		self.show_edges = True
		self.show_strain = False
		self.transparent_strain = True
		self.draw()
	
	def set_flip (self, flip_vertical = False):
		self.flip_vertical = flip_vertical
		self.set_bounds()
		self.draw()
	
	def set_zoom (self, zoomed = False):
		self.zoomed = zoomed
		self.set_bounds()
		self.draw()
	
	def set_stain_overlay (self, show_strain = False):
		self.show_strain = show_strain
		self.plot_strain()
	
	def set_bounds (self):
		if self.zoomed and self.focus_box is not None:
			self.ax.set_xlim(
						left = self.focus_box[0,0] + self.drift_adjust[0],
						right = self.focus_box[0,1] + self.drift_adjust[0] )
			self.ax.set_ylim(
						bottom = self.focus_box[1,0] + self.drift_adjust[1],
						top = self.focus_box[1,1] + self.drift_adjust[1] )
		else:
			self.ax.set_xlim(left = 0, right = self.image_array.shape[1])
			self.ax.set_ylim(bottom = 0, top = self.image_array.shape[0])
		if self.flip_vertical:
			self.ax.invert_yaxis()
	
	def update_image (self, image_array = np.array([[0]], dtype = int),
							drift_adjust = np.array([0,0], dtype = int)):
		drift_adjust = np.around(drift_adjust).astype(int)
		self.image_array = image_array
		if np.all(self.drift_adjust == drift_adjust):
			self.drift_changed = False
		else:
			self.drift_changed = True
			self.drift_adjust = drift_adjust
		self.plot_image()
	
	def update_points (self, track_points = np.zeros((0,2), dtype = int),
							 points_adjust = np.zeros((0,2), dtype = int),
							 bad_points = np.zeros((0), dtype = int),
							 selected_point = None):
		points_adjust = np.around(points_adjust).astype(int)
		self.track_points = track_points
		self.points_adjust = points_adjust
		self.bad_points = bad_points
		self.selected_point = selected_point
		self.plot_points()
	
	def update_edges (self, edges = np.zeros((0,2), dtype = int)):
		self.edges = edges
		self.plot_edges()
	
	def update_triangles (self, triangles = np.zeros((0,3), dtype = int)):
		self.triangles = triangles
	
	def update_strains (self, strains = np.zeros(0, dtype = float),
							strain_max=0):
		if strains is not None:
			self.strains = strains
			self.strain_max = strain_max
		self.plot_strain()
	
	def update_focus_box (self, focus_box):
		self.focus_box = focus_box
		self.plot_box()
	
	def update_colors (self, colormap = 'Greys_r', #'afmhot'
							 marker_color = 'white', # 'blue'
							 marker_alpha = 0.8):
		self.colormap = colormap
		self.marker_color = marker_color
		self.marker_alpha = marker_alpha
	
	def plot_image (self):
		self.remove_plot_element(self.image_plot)
	#	self.ax.set_xlim(left = 0, right = self.image_array.shape[1])
	#	self.ax.set_ylim(bottom = 0, top = self.image_array.shape[0])
		if self.image_array is None:
			return
		if len(self.image_array.shape) == 3 and self.image_array.shape[-1] > 1:
			self.image_plot = self.ax.imshow(self.image_array,
											 zorder = 1)
		else:
			self.image_plot = self.ax.imshow(self.image_array,
											 cmap = self.colormap,
											 zorder = 1)
		if self.drift_changed:
			self.plot_box()
			self.plot_points()
		self.set_bounds()
		self.draw()
	
	def plot_points (self):
		self.remove_plot_element(self.points_plot)
		self.remove_plot_element(self.bad_points_plot)
		self.remove_plot_element(self.selected_point_plot)
		if self.track_points is not None:
			if len(self.track_points) > 0:
				self.show_points = True
			else:
				self.show_points = False
		else:
			self.show_points = False
		if self.show_points:
			track_points = self.track_points + [self.drift_adjust]
			if len(self.points_adjust) == len(self.track_points):
				track_points += self.points_adjust
			self.points_plot = self.ax.plot(track_points[:,0],
											track_points[:,1],
											color = self.marker_color,
											linestyle = '',
											marker = 'x',
											markersize = 4.,
											alpha = self.marker_alpha,
											zorder = 5)
			if len(self.bad_points) > 0:
				self.bad_points_plot = self.ax.plot(track_points[
														self.bad_points,0],
													track_points[
														self.bad_points,1],
													color = 'fuchsia',
													linestyle = '',
													marker = 'o',
													markersize = 3.,
													alpha = 0.5,
													zorder = 6)
			if self.selected_point is not None:
				if self.selected_point < len(track_points):
					self.selected_point_plot = self.ax.plot(
													track_points[
														self.selected_point,0],
													track_points[
														self.selected_point,1],
													color = 'tab:green',
													linestyle = '',
													marker = 'o',
													markersize = 4.,
													alpha = 0.5,
													zorder = 4)
		else:
			self.points_plot = None
			self.bad_points_plot = None
		self.draw()
	
	def plot_edges (self):
		self.remove_plot_element(self.edge_plot)
		if self.show_edges and len(self.edges) > 0:
			track_points = self.track_points + [self.drift_adjust]
			if len(self.points_adjust) == len(self.track_points):
				track_points += self.points_adjust
			self.edge_plot = self.ax.plot(track_points[self.edges.T,0],
										  track_points[self.edges.T,1],
										  color = 'gray',
										  linestyle = '-',
										  linewidth = 1,
										  alpha = 0.8,
										  zorder = 4)
		self.draw()
	
	def plot_strain (self):
		self.remove_plot_element(self.strain_plot)
		if self.show_strain and len(self.strains) > 0 and \
								len(self.triangles) > 0 and \
								len(self.strains) == len(self.track_points):
			points = self.track_points + [self.drift_adjust]
			if len(self.points_adjust) == len(self.track_points):
				points += self.points_adjust
			colormap = plt.get_cmap('coolwarm')
			if self.transparent_strain:
				colormap_adj = colormap(np.arange(colormap.N))
				colormap_adj[:,-1] = \
					(2*np.abs(np.arange(colormap.N)-colormap.N/2)) /\
									(colormap.N)
				colormap_adj += 0.2
				colormap_adj /= np.amax(colormap_adj)
				colormap = ListedColormap(colormap_adj)
			norm = plt.Normalize(self.strain_min, self.strain_max)
			colors = colormap(norm(self.strains))
			self.strain_plot = []
			for tri_index, triangle in enumerate(self.triangles):
				try:
					grid_x, grid_y = np.mgrid[
						points[triangle][:,0].min():\
								points[triangle][:,0].max():20j,
						points[triangle][:,1].min():\
								points[triangle][:,1].max():20j]
					grid_z = griddata(points[triangle], self.strains[triangle],
									(grid_x, grid_y), method='linear')
					tri_plot = self.ax.pcolormesh(
								grid_x, grid_y, grid_z,
								cmap = colormap,
								vmin = -self.strain_max,
								vmax = self.strain_max,
								shading = 'auto',
								zorder = 3)
					self.strain_plot.append(tri_plot)
				except:
					pass
		self.draw()
	
	def plot_box (self):
		self.remove_plot_element(self.box_plot)
		focus_box = self.focus_box.copy()
		if np.any(self.drift_adjust != 0):
			focus_box[0] += self.drift_adjust[0]
			focus_box[1] += self.drift_adjust[1]
		if self.show_box:
			self.box_plot = self.ax.plot((focus_box[0,0],
										  focus_box[0,1],
										  focus_box[0,1],
										  focus_box[0,0],
										  focus_box[0,0]),
										 (focus_box[1,0],
										  focus_box[1,0],
										  focus_box[1,1],
										  focus_box[1,1],
										  focus_box[1,0]),
										 color='gold',
										 linestyle='-',
										 linewidth = 1,
										 alpha = 0.8,
										 zorder = 6)
		else:
			self.box_plot = None
		self.draw()
	
	def plot_selector (self, p_1, p_2):
		self.remove_selector()
		self.select_box = self.ax.plot((p_1[0], p_2[0], p_2[0], p_1[0],
										p_1[0]),
									   (p_1[1], p_1[1], p_2[1], p_2[1],
										p_1[1]),
									color = 'white',
									linestyle = '-',
									linewidth = 1,
									zorder = 7)
		self.draw()
	
	def remove_selector (self):
		if self.select_box is not None:
			if isinstance(self.select_box,list):
				for line in self.select_box:
					line.remove()
			else:
				self.select_box.remove()
			self.select_box = None
		self.draw()
	
	def remove_plot_element (self, plot_element):
		if plot_element is not None:
			if isinstance(plot_element,list):
				for line in plot_element:
					try:
						line.remove()
					except:
						pass
			else:
				try:
					plot_element.remove()
				except:
					pass

################################################################################
# sune function for fits #
##########################

def sine (x, A, omega, phi, B):
	return A*np.sin(omega*x + phi) + B

################################################################################
# data structure for fit results
################################################################################

class FitResults ():
	def __init__ (self,
					time_points = None, data_points = None,
					startpoint = None, endpoint = None,
					fit_function = None, best_params = None):
		self.time_points = time_points
		self.data_points = data_points
		self.startpoint = startpoint
		self.endpoint = endpoint
		self.fit_function = fit_function
		self.best_params = best_params

################################################################################
# mpl canvas for simple plots #
###############################

class MPLPlot(FigureCanvas):
	def __init__ (self, parent=None, width=10, height=4, dpi=100):
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.ax = self.fig.add_subplot(111)
		self.ax.set_xlim([0,1])
		self.ax.set_ylim([-1,1])
		FigureCanvas.__init__(self, self.fig)
		self.setParent(parent)
		FigureCanvas.setSizePolicy(self,
				QSizePolicy.Expanding,
				QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		self.fig.tight_layout()
		self.fig.set_tight_layout(True)
		# stuff to plot
		self.results = None
		# plot objects
		self.point_plot = None
		self.used_plot = None
		self.line_plot = None
		self.dotted_plot = None
		self.legend = None
	
	def update_plot (self, results = None):
		if results is not None:
			self.results = results
		self.plot()
	
	def clear_canvas (self):
		# stuff to plot
	#	self.points = np.array([[]], dtype = float)
	#	self.chosen = np.array([], dtype = bool)
	#	self.params = np.array([], dtype = float)
		# plot objects
		self.remove_plot_element(self.point_plot)
		self.point_plot = None
		self.remove_plot_element(self.used_plot)
		self.used_plot = None
		self.remove_plot_element(self.line_plot)
		self.line_plot = None
		self.remove_plot_element(self.dotted_plot)
		self.dotted_plot = None
		self.remove_plot_element(self.legend)
		self.legend = None
		self.ax.cla()
		#
		self.draw()
	
	def plot (self):
		self.clear_canvas()
		time_points = self.results.time_points
		data_points = self.results.data_points
		startpoint = self.results.startpoint
		endpoint = self.results.endpoint
		fit_function = self.results.fit_function
		best_params = self.results.best_params
		if (time_points is not None) and (data_points is not None):
			self.point_plot = self.ax.plot(
					time_points,
					data_points,
					marker = '.',
					linestyle = 'none',
					color = 'tab:orange',
					zorder = 5,
					label = 'Data')
			if (startpoint is not None) and (endpoint is not None):
				self.used_plot = self.ax.plot(
						time_points[startpoint:endpoint],
						data_points[startpoint:endpoint],
						marker = '.',
						linestyle = 'none',
						color = 'tab:blue',
						zorder = 6,
						label = 'Used')
			if (fit_function is not None) and (best_params is not None):
				fit_time_points = np.linspace(np.amin(time_points),
												np.amax(time_points), 2500)
				fit_data_points = fit_function(fit_time_points, *best_params)
				if (startpoint is not None) and (endpoint is not None):
					fit_time_points = fit_time_points
					fit_data_points = fit_data_points
				self.line_plot = self.ax.plot(
						fit_time_points, fit_data_points,
						linestyle = 'solid',
						color = 'tab:red',
						label = 'Fit')
			self.ax.set_ylim([np.amin(data_points), np.amax(data_points)])
			self.ax.set_xlim([np.amin(time_points), np.amax(time_points)])
		#	self.ax.legend()
			self.draw()
	
	def remove_plot_element (self, plot_element):
		if plot_element is not None:
			if isinstance(plot_element,list):
				for line in plot_element:
					try:
						line.remove()
					except:
						pass
			else:
				try:
					plot_element.remove()
				except:
					pass

################################################################################
# worker thread handling object to run long tasks #
###################################################

class Worker (QObject):
	finished = pyqtSignal()
	progress = pyqtSignal(int)

################################################################################
# main window object #
######################

class Window(QWidget):
	def __init__ (self):
		super().__init__()
		self.title = "Displacement Tracking Tool"
		self.canvas = MPLCanvas()
		self.toolbar = NavigationToolbar(self.canvas, self)
		self.plot_canvas = MPLPlot()
		self.plot_toolbar = NavigationToolbar(self.plot_canvas, self)
		self.select_mode = 'None' # 'Select' 'Add' 'Delete' 'Move'
		self.selecting_area = False
		self.click_id = 0
		self.move_id = 0
		self.position = np.array([0,0])
		#
		self.file_path = None
		self.image_stack = None
		#
		self.image_array = np.array([[0]])
		#
		self.x_size = 1
		self.x_lower = 0
		self.x_upper = self.x_size-1
		self.y_size = 1
		self.y_lower = 0
		self.y_upper = self.y_size-1
		self.z_size = 1
		self.z_lower = 0
		self.z_upper = self.z_size-1
		self.t_size = 1
		self.t_lower = 0
		self.t_upper = self.t_size-1
		self.c_size = 0
		self.z_position = 0
		self.t_position = 0
		self.channel = None
		self.channel_names = None
		self.zoomed = False
		self.strain_direction = None
		self.show_strain = False
		self.fit_range_min = 0
		self.fit_range_max = self.t_size-1
		self.fit_frequency = 5.
		self.elast_direction = 0 #TODO: some option to switch to y-direction
		self.elast_frequency_thresh = 0.01
		self.elast_amplitude_thresh = 0.02
		self.elast_fit_thresh = 1.0
		self.elast_gaussian_radius = 60.
		self.fit_results_average = None
		self.fit_results_points = None
		self.phase_array = None
		self.derivatives = None
		#
		self.grid_defaults = np.array([8,8])
		self.grid_number_x = self.grid_defaults[0]
		self.grid_number_y = self.grid_defaults[1]
		self.centres_defaults = np.array([16,40,2])
		self.neighbourhood_size = self.centres_defaults[0]
		self.threshold_difference = self.centres_defaults[1]
		self.gauss_deviation = self.centres_defaults[2]
		self.search_defaults = np.array([2,2,12])
		self.coarse_gaussian = self.search_defaults[0]
		self.fine_gaussian = self.search_defaults[1]
		self.search_distance = self.search_defaults[2]
		self.tukey_alpha = 0.5 # None uses Hann window
		self.max_length = 0
		#
		self.track_points = np.zeros((0,2), dtype = int)
		self.bad_points = np.array([], dtype = int)
		self.selected_point = None
		#
		self.coarse_search_done = False
		self.coarse_results = None
		self.fine_search_done = False
		self.fine_results = None
		self.strains_done = False
		#
		self.edges = np.zeros((0,2), dtype = int)
		self.triangles = np.zeros((0,2), dtype = int)
		self.strains = np.zeros((0,0,3), dtype = float)
		#
		self.process_running = False
		self.coarse_tracking = False
		self.fine_tracking = False
		self.process = None
		#
		self.click_id = self.canvas.mpl_connect(
							'button_press_event', self.on_click)
		#
		self.setupGUI()
	
	def reset_defaults (self):
		self.grid_number_x = self.grid_defaults[0]
		self.grid_number_y = self.grid_defaults[1]
		self.neighbourhood_size = self.centres_defaults[0]
		self.threshold_difference = self.centres_defaults[1]
		self.gauss_deviation = self.centres_defaults[2]
		self.coarse_gaussian = self.search_defaults[0]
		self.fine_gaussian = self.search_defaults[1]
		self.search_distance = self.search_defaults[2]
	
	def clear_analysis (self):
		self.track_points = np.zeros((0,2), dtype = int)
		self.bad_points = np.array([], dtype = int)
		self.coarse_search_done = False
		self.coarse_results = None
		self.fine_search_done = False
		self.fine_results = None
		self.strains_done = False
		self.edges = np.zeros((0,2), dtype = int)
		self.triangles = np.zeros((0,2), dtype = int)
		self.strains = np.zeros((0,0,4), dtype = float)
		#TODO reset canvas variables
	
	def setupGUI (self):
		self.setWindowTitle(self.title)
		# layout for full window
		outer_layout = QVBoxLayout()
		# top section for plot and options
		main_layout = QHBoxLayout()
		# main left for plot
		plot_layout = QVBoxLayout()
		plot_layout.addWidget(self.canvas)
		# time boxes and sliders
		plot_layout.addLayout(self.setup_time_layout())
		# plot toolbar and z boxes
		plot_layout.addLayout(self.setup_toolbar_layout())
		# time plot and toolbar
		plot_layout.addWidget(self.plot_canvas)
		plot_layout.addLayout(self.setup_plotbar_layout())
		main_layout.addLayout(plot_layout)
		# main right for options
		options_layout = QHBoxLayout()
		z_select_layout = QVBoxLayout()
		self.slider_z = setup_slider(z_select_layout, self.z_slider_select,
									maximum_value = self.z_size-1,
									direction = Qt.Vertical)
		z_select_layout.addWidget(self.slider_z)
		options_layout.addLayout(z_select_layout)
		# options tabs
		tabs = QTabWidget()
		tabs.setMinimumWidth(420)
		tabs.setMaximumWidth(420)
		setup_tab(tabs, self.setup_focus_layout(), 'focus')
		setup_tab(tabs, self.setup_points_layout(), 'points')
		setup_tab(tabs, self.setup_strain_layout(), 'strain')
	#	setup_tab(tabs, self.setup_colours_layout(), 'colours')
		options_layout.addWidget(tabs)
		#
		main_layout.addLayout(options_layout)
		outer_layout.addLayout(main_layout)
		# horizontal row of buttons
		outer_layout.addLayout(self.setup_bottom_layout())
		# instructions box
		outer_layout.addWidget(self.setup_instruction_box())
		# Set the window's main layout
		self.setLayout(outer_layout)
	
	def setup_instruction_box (self):
		self.instruction_box = QFrame()
		layout = QHBoxLayout()
		self.instruction_box.setFrameShape(QFrame.StyledPanel)
	#	self.instruction_box.setSizePolicy(QSizePolicy.Expanding)
		label = QLabel('<font color="red">Instructions: </font>')
		label.setAlignment(Qt.AlignLeft)
		self.instruction_text = QLabel('"Open File" to begin.')
		self.instruction_text.setAlignment(Qt.AlignLeft)
	#	self.instruction_text.setWordWrap(True)
		layout.addWidget(label)
		layout.addWidget(self.instruction_text)
		layout.addStretch()
		self.instruction_box.setLayout(layout)
		return self.instruction_box
	
	def setup_time_layout (self):
		time_layout = QHBoxLayout()
		self.slider_t = setup_slider(time_layout, self.t_slider_select,
									 maximum_value = self.t_size-1,
									 direction = Qt.Horizontal)
		time_layout.addWidget(self.slider_t)
		self.textbox_t = setup_textbox(
							self.t_textbox_select,
							time_layout, 'T:',
							self.t_position,
							is_int = True)
		self.button_t_min = setup_button(
							self.t_min_button,
							time_layout, 'Set T Min')
		self.button_t_max = setup_button(
							self.t_max_button,
							time_layout, 'Set T Max')
		return time_layout
	
	def setup_toolbar_layout (self):
		toolbar_layout = QHBoxLayout()
		toolbar_layout.addWidget(self.toolbar)
		self.textbox_z = setup_textbox(
							self.z_textbox_select,
							toolbar_layout, 'Z:',
							self.z_position,
							is_int = True)
		self.button_z_min = setup_button(
							self.z_min_button,
							toolbar_layout, 'Set Z Min')
		self.button_z_max = setup_button(
							self.z_max_button,
							toolbar_layout, 'Set Z Max')
		return(toolbar_layout)
	
	def setup_plotbar_layout (self):
		toolbar_layout = QHBoxLayout()
		toolbar_layout.addWidget(self.plot_toolbar)
		self.button_plot_average = setup_button(
							self.plot_average,
							toolbar_layout, 'Plot Avg')
		self.button_guess_range = setup_button(
							self.guess_range,
							toolbar_layout, 'Guess Range')
		self.textbox_range_min = setup_textbox(
							self.range_textbox_select,
							toolbar_layout, 'Min:',
							self.fit_range_min,
							is_int = True)
		self.textbox_range_max = setup_textbox(
							self.range_textbox_select,
							toolbar_layout, 'Max:',
							self.fit_range_max,
							is_int = True)
		return(toolbar_layout)
	
	def setup_focus_layout (self):
		focus_layout = QVBoxLayout()
		#
		focus_layout.addWidget(QLabel('XY Working Space'))
		self.button_select = setup_button(
							self.select_bounds,
							focus_layout, 'Select Box',
							toggle = True)
		self.button_reset = setup_button(
							self.reset_bounds,
							focus_layout, 'Select All')
		self.textbox_x_min = setup_textbox(
							self.bound_textbox_select,
							focus_layout, 'X Min:',
							is_int = True)
		self.textbox_x_max = setup_textbox(
							self.bound_textbox_select,
							focus_layout, 'X Max:',
							is_int = True)
		self.textbox_y_min = setup_textbox(
							self.bound_textbox_select,
							focus_layout, 'Y Min:',
							is_int = True)
		self.textbox_y_max = setup_textbox(
							self.bound_textbox_select,
							focus_layout, 'Y Max:',
							is_int = True)
		self.checkbox_zoom = setup_checkbox(
							self.zoom_checkbox,
							focus_layout, 'zoomed',
							self.zoomed)
		self.checkbox_flip = setup_checkbox(
							self.flip_checkbox,
							focus_layout, 'flipped',
							False)
		#
		focus_layout.addStretch()
		horizontal_separator(focus_layout, self.palette())
		#
		focus_layout.addWidget(QLabel('Z Bounds' + \
								' (<font color="red">TODO</font>)'))
		self.textbox_z_min = setup_textbox(
							self.bound_textbox_select,
							focus_layout, 'Z Min:',
							is_int = True)
		self.textbox_z_max = setup_textbox(
							self.bound_textbox_select,
							focus_layout, 'Z Max:',
							is_int = True)
		#
		focus_layout.addStretch()
		horizontal_separator(focus_layout, self.palette())
		#
		focus_layout.addWidget(QLabel('Time Bounds'))
		self.textbox_t_min = setup_textbox(
							self.bound_textbox_select,
							focus_layout, 'T Min:',
							is_int = True)
		self.textbox_t_max = setup_textbox(
							self.bound_textbox_select,
							focus_layout, 'T Max:',
							is_int = True)
		#
		focus_layout.addStretch()
		horizontal_separator(focus_layout, self.palette())
		#
		focus_layout.addWidget(QLabel('Average Tracking'))
		self.textbox_gaussian_coarse = setup_textbox(
							self.bound_textbox_select,
							focus_layout, 'Gaussian Filter:',
							is_int = True)
		self.button_track_coarse = setup_button(
							lambda: self.track_coarse('ski'),
							focus_layout, 'Track Average')
		#
		focus_layout.addStretch()
		#
		self.setup_bound_textboxes()
		return focus_layout
	
	def setup_points_layout (self):
		points_layout = QVBoxLayout()
		#
		points_layout.addWidget(QLabel('Uniform Grid'))
		self.textbox_grid_x = setup_textbox(
							self.points_textbox_select,
							points_layout, 'Grid X:',
							is_int = True)
		self.textbox_grid_y = setup_textbox(
							self.points_textbox_select,
							points_layout, 'Grid Y:',
							is_int = True)
		self.button_make_grid = setup_button(
							self.make_grid,
							points_layout, 'Make Grid')
		#
		points_layout.addStretch()
		horizontal_separator(points_layout, self.palette())
		#
		points_layout.addWidget(QLabel('Bright Points'))
		self.channel_selector = setup_combobox(
							self.channel_select,
							points_layout, 'Channel:')
		self.textbox_neighbourhood = setup_textbox(
							self.points_textbox_select,
							points_layout, 'Neighbourhood:',
							is_int = True)
		self.textbox_threshold = setup_textbox(
							self.points_textbox_select,
							points_layout, 'Threshold Diff:',
							is_int = True)
		self.textbox_gaussian = setup_textbox(
							self.points_textbox_select,
							points_layout, 'Gaussian Filter:',
							is_int = True)
		self.button_find_points = setup_button(
							self.find_points,
							points_layout, 'Find Points')
		#
		points_layout.addStretch()
		horizontal_separator(points_layout, self.palette())
		#
		points_layout.addWidget(QLabel('Manual Selection'))
		self.button_select_points = setup_button(
							self.select_points_button,
							points_layout, 'Select Points',
							toggle = True)
		self.button_add_points = setup_button(
							self.add_points_button,
							points_layout, 'Add Points',
							toggle = True)
		self.button_delete_points = setup_button(
							self.delete_points_button,
							points_layout, 'Delete Points',
							toggle = True)
		self.button_move_points = setup_button(
							self.move_points_button,
							points_layout, 'Move Points',
							toggle = True)
		self.button_remove_bad = setup_button(
							self.remove_bad_points,
							points_layout, 'Remove Bad')
		self.button_clear_points = setup_button(
							self.clear_points,
							points_layout, 'Clear Points')
		#
		points_layout.addStretch()
		horizontal_separator(points_layout, self.palette())
		#
		points_layout.addWidget(QLabel('Points Tracking'))
		self.textbox_search_dist = setup_textbox(
							self.points_textbox_select,
							points_layout, 'Search Distance:',
							is_int = True)
		self.textbox_gaussian_fine = setup_textbox(
							self.points_textbox_select,
							points_layout, 'Gaussian Filter:',
							is_int = True)
		self.button_track_fine_fft = setup_button(
							lambda: self.track_fine('ski'),
							points_layout, 'Track Points')
	#	self.button_track_fine_con = setup_button(
	#						lambda: self.track_fine('con'),
	#						points_layout, 'Track Points (Con)')
		#
		points_layout.addStretch()
		#
		self.setup_points_textboxes()
		#
		return points_layout
	
	def setup_strain_layout (self):
		strain_layout = QVBoxLayout()
		#
		strain_layout.addWidget(QLabel('Triangulation'))
		self.textbox_maxlength = setup_textbox(
							self.strain_textbox_select,
							strain_layout, 'Max Length:',
							is_int = True)
		self.button_triangulate = setup_button(
							self.triangulate,
							strain_layout, 'Triangulate')
		self.button_save_lengths = setup_button(
							self.export_lengths,
							strain_layout, 'Export Lengths')
		#
		strain_layout.addStretch()
		horizontal_separator(strain_layout, self.palette())
		#
		strain_layout.addWidget(QLabel('Strain Analysis'))
		self.button_compute_strains = setup_button(
							self.compute_strains,
							strain_layout, 'Compute Strains')
		self.strain_selector = setup_combobox(
							self.strain_select,
							strain_layout, 'Strain Display:')
		self.checkbox_strain = setup_checkbox(
							self.strain_checkbox,
							strain_layout, 'show strain',
							self.show_strain)
		#
		strain_layout.addStretch()
		horizontal_separator(strain_layout, self.palette())
		#
		strain_layout.addWidget(QLabel('Elastography'))
		#
		self.elast_direction_selector = setup_combobox(
							self.elast_direction_select,
							strain_layout, 'Direction:')
		self.textbox_frequency = setup_textbox(
							self.strain_textbox_select,
							strain_layout, 'Frequency (Hz):',
							is_int = False)
#		self.textbox_freq_thresh = setup_textbox(
#							self.strain_textbox_select,
#							strain_layout, 'Freq Threshold:',
#							is_int = False)
		self.textbox_amp_thresh = setup_textbox(
							self.strain_textbox_select,
							strain_layout, 'Amp Threshold:',
							is_int = False)
		self.textbox_fit_thresh = setup_textbox(
							self.strain_textbox_select,
							strain_layout, 'Fit Threshold:',
							is_int = False)
		self.textbox_fit_gauss_rad = setup_textbox(
							self.strain_textbox_select,
							strain_layout, 'Gaussian Radius:',
							is_int = False)
		self.button_frequency = setup_button(
							self.guess_frequency,
							strain_layout, 'Guess Frequency')
		self.button_fit_average = setup_button(
							self.fit_average,
							strain_layout, 'Fit Average')
		self.button_fit_points = setup_button(
							self.fit_points,
							strain_layout, 'Fit Points')
		self.button_phase_plot = setup_button(
							self.plot_phases,
							strain_layout, 'Plot Phases')
		self.button_derivatives_plot = setup_button(
							self.plot_derivatives,
							strain_layout, 'Plot Derivatives')
		#
		strain_layout.addStretch()
		horizontal_separator(strain_layout, self.palette())
		#
		strain_layout.addWidget(QLabel('Conduction Velocity'))
		#
		strain_layout.addStretch()
		horizontal_separator(strain_layout, self.palette())
		#
		self.setup_strain_textboxes()
		#
		return strain_layout
	
	def setup_colours_layout (self):
		self.colours_layout = QVBoxLayout()
		return self.colours_layout
	
	def setup_bottom_layout (self):
		bottom_layout = QHBoxLayout()
		self.button_open_file = setup_button(
					self.open_file,
					bottom_layout, 'Open File')
		self.button_reset = setup_button(
					self.reset_defaults,
					bottom_layout, 'Reset Defaults')
		self.progress_bar = setup_progress_bar(bottom_layout)
		self.button_cancel = setup_button(
					self.cancel_tracking,
					bottom_layout, 'Cancel' + \
								' (TODO)')
		self.button_save_tracks = setup_button(
					self.save_tracks,
					bottom_layout, 'Save Tracks')
		self.button_save_phases = setup_button(
					self.save_phases,
					bottom_layout, 'Save Phases')
		self.button_save_frames = setup_button(
					self.save_frames,
					bottom_layout, 'Save Frames')
		return bottom_layout
	
	def t_slider_select (self):
		self.t_position = self.slider_t.value()
		self.textbox_t.setText(str(self.t_position))
		self.refresh_image()
	
	def t_textbox_select (self):
		input_t = int(self.textbox_t.text())
		if input_t >= 0 and input_t < self.t_size:
			self.t_position = input_t
			self.slider_t.setValue(input_t)
			self.refresh_image()
	
	def t_min_button (self):
		self.t_lower = self.t_position
		self.setup_bound_textboxes()
	
	def t_max_button (self):
		self.t_upper = self.t_position
		self.setup_bound_textboxes()
	
	def z_slider_select (self):
		self.z_position = self.slider_z.value()
		self.textbox_z.setText(str(self.z_position))
		self.refresh_image()
	
	def z_textbox_select (self):
		input_z = int(self.textbox_z.text())
		if input_z >= 0 and input_z < self.z_size:
			self.z_position = input_z
			self.slider_z.setValue(input_z)
			self.refresh_image()
	
	def z_min_button (self):
		self.z_lower = self.z_position
		self.setup_bound_textboxes()
	
	def z_max_button (self):
		self.z_upper = self.z_position
		self.setup_bound_textboxes()
	
	def setup_range_textboxes (self):
		self.textbox_range_min.setText(str(self.fit_range_min))
		self.textbox_range_max.setText(str(self.fit_range_max))
	
	def range_textbox_select (self):
		self.fit_range_min = get_textbox(self.textbox_range_min,
											minimum_value = 0,
											maximum_value = self.t_size-1,
											is_int = True)
		self.fit_range_max = get_textbox(self.textbox_range_max,
											minimum_value = 0,
											maximum_value = self.t_size-1,
											is_int = True)
		self.plot_canvas.update_plot()
	
	def setup_bound_textboxes (self):
		self.textbox_x_min.setText(str(self.x_lower))
		self.textbox_x_max.setText(str(self.x_upper))
		self.textbox_y_min.setText(str(self.y_lower))
		self.textbox_y_max.setText(str(self.y_upper))
		self.textbox_z_min.setText(str(self.z_lower))
		self.textbox_z_max.setText(str(self.z_upper))
		self.textbox_t_min.setText(str(self.t_lower))
		self.textbox_t_max.setText(str(self.t_upper))
		self.textbox_gaussian_coarse.setText(str(self.coarse_gaussian))
	
	def bound_textbox_select (self):
		self.x_lower = get_textbox(self.textbox_x_min,
									minimum_value = 0,
									maximum_value = self.x_size-1,
									is_int = True)
		self.x_upper = get_textbox(self.textbox_x_max,
									minimum_value = self.x_lower,
									maximum_value = self.x_size-1,
									is_int = True)
		self.y_lower = get_textbox(self.textbox_y_min,
									minimum_value = 0,
									maximum_value = self.y_size-1,
									is_int = True)
		self.y_upper = get_textbox(self.textbox_y_max,
									minimum_value = self.y_lower,
									maximum_value = self.y_size-1,
									is_int = True)
		self.z_lower = get_textbox(self.textbox_z_min,
									minimum_value = 0,
									maximum_value = self.z_size-1,
									is_int = True)
		self.z_upper = get_textbox(self.textbox_z_max,
									minimum_value = self.z_lower,
									maximum_value = self.z_size-1,
									is_int = True)
		self.t_lower = get_textbox(self.textbox_t_min,
									minimum_value = 0,
									maximum_value = self.t_size-1,
									is_int = True)
		self.t_upper = get_textbox(self.textbox_t_max,
									minimum_value = self.t_lower,
									maximum_value = self.t_size-1,
									is_int = True)
		self.coarse_gaussian = get_textbox(self.textbox_gaussian_coarse,
									minimum_value = 0,
									maximum_value = 12,
									is_int = True)
		self.canvas.focus_box = np.array(
									[[self.x_lower, self.x_upper],
									 [self.y_lower, self.y_upper]],
								dtype = int)
		if self.x_lower > 0 or self.x_upper < self.x_size or \
		   self.y_lower > 0 or self.y_upper < self.y_size:
			self.canvas.show_box = True
		else:
			self.canvas.show_box = False
		self.canvas.plot_box()
	
	def setup_points_textboxes (self):
		self.textbox_grid_x.setText(str(self.grid_number_x))
		self.textbox_grid_y.setText(str(self.grid_number_y))
		self.textbox_neighbourhood.setText(str(self.neighbourhood_size))
		self.textbox_threshold.setText(str(self.threshold_difference))
		self.textbox_gaussian.setText(str(self.gauss_deviation))
		self.textbox_search_dist.setText(str(self.search_distance))
		self.textbox_gaussian_fine.setText(str(self.fine_gaussian))
	
	def points_textbox_select (self):
		self.grid_number_x = get_textbox(self.textbox_grid_x,
										minimum_value = 0,
										maximum_value = 24,
										is_int = True)
		self.grid_number_y = get_textbox(self.textbox_grid_y,
										minimum_value = 0,
										maximum_value = 24,
										is_int = True)
		self.neighbourhood_size = get_textbox(self.textbox_neighbourhood,
										minimum_value = 0,
										maximum_value = 60,
										is_int = True)
		self.threshold_difference = get_textbox(self.textbox_threshold,
										minimum_value = 0,
										maximum_value = 120,
										is_int = True)
		self.gauss_deviation = get_textbox(self.textbox_gaussian,
										minimum_value = 0,
										maximum_value = 12,
										is_int = True)
		self.search_distance = get_textbox(self.textbox_search_dist,
										minimum_value = 0,
										maximum_value = 60,
										is_int = True)
		self.fine_gaussian = get_textbox(self.textbox_gaussian_fine,
										minimum_value = 0,
										maximum_value = 12,
										is_int = True)
	
	def setup_strain_textboxes (self):
		self.textbox_maxlength.setText(str(self.max_length))
		self.textbox_frequency.setText(str(self.fit_frequency))
#		self.textbox_freq_thresh.setText(str(self.elast_frequency_thresh))
		self.textbox_amp_thresh.setText(str(self.elast_amplitude_thresh))
		self.textbox_fit_thresh.setText(str(self.elast_fit_thresh))
		self.textbox_fit_gauss_rad.setText(str(self.elast_gaussian_radius))
	
	def strain_textbox_select (self):
		self.max_length = get_textbox(self.textbox_maxlength,
										minimum_value = 0,
										maximum_value = 240,
										is_int = False)
		self.fit_frequency = get_textbox(self.textbox_frequency,
										minimum_value = 0.,
										maximum_value = 1000.,
										is_int = False)
#		self.elast_frequency_thresh = get_textbox(self.textbox_freq_thresh,
#										minimum_value = 0.0001,
#										maximum_value = 1.,
#										is_int = False)
		self.elast_amplitude_thresh = get_textbox(self.textbox_amp_thresh,
										minimum_value = 0.0001,
										maximum_value = 1.,
										is_int = False)
		self.elast_fit_thresh = get_textbox(self.textbox_fit_thresh,
										minimum_value = 0.,
										maximum_value = 1000.,
										is_int = False)
		self.elast_gaussian_radius = get_textbox(self.textbox_fit_gauss_rad,
										minimum_value = 0.,
										maximum_value = 1000.,
										is_int = False)
	
	def channel_select (self, index):
		self.channel = index
	
	def strain_select (self, index):
		self.strain_direction = index
		self.update_strains()
	
	def elast_direction_select (self, index):
		self.elast_direction = index
	
	def reset_channel_selector (self):
		self.channel_selector.clear()
		self.channel_selector.addItems(self.channel_names)
		self.channel_selector.setCurrentIndex(0)
		self.channel = 0
	
	def reset_strain_direction (self):
		self.strain_selector.clear()
		self.strain_selector.addItems(np.array(['XX','YY','XY','VM']))
		self.strain_selector.setCurrentIndex(0)
		self.strain_direction = 0
	
	def reset_elast_direction (self):
		self.elast_direction_selector.clear()
		self.elast_direction_selector.addItems(np.array(['X','Y']))
		self.elast_direction_selector.setCurrentIndex(0)
		self.elast_direction = 0
	
	def zoom_checkbox (self):
		self.zoomed = self.checkbox_zoom.isChecked()
		self.canvas.set_zoom(self.zoomed)
	
	def flip_checkbox (self):
		flipped = self.checkbox_flip.isChecked()
		self.canvas.set_flip(flipped)
	
	def strain_checkbox (self):
		self.show_strain = self.checkbox_strain.isChecked()
		self.canvas.set_stain_overlay(self.show_strain)
	
	def select_bounds (self):
		if self.selecting_area:
			self.button_select.setChecked(False)
			self.selecting_area = False
		else:
			self.button_select.setChecked(True)
			self.clear_analysis()
			self.zoomed = False
			self.checkbox_zoom.setChecked(False)
			self.selecting_area = True
	
	def on_click (self, event):
		self.position = np.array([int(np.floor(event.xdata)),
								  int(np.floor(event.ydata))])
		if self.selecting_area:
		#	self.position = np.array([int(np.floor(event.xdata)),
		#							  int(np.floor(event.ydata))])
			self.canvas.mpl_disconnect(self.click_id)
			self.click_id = self.canvas.mpl_connect(
								'button_release_event', self.off_click)
			self.move_id = self.canvas.mpl_connect(
								'motion_notify_event', self.mouse_moved)
		#TODO: something if search done already
		else:
			if (self.position[0] < self.x_lower) or \
			   (self.position[0] > self.x_upper) or \
			   (self.position[1] < self.y_lower) or \
			   (self.position[1] > self.y_upper):
				pass
			if self.select_mode == 'Select':
				if self.track_points is None:
					return False
				if len(self.track_points) == 0:
					return False
				self.selected_point = np.argmin(np.linalg.norm(
						self.track_points - self.position, axis=1))
				self.update_points()
				if self.fit_results_points is None:
					return False
				if len(self.fit_results_points) == len(self.track_points):
					self.plot_canvas.update_plot(self.fit_results_points[
														self.selected_point])
			elif self.select_mode == 'Add':
				self.add_point()
			elif self.select_mode == 'Delete':
				self.remove_point()
			elif self.select_mode == 'Move':
				self.canvas.mpl_disconnect(self.click_id)
				self.click_id = self.canvas.mpl_connect(
								'button_release_event', self.off_click)
	
	def mouse_moved (self, event):
		if self.selecting_area:
			p_1 = np.array([int(np.floor(event.xdata)),
							int(np.floor(event.ydata))])
			p_2 = self.position
			self.canvas.plot_selector(p_1, p_2)
	
	def off_click (self, event):
		self.canvas.mpl_disconnect(self.click_id)
		self.canvas.mpl_disconnect(self.move_id)
		self.click_id = self.canvas.mpl_connect(
								'button_press_event', self.on_click)
		if self.selecting_area:
			p_1 = np.array([int(np.floor(event.xdata)),
							int(np.floor(event.ydata))])
			p_2 = self.position
			self.canvas.remove_selector()
			self.selecting_area = False
			x_lower = np.amin(np.array([p_1[0], p_2[0]]))
			x_upper = np.amax(np.array([p_1[0], p_2[0]]))
			y_lower = np.amin(np.array([p_1[1], p_2[1]]))
			y_upper = np.amax(np.array([p_1[1], p_2[1]]))
			self.x_lower = x_lower
			self.x_upper = x_upper
			self.y_lower = y_lower
			self.y_upper = y_upper
			self.setup_bound_textboxes()
			self.bound_textbox_select()
			self.button_select.setChecked(False)
			self.selecting_area = False
		elif self.select_mode == 'Move':
			pass #TODO
	
	def add_point (self):
		if self.track_points is not None and \
		   len(self.track_points) > 0:
			distances = np.linalg.norm(self.track_points - \
									   self.position, axis=1)
			closest = np.argmin(distances)
			if distances[closest] > self.neighbourhood_size:
				self.track_points = np.append(self.track_points,
											  [self.position],
												axis=0)
		else:
			self.track_points = np.array([[0,0]], dtype = int)
			self.track_points[0] = self.position
		self.update_points()
	
	def remove_point (self):
		if self.track_points is not None:
			if len(self.track_points) > 0:
				distances = np.linalg.norm(self.track_points - \
										   self.position, axis=1)
				closest = np.argmin(distances)
				if distances[closest] <= self.neighbourhood_size * 2:
					if self.fine_search_done:
						if self.fine_results.shape[1] == \
									len(self.track_points):
							self.fine_results = np.delete(
													self.fine_results,
													closest, axis=0)
					if self.fit_results_points is not None:
						if len(self.fit_results_points) == \
									len(self.track_points):
							self.fit_results_points = np.delete(
													self.fit_results_points,
													closest, axis=0)
					if len(self.bad_points) > 0:
						self.bad_points = np.delete(self.bad_points,
								self.bad_points == closest, axis=0)
						self.bad_points -= (self.bad_points > closest)
					self.track_points = np.delete(self.track_points,
												  closest, axis=0)
			if len(self.track_points) == 0:
				self.track_points = None
		self.update_points()
	
	def reset_bounds (self):
		self.x_lower = 0
		self.x_upper = self.x_size
		self.y_lower = 0
		self.y_upper = self.y_size
		self.setup_bound_textboxes()
		self.bound_textbox_select()
		self.setup_points_textboxes()
		self.points_textbox_select()
	
	def open_file (self):
		if self.process_running:
			display_error('Cannot change file while processing!')
		elif self.file_dialog():
			if self.file_path.suffix.lower() == '.tif' or \
					self.file_path.suffix.lower() == '.tiff':
				self.image_stack = single_tif_stack(self.file_path)
			elif self.file_path.suffix.lower() == '.zip':
				self.image_stack = zip_tif_stack(self.file_path)
			elif self.file_path is not None:
				self.image_stack = BioImage(str(self.file_path))
			image_shape = self.image_stack.shape
			self.x_size = image_shape[4]
			self.y_size = image_shape[3]
			self.z_size = image_shape[2]
			self.c_size = image_shape[1]
			self.t_size = image_shape[0]
			self.channel_names = self.image_stack.channel_names
			self.scale = self.image_stack.physical_pixel_sizes[::-1]
			print(self.image_stack.standard_metadata) #TODO
			self.x_lower = 0
			self.x_upper = self.x_size-1
			self.y_lower = 0
			self.y_upper = self.y_size-1
			self.z_lower = 0
			self.z_upper = self.z_size-1
			self.t_lower = 0
			self.t_upper = self.t_size-1
			self.z_position = 0
			self.t_position = 0
			self.setup_bound_textboxes()
			self.setup_range_textboxes()
			update_slider(self.slider_t, value = self.t_position,
						  maximum_value = self.t_size-1)
			update_slider(self.slider_z, value = self.z_position,
						   maximum_value = self.z_size-1)
			self.canvas.clear_canvas()
			self.clear_analysis()
			self.refresh_image()
			self.update_points()
			self.update_edges()
			self.reset_channel_selector()
			self.reset_strain_direction()
			self.reset_elast_direction()
			self.instruction_text.setText('Use "focus" tab to setup ' + \
										  'working area and "points" ' + \
										  'tab to choose points to track.')
	
	def file_dialog (self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file_name, _ = QFileDialog.getOpenFileName(self,
								'Open Microscope File', '',
								'CZI Files (*.czi);;' + \
								'ND2 Files (*.nd2);;' + \
								'ZIP Files (*.zip);;' + \
								'TIF Files (*.tif);;' + \
								'All Files (*)',
								options=options)
		if file_name == '':
			return False
		else:
			file_path = Path(file_name)
			if file_path.suffix.lower() == '.nd2' or \
			   file_path.suffix.lower() == '.czi' or \
			   file_path.suffix.lower() == '.tif' or \
			   file_path.suffix.lower() == '.zip':
				self.file_path = file_path
				return True
			else:
				self.file_path = None
				return False
	
	def refresh_image (self):
		self.image_array = get_image(self.image_stack,
									 self.t_position,
									 self.z_position,
									 self.channel)
		if self.coarse_search_done and self.coarse_results is not None and \
		   self.t_position > self.t_lower and \
		   self.t_position <= self.t_upper:
			if self.t_position-self.t_lower-1 < len(self.coarse_results):
				drift_adjust = self.coarse_results[
									self.t_position-self.t_lower-1]
			else:
				drift_adjust = np.array([0.,0.])
		else:
			drift_adjust = np.array([0.,0.])
		self.canvas.update_image(self.image_array, drift_adjust)
		self.update_points()
		self.update_edges()
		self.update_strains()
	
	def update_points (self, points = None, bad_points = None):
		if points is not None:
			self.track_points = points
		if bad_points is not None:
			self.bad_points = bad_points
		shifts = np.zeros_like(self.track_points)
		if self.fine_search_done and self.fine_results is not None and \
		   self.t_position > self.t_lower and \
		   self.t_position <= self.t_upper:
			if self.t_position-self.t_lower-1 < len(self.fine_results):
				shifts = self.fine_results[self.t_position - self.t_lower - 1]
			else:
				shifts = np.zeros_like(self.track_points)
		self.canvas.update_points(self.track_points, shifts,
									self.bad_points, self.selected_point)
	
	def update_edges (self, edges = None):
		if edges is not None:
			self.edges = edges
		self.canvas.update_edges(self.edges)
	
	def update_triangles (self, triangles = None):
		if triangles is not None:
			self.triangles = triangles
		self.canvas.update_triangles(self.triangles)
	
	def update_strains (self):
		if self.strains_done and self.t_position > self.t_lower and \
				self.t_position < self.t_upper and \
				len(self.strains) == self.t_upper - self.t_lower:
			max_val = np.amax(
				[-np.amin(self.strains[:,:,self.strain_direction]),
				  np.amax(self.strains[:,:,self.strain_direction])])
			self.canvas.update_strains(
						self.strains[self.t_position-self.t_lower,:,
										self.strain_direction], max_val)
		else:
			self.canvas.update_strains(None)
	
	def make_grid (self):
		x_values = np.linspace(self.x_lower, self.x_upper,
							   self.grid_number_x, endpoint = True,
							   dtype = int)
		y_values = np.linspace(self.y_lower, self.y_upper,
							   self.grid_number_y, endpoint = True,
							   dtype = int)
		x_grid, y_grid = np.meshgrid(x_values, y_values)
		centres = np.vstack([x_grid.flatten(), y_grid.flatten()]).T
		self.track_points = centres
		self.find_bad_points()
		self.update_points()
	
	def find_points (self):
		if self.coarse_search_done == True:
			coarse_results = self.coarse_results
			self.clear_analysis()
			self.coarse_results = coarse_results
			self.coarse_search_done = True
		else:
			self.clear_analysis()
		if len(self.image_array.shape) == 2:
			frame = self.image_array[self.y_lower:self.y_upper,
									 self.x_lower:self.x_upper]
		else:
			frame = self.image_array[self.y_lower:self.y_upper,
									 self.x_lower:self.x_upper,
									 self.channel]
		centres = find_centres(frame,
							   self.neighbourhood_size,
							   self.threshold_difference,
							   self.gauss_deviation) + \
					np.array([[self.x_lower,self.y_lower]])
		self.track_points = centres
		self.find_bad_points()
		self.update_points()
	
	def select_points_button (self):
		if self.select_mode == 'Select':
			self.select_mode = 'None'
			self.button_select_points.setChecked(False)
		else:
			self.select_mode = 'Select'
			self.button_select_points.setChecked(True)
		self.button_add_points.setChecked(False)
		self.button_delete_points.setChecked(False)
		self.button_move_points.setChecked(False)
	
	def add_points_button (self):
		if self.select_mode == 'Add':
			self.select_mode = 'None'
			self.button_add_points.setChecked(False)
		else:
			self.select_mode = 'Add'
			self.button_add_points.setChecked(True)
		self.button_select_points.setChecked(False)
		self.button_delete_points.setChecked(False)
		self.button_move_points.setChecked(False)
	
	def delete_points_button (self):
		if self.select_mode == 'Delete':
			self.select_mode = 'None'
			self.button_delete_points.setChecked(False)
		else:
			self.select_mode = 'Delete'
			self.button_delete_points.setChecked(True)
		self.button_select_points.setChecked(False)
		self.button_add_points.setChecked(False)
		self.button_move_points.setChecked(False)
	
	def move_points_button (self):
		if self.select_mode == 'Move':
			self.select_mode = 'None'
			self.button_move_points.setChecked(False)
		else:
			self.select_mode = 'Move'
			self.button_move_points.setChecked(True)
		self.button_select_points.setChecked(False)
		self.button_add_points.setChecked(False)
		self.button_delete_points.setChecked(False)
	
	def find_bad_points (self):
		bad_points = np.array([], dtype = int)
		average_intensity = np.mean(self.image_array[
										self.x_lower:self.x_upper,
										self.y_lower:self.y_upper])
		minimum_intensity = np.amin(self.image_array[
										self.x_lower:self.x_upper,
										self.y_lower:self.y_upper])
		for index, point in enumerate(self.track_points):
			x_lower = int(max(point[0]-self.search_distance, 0))
			x_upper = int(min(point[0]+self.search_distance, self.x_size-1))
			y_lower = max(point[1]-self.search_distance, 0)
			y_upper = min(point[1]+self.search_distance, self.y_size-1)
			patch = self.image_array[y_lower:y_upper, x_lower:x_upper]
			patch_max = np.amax(patch)
			patch_min = np.amin(patch)
			if patch_max - patch_min < (average_intensity -\
										 minimum_intensity)/2:
				bad_points = np.append(bad_points, index)
		self.bad_points = bad_points
	
	def remove_bad_points (self):
		if self.bad_points is None or len(self.bad_points) == 0:
			return
		if self.fine_search_done:
			if len(self.fine_results) == len(self.track_points):
				self.fine_results = np.delete(self.fine_results,
										self.bad_points, axis=1)
		if self.fit_results_points is not None:
			if len(self.fit_results_points) == len(self.track_points):
				self.fit_results_points = np.delete(self.fit_results_points,
										self.bad_points, axis=0)
		self.track_points = np.delete(self.track_points, self.bad_points,
										axis=0)
		self.bad_points = np.array([], dtype = int)
		self.update_points()
	
	def clear_points (self):
		self.track_points = np.zeros((0,2), dtype = int)
		self.bad_points = np.zeros(0, dtype = int)
		self.update_points()
		self.edges = np.zeros((0,2), dtype = int)
		self.triangles = np.zeros((0,2), dtype = int)
		self.update_edges()
	
	def track_coarse (self, tracking_method = 'ski'):
		if self.process_running:
			return
		elif (self.x_lower < 0) or \
			 (self.x_upper >= self.x_size) or \
			 (self.y_lower < 0) or \
			 (self.y_upper >= self.y_size):
			self.instruction_text.setText('Focus area must be ' + \
										  'defined within image area.')
		elif self.t_upper - self.t_lower < 1:
			self.instruction_text.setText('Must have at least two frames. ' + \
										  '(Adjust time bounds.)')
		else:
			self.process_running = True
		#	try:
			self.track_coarse_process(tracking_method)
	
	def track_fine (self, tracking_method = 'ski'):
		if self.process_running:
			return
		elif (self.x_lower < 0) or \
			 (self.x_upper >= self.x_size) or \
			 (self.y_lower < 0) or \
			 (self.y_upper >= self.y_size):
			self.instruction_text.setText('Focus area must be ' + \
											  'defined within image area.')
		elif self.t_upper - self.t_lower < 1:
			self.instruction_text.setText('Must have at least two frames. ' + \
										  '(Adjust time bounds.)')
		elif self.track_points is None:
			self.instruction_text.setText('Must choose some points to track.')
		elif len(self.track_points) == 0:
			self.instruction_text.setText('Must choose some points to track.')
		else:
			self.process_running = True
		#	try:
			self.track_fine_process(tracking_method)
	
	def cancel_tracking (self):
		if self.process_running:
			if self.process is not None:
				self.process.quit()
			self.progress_bar.setMinimum(0)
			self.progress_bar.setFormat('')
			self.progress_bar.setMaximum(1)
			self.progress_bar.setValue(0)
			self.process_running = False
			self.process = None
	
	def track_coarse_process (self, tracking_method = 'ski'):
		coarse_results = np.zeros((self.t_upper - self.t_lower, 2),
									dtype = float)
		update_progress_bar(self.progress_bar, value = self.t_lower,
							minimum_value = self.t_lower,
							maximum_value = self.t_upper,
							text = 'Tracking Points: %p%')
		result = np.array([0.,0.], dtype = float)
		if tracking_method == 'ski':
			window = None
		else:
			if self.tukey_alpha is not None:
				window = np.outer(windows.tukey(self.y_upper - self.y_lower,
												self.tukey_alpha),
								  windows.tukey(self.x_upper - self.x_lower,
												self.tukey_alpha))
			else:
				window = np.outer(windows.hann(self.y_upper - self.y_lower),
								  windows.hann(self.x_upper - self.x_lower))
		full_image_1 = get_image(self.image_stack,
								 self.t_lower,
								 self.z_position,
								 self.channel).astype(float)
		for t in np.arange(self.t_lower, self.t_upper):
			int_result = np.around(result).astype(int)
			x_lower = max(self.x_lower + int_result[0], 0)
			x_upper = min(self.x_upper + int_result[0], self.x_size-1)
			y_lower = max(self.y_lower + int_result[1], 0)
			y_upper = min(self.y_upper + int_result[1], self.y_size-1)
			image_1 = full_image_1[y_lower:y_upper, x_lower:x_upper]
			image_1 = ndi.gaussian_filter(image_1, self.coarse_gaussian)
		#	image_1 = ndi.spline_filter(image_1)
			full_image_2 = get_image(self.image_stack,
									 t+1,
									 self.z_position,
									 self.channel).astype(float)
			image_2 = full_image_2[y_lower:y_upper, x_lower:x_upper]
			image_2 = ndi.gaussian_filter(image_2, self.coarse_gaussian)
		#	image_2 = ndi.spline_filter(image_2)
			result += get_shift(image_1, image_2,
								tracking_method = tracking_method,
								window = window)
			coarse_results[t - self.t_lower] = result
			full_image_1 = full_image_2
			update_progress_bar(self.progress_bar, value = t)
		self.coarse_results = coarse_results
		self.coarse_search_done = True
		clear_progress_bar(self.progress_bar)
		self.process_running = False
		self.results_elast_avg = FitResults(
				time_points = np.arange(self.t_lower, self.t_upper),
				data_points = coarse_results[:,self.elast_direction],
				startpoint = None, endpoint = None,
				fit_function = None, best_params = None)
		self.plot_average()
		self.guess_range()
		self.guess_frequency()
	
	def track_fine_process (self, tracking_method = 'ski'):
		fine_results = np.zeros((self.t_upper - self.t_lower,
								 len(self.track_points), 2),
									dtype = float)
		if tracking_method == 'ski':
			window = None
		else:
			if self.tukey_alpha is not None:
				window = np.outer(windows.tukey(self.search_distance*4+1,
												self.tukey_alpha),
								  windows.tukey(self.search_distance*4+1,
												self.tukey_alpha))
			else:
				window = np.outer(windows.hann(self.search_distance*4+1),
								  windows.hann(self.search_distance*4+1))
		update_progress_bar(self.progress_bar, value = self.t_lower,
							minimum_value = self.t_lower,
							maximum_value = self.t_upper,
							text = 'Tracking Points: %p%')
		results = np.zeros_like(self.track_points).astype(float)
		full_image_1 = get_image(self.image_stack,
								 self.t_lower,
								 self.z_position,
								 self.channel).astype(float)
		full_image_1 = ndi.gaussian_filter(full_image_1,
											self.fine_gaussian)
	#	full_image_1 = ndi.spline_filter(full_image_1)
		try:
			for t in np.arange(self.t_lower, self.t_upper):
				t_index = t - self.t_lower
				full_image_2 = get_image(self.image_stack,
										 t+1,
										 self.z_position,
										 self.channel).astype(float)
				full_image_2 = ndi.gaussian_filter(full_image_2,
													self.fine_gaussian)
			#	full_image_2 = ndi.spline_filter(full_image_2)
				for p_index, o_point in enumerate(self.track_points):
					point = o_point + np.around(results[p_index]).astype(int)
					x_1_lower = point[0] - self.search_distance * 2
					x_1_upper = point[0] + self.search_distance * 2
					y_1_lower = point[1] - self.search_distance * 2
					y_1_upper = point[1] + self.search_distance * 2
					x_2_lower = point[0] - self.search_distance * 2
					x_2_upper = point[0] + self.search_distance * 2
					y_2_lower = point[1] - self.search_distance * 2
					y_2_upper = point[1] + self.search_distance * 2
					if self.coarse_search_done:
						if len(self.coarse_results) == self.t_upper - \
													   self.t_lower:
							if t_index > 0:
								int_result = np.around(
									self.coarse_results[t_index-1]).astype(int)
								x_1_lower += int_result[0]
								x_1_upper += int_result[0]
								y_1_lower += int_result[1]
								y_1_upper += int_result[1]
							int_result = np.around(
									self.coarse_results[t_index]).astype(int)
							x_2_lower += int_result[0]
							x_2_upper += int_result[0]
							y_2_lower += int_result[1]
							y_2_upper += int_result[1]
					image_1 = full_image_1[y_1_lower:y_1_upper+1,
										   x_1_lower:x_1_upper+1]
					image_2 = full_image_2[y_2_lower:y_2_upper+1,
										   x_2_lower:x_2_upper+1]
					results[p_index] += get_shift(image_1, image_2,
										tracking_method = tracking_method,
										window = window,
										search_distance = self.search_distance)
				fine_results[t_index] = results
				full_image_1 = full_image_2
				update_progress_bar(self.progress_bar, value = t)
		except Exception as error:
			message = "An error occurred:"+type(error).__name__+"–"+str(error)
			display_error(message)
		self.fine_results = fine_results
		self.fine_search_done = True
		clear_progress_bar(self.progress_bar)
		self.process_running = False
	
	def plot_average (self):
		if self.results_elast_avg is not None:
			self.plot_canvas.update_plot(self.results_elast_avg)
	
	def guess_range (self):
		if (not self.coarse_search_done) or (self.coarse_results is None):
			return
		time_points = np.arange(self.t_lower, self.t_upper)
		data_points = self.coarse_results[:,self.elast_direction]
		shifted = np.abs(data_points-data_points[0])
		mask = shifted > np.amax(shifted)*0.2
		self.fit_range_min = np.argmax(mask) + 5 # let it settle
		self.fit_range_max = self.t_upper
		self.setup_range_textboxes()
		results = FitResults(
				time_points = np.arange(self.t_lower, self.t_upper),
				data_points = self.coarse_results[:,self.elast_direction],#TODO
				startpoint = self.fit_range_min,
				endpoint = self.fit_range_max,
				fit_function = None, best_params = None)
		self.plot_canvas.update_plot(results)
	
	def guess_frequency (self):
		if (not self.coarse_search_done) or (self.coarse_results is None):
			return
		time_points = np.arange(self.t_lower, self.t_upper)
		time_points = time_points[self.fit_range_min:self.fit_range_max]
		data_points = self.coarse_results[:,self.elast_direction].copy()
		data_points = data_points[self.fit_range_min:self.fit_range_max]
		data_points -= np.mean(data_points)
		t = np.linspace(time_points[0],
						time_points[-1], 2500) #TODO: interval
		interpolated = np.interp(t, time_points, data_points)
		fourier = fftpack.fft(interpolated)
		frequencies = fftpack.fftfreq(interpolated.size, d=t[1]-t[0])
		self.fit_frequency = frequencies[np.argmax(np.abs(fourier))]
		self.setup_strain_textboxes()
	
	def fit_average (self):
		if (not self.coarse_search_done) or (self.coarse_results is None):
			return
		time_points = np.arange(self.t_lower, self.t_upper)
		fit_time_points = time_points[self.fit_range_min:self.fit_range_max]
		data_points = self.coarse_results[:,self.elast_direction]
		fit_data_points = data_points[self.fit_range_min:self.fit_range_max]
		initial_guess = np.array([(
				np.amax(fit_data_points)-np.amin(fit_data_points))/2,
				self.fit_frequency*2*np.pi,
				0.0,
				np.mean(fit_data_points)
				])
		try:
			fit_result, cov_matrix, infodict, *_ = curve_fit(sine,
														fit_time_points,
														fit_data_points,
														initial_guess,
														full_output = True)
#			print(np.sum(infodict['fvec']**2) / np.abs(fit_result[0]) /\
#									(len(fit_time_points)-4))
			self.fit_results_average = FitResults(
					time_points = time_points,
					data_points = data_points,
					startpoint = self.fit_range_min,
					endpoint = self.fit_range_max,
					fit_function = sine, best_params = fit_result)
		except Exception as error:
			message = "An error occurred:"+type(error).__name__+"–"+str(error)
			display_error(message)
		self.plot_canvas.update_plot(self.fit_results_average)
		
	
	def fit_points (self):
		if (not self.coarse_search_done) or (self.coarse_results is None):
			return
		if (not self.fine_search_done) or (self.fine_results is None):
			return
		if self.fit_results_average is None:
			self.fit_average()
		results = self.fine_results + \
							self.track_points[np.newaxis,:,:].astype(float)
		if self.coarse_search_done and \
				len(self.coarse_results) == len(self.fine_results):
			results += self.coarse_results[:,np.newaxis,:]
		average_amplitude = self.fit_results_average.best_params[0]
		average_frequency = self.fit_results_average.best_params[1]
		average_phase = self.fit_results_average.best_params[2]
		average_phase = average_phase % (2*np.pi)
		average_offset = self.fit_results_average.best_params[3]
		initial_guess = np.array([
				average_amplitude,
			#	average_frequency,
				average_phase,
				0	#	average_offset
				])
		fit_function = lambda x,A,phi,B: sine(x,A,average_frequency,phi,B)
		time_points = np.arange(self.t_lower, self.t_upper)
		fit_time_points = time_points[self.fit_range_min:self.fit_range_max]
		self.bad_points = np.zeros(results.shape[1], dtype = bool)
		self.fit_results_points = np.empty(results.shape[1], dtype = object)
		for point_index in range(results.shape[1]):
			data_points = results[:,point_index,self.elast_direction]
			data_points -= np.mean(data_points)
			fit_data_points = data_points[
									self.fit_range_min:self.fit_range_max]
			try:
				fit_result, cov_matrix, infodict, *_ = curve_fit(
														fit_function,
														fit_time_points,
														fit_data_points,
														initial_guess,
														full_output = True)
				self.fit_results_points[point_index] = FitResults(
											time_points = time_points,
											data_points = data_points,
											startpoint = self.fit_range_min,
											endpoint = self.fit_range_max,
											fit_function = sine,
											best_params = [
													fit_result[0],
													average_frequency,
													fit_result[1],
													fit_result[2] ])
			#	if np.abs(fit_result[1]-average_frequency)/average_frequency >\
			#								self.elast_frequency_thresh:
			#		self.bad_points[point_index] = True
				if np.abs(fit_result[0]-average_amplitude)/average_amplitude >\
											self.elast_amplitude_thresh:
					self.bad_points[point_index] = True
				if np.sum(infodict['fvec']**2) / np.abs(fit_result[0]) /\
									(len(fit_time_points)-4) >\
											self.elast_fit_thresh:
					self.bad_points[point_index] = True
			# if fit didn't work mark point bad.
			except Exception as error:
				message = "An error occurred:"+type(error).__name__+"–"+str(error)
				print(message)
				self.bad_points[point_index] = True
		if np.any(np.logical_not(self.bad_points)):
			self.phase_array = self.track_points[self.bad_points != True]
			phases = np.zeros(len(self.phase_array), dtype=float)
			phase_index = 0
			for index in range(len(self.track_points)):
				if self.bad_points[index]:
					continue
				phase = self.fit_results_points[index].best_params[2]
				phases[phase_index] = (phase+np.pi) % (2*np.pi) - np.pi
				phase_index+=1
			self.phase_array = np.hstack([self.phase_array,
											phases[:,np.newaxis]])
			x = self.phase_array[:,0]
			y = self.phase_array[:,1]
			z = self.phase_array[:,2]
			delta_x = x[:,None]-x
			delta_y = y[:,None]-y
			sigma = self.elast_gaussian_radius
			weights = np.exp(-(delta_x*delta_x+delta_y*delta_y) /\
						(2*sigma*sigma)) / (np.sqrt(2*np.pi)*sigma)
			weights /= np.sum(weights, axis=1, keepdims=True)
			self.phase_array[:,2] = np.dot(weights,z)
		self.update_points()
		# bad_points is an array of indices not a boolean array elsewhere...
		self.bad_points = np.argwhere(self.bad_points == True).flatten()
		self.remove_bad_points()
		self.triangulate()
		self.update_edges()
		self.update_triangles()
		derivatives = self.phase_array.copy()
		derivatives[:,2] = 0
		num_triangles = np.zeros(len(derivatives), dtype=float)
		for triangle in self.triangles:
			vector_1 = self.phase_array[triangle[1]] - \
							self.phase_array[triangle[0]]
			vector_2 = self.phase_array[triangle[2]] - \
							self.phase_array[triangle[0]]
			cross_product = np.cross(vector_1, vector_2)
			if cross_product[2] < 0:
				cross_product *= -1
			if self.elast_direction == 0:
				derivative = cross_product[1]/cross_product[2]
			else:
				derivative = cross_product[0]/cross_product[2]
			derivatives[triangle] += derivative
			num_triangles[triangle] += 1
		derivatives[:,2] /= num_triangles
		self.derivatives = derivatives
		# put into select mode
		if self.select_mode != 'Select':
			self.select_points_button()
	
	def plot_phases (self):
		if self.phase_array is None:
			return False
		if len(self.phase_array) == 0:
			return False
		try:
			fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
			ax.scatter(self.phase_array[:,0], self.phase_array[:,1],
								self.phase_array[:,2])
			ax.set_xlabel('X')
			ax.set_ylabel('Y')
			plt.show()
			return True
		except:
			return False
	
	def plot_derivatives (self):
		if not (self.fine_search_done and self.fine_results is not None):
			self.instruction_text.setText('Must do some tracking first.')
			return False
		if self.phase_array is None:
			self.instruction_text.setText('Must do fitting first.')
			return False
		if self.derivatives is None:
			return False
		try:
			plt.scatter(self.derivatives[:,0],
						self.derivatives[:,1],
						c = self.derivatives[:,2])
			plt.show()
			return True
		except:
			return False
	
	def triangulate (self):
		if len(self.track_points) > 3:
			self.edges, self.triangles = triangulate(self.track_points,
													 self.max_length)
		self.update_edges()
		self.update_triangles()
	
	def compute_strains (self):
		if len(self.triangles) == 0 or len(self.track_points) == 0:
			return
		if not (self.fine_search_done and self.fine_results is not None):
			self.instruction_text.setText('Must do some tracking first.')
			return
		if len(self.fine_results) != (self.t_upper-self.t_lower):
			self.instruction_text.setText('Redo tracking first.')
			return
		results = self.fine_results + \
							self.track_points[np.newaxis,:,:].astype(float)
		if self.coarse_search_done and \
				len(self.coarse_results) == len(self.fine_results):
			results += self.coarse_results[:,np.newaxis,:]
		results = np.insert(results, 0, self.track_points, axis=0)
		self.strains = np.zeros((self.t_upper - self.t_lower,
									len(self.track_points),4), dtype = float)
		update_progress_bar(self.progress_bar, value = self.t_lower,
							minimum_value = self.t_lower,
							maximum_value = self.t_upper,
							text = 'Computing Strains: %p%')
		for t in np.arange(self.t_lower, self.t_upper):
			areas = np.zeros((len(self.track_points)), dtype = float)
			t_index = t - self.t_lower
			update_progress_bar(self.progress_bar, value = t)
			for tri_index,triangle in enumerate(self.triangles):
				tri_1 = results[t_index,triangle,:]
				tri_1 = results[0,triangle,:]
				tri_2 = results[t_index+1,triangle,:]
				tri_strain = np.array(compute_strain(tri_1, tri_2))
				tri_strain = np.append(tri_strain, np.sqrt(
						tri_strain[0]**2 + tri_strain[1]**2 - \
						tri_strain[0]*tri_strain[1] + \
						3 * tri_strain[2]**2))
				area = np.abs(tri_1[0,0]*(tri_1[1,1]-tri_1[2,1]) + \
							  tri_1[1,0]*(tri_1[2,1]-tri_1[0,1]) + \
							  tri_1[2,0]*(tri_1[0,1]-tri_1[1,1]))/2
				self.strains[t_index,triangle,:] += tri_strain * area
				areas[triangle] += area
			self.strains[t_index,areas!=0,:] /= areas[areas!=0, np.newaxis]
	#	for time in np.arange(1,len(self.strains)):
	#		self.strains[time] += self.strains[time-1]
		self.strains_done = True
		clear_progress_bar(self.progress_bar)
		self.checkbox_strain.setChecked(True)
		self.strain_checkbox()
		self.update_strains()
		self.refresh_image()
	
	def export_lengths (self):
		if len(self.edges) > 0 and len(self.track_points) > 0:
			lengths = np.linalg.norm(
					self.track_points[self.edges[:,0],:] - \
					self.track_points[self.edges[:,1],:], axis=-1)
			if self.file_path is not None:
				np.savetxt(self.file_path.with_suffix(
					'.{0:s}.lengths.csv'.format(
									time.strftime("%Y.%m.%d-%H.%M.%S"))),
					lengths,  delimiter = ',')
	
	def save_tracks (self):
		if not (self.coarse_search_done or self.fine_search_done):
			self.instruction_text.setText('Must do tracking first.')
			return False
	#	results = np.around(self.fine_results).astype(int) + \
	#								self.track_points[np.newaxis,:,:]
		results = self.fine_results + \
							self.track_points[np.newaxis,:,:].astype(float)
		if self.coarse_search_done and \
				len(self.coarse_results) == len(self.fine_results):
			results += self.coarse_results[:,np.newaxis,:]
		results = results*self.scale[0:2]
		if self.strains_done and len(self.strains) == len(results):
			data_format = '%.18e', '%.18e', '%1d', '%1d', '%1d',\
											'%.18e', '%.18e', '%.18e'
			header = 'X,Y,Time,TrackID,ID,StrainXX,StrainYY,StrainXY'
			output_array = np.zeros(((results.shape[0]+1)*results.shape[1],8),
																dtype = float)
			counter = 1
			for point in range(results.shape[1]):
				output_array[point] = [
							self.track_points[point,0],
							self.track_points[point,1],
							0, point+1, counter, 0, 0, 0]
				counter += 1
			for time_point in range(results.shape[0]):
				for point in range(results.shape[1]):
					output_array[(time_point+1)*results.shape[1]+point] = [
							results[time_point,point,0],
							results[time_point,point,1],
							time_point+1, point+1, counter,
							self.strains[time_point,point,0],
							self.strains[time_point,point,1],
							self.strains[time_point,point,2]]
					counter += 1
		else:
			data_format = '%.18e', '%.18e', '%1d', '%1d', '%1d'
			header = 'X,Y,Time,TrackID,ID'
			output_array = np.zeros(((results.shape[0]+1)*results.shape[1],5),
																dtype = float)
			counter = 1
			for point in range(results.shape[1]):
				output_array[point] = [
							self.track_points[point,0],
							self.track_points[point,1],
							0, point+1, counter]
				counter += 1
			for time_point in range(results.shape[0]):
				for point in range(results.shape[1]):
					output_array[(time_point+1)*results.shape[1]+point] = [
							results[time_point,point,0],
							results[time_point,point,1],
							time_point+1, point+1, counter]
					counter += 1
		if self.file_path is not None:
			np.savetxt(self.file_path.with_suffix(
				'.{0:s}.csv'.format(time.strftime("%Y.%m.%d-%H.%M.%S"))),
				output_array,  delimiter = ',',
				fmt = data_format,
				header = header)
	
	def save_phases (self):
		if not (self.coarse_search_done or self.fine_search_done):
			self.instruction_text.setText('Must do tracking first.')
			return False
		if self.phase_array is None:
			self.instruction_text.setText('Must do fitting first.')
			return False
		data_format = '%.18e', '%.18e', '%.18e'
		header = 'X,Y,Phase'
		np.savetxt(self.file_path.with_suffix(
			'.{0:s}.phase.csv'.format(time.strftime("%Y.%m.%d-%H.%M.%S"))),
			self.phase_array,  delimiter = ',',
			fmt = data_format,
			header = header)
		
	
	def save_frames (self):
		if self.file_path is None:
			return
		dir_path = self.file_path.with_suffix(
					'.{0:s}'.format(time.strftime("%Y.%m.%d-%H.%M.%S")))
		Path.mkdir(dir_path)
		initial_time = self.t_position
		update_progress_bar(self.progress_bar, value = self.t_lower,
							minimum_value = self.t_lower,
							maximum_value = self.t_upper,
							text = 'Exporting Frames: %p%')
		for index, current_time in enumerate(
									range(self.t_lower, self.t_upper+1)):
			self.t_position = current_time
			self.refresh_image()
			current_frame = copy.deepcopy(self.canvas.fig)
			current_frame.axes[0].get_xaxis().set_visible(False)
			current_frame.axes[0].get_yaxis().set_visible(False)
			current_frame.savefig(dir_path/(f'frame_{index:03d}.png'),
									bbox_inches='tight',pad_inches = 0)
			update_progress_bar(self.progress_bar, value = current_time)
		self.t_position = initial_time
		self.refresh_image()
		clear_progress_bar(self.progress_bar)
		return dir_path

################################################################################

if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = Window()
	window.show()
	sys.exit(app.exec_())

################################################################################
# EOF

