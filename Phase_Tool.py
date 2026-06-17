#!/usr/bin/env python3

import sys
import time
import copy
import subprocess
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
from scipy.optimize import curve_fit
from matplotlib import tri
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qt5agg import (
							FigureCanvasQTAgg as FigureCanvas,
							NavigationToolbar2QT as NavigationToolbar
							)
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
from functools import partial
from pathlib import Path

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
				   initial_value = 0, is_int = False):
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
# find little neighbourhoods around a set of points #
#####################################################

def get_neighbourhoods (points, resolution = 50):
	neighbourhoods = []
	for point in points:
		distances = np.linalg.norm(points - point, axis=1)
		indicies = np.where(distances<resolution)[0]
		neighbourhoods.append(indicies)
	return neighbourhoods

################################################################################
# smooth out 2d data using guassian weights #
#############################################

def smooth_values (points, values, gaussian_radius = 0):
	if gaussian_radius > 0:
		x = points[:,0]
		y = points[:,1]
		z = values
		delta_x = x[:,None]-x
		delta_y = y[:,None]-y
		sigma = gaussian_radius
		weights = np.exp(-(delta_x*delta_x+delta_y*delta_y) /\
					(2*sigma*sigma)) / (np.sqrt(2*np.pi)*sigma)
		weights /= np.sum(weights, axis=1, keepdims=True)
		return np.dot(weights,z)
	else:
		return values

################################################################################
# matplotlib canvas widget #
############################

class MPLCanvas_PhaseMap(FigureCanvas):
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
		self.points = np.empty((0,2), dtype = float)
		self.phases = np.empty(0, dtype = float)
		self.excluded = np.empty(0, dtype = int)
		# plot objects
		self.points_plot = None
		self.excluded_plot = None
	
	def update_points (self, points = None, phases = None, excluded = None):
		if points is None:
			self.points = np.empty((0,2), dtype = float)
		else:
			self.points = points
		if phases is None:
			self.phases = np.empty(0, dtype = float)
		else:
			self.phases = phases
		if excluded is None:
			self.excluded = np.empty(0, dtype = int)
		else:
			self.excluded = excluded
		self.clear_canvas()
		self.plot_points()
	
	def remove_plot_element (self, plot_element):
		if plot_element is not None:
			if isinstance(plot_element,list):
				for sub_element in plot_element:
					self.remove_plot_element(sub_element)
			else:
				try:
					plot_element.remove()
				except:
					pass
	
	def clear_canvas (self):
		# plot objects
		self.remove_plot_element(self.points_plot)
		self.points_plot = None
		self.remove_plot_element(self.excluded_plot)
		self.excluded_plot = None
		self.ax.cla()
		self.draw()
	
	def plot_points (self):
		if self.points is None:
			return False
		elif len(self.points) == 0:
			return False
		if self.phases is None:
			return False
		elif len(self.phases) == 0:
			return False
		color_norm = self.phases - np.amin(self.phases)
		color_norm /= np.amax(color_norm)
		self.points_plot = self.ax.scatter(self.points[:,0], self.points[:,1],
										marker = '.',
										linestyle = '',
										color = plt.cm.viridis(color_norm),
										zorder = 5)
		if self.excluded is not None and \
		   len(self.excluded) > 0:
			self.excluded_plot = self.ax.plot(self.points[self.excluded,0],
											  self.points[self.excluded,1],
											marker = 'x',
											linestyle = '',
											color = 'tab:red',
											zorder = 6)
		self.draw()

################################################################################
# matplotlib canvas widget #
############################

class MPLCanvas_PhaseFit(FigureCanvas):
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
		self.points = np.empty((0,2), dtype = float)
		self.phases = np.empty(0, dtype = float)
		self.excluded = np.empty(0, dtype = int)
		self.fit_function = None
		# plot objects
		self.points_plot = None
		self.fit_plot = None
	
	def update_points (self, points = None, phases = None, excluded = None,
							fit_function = None):
		if points is None:
			self.points = np.empty((0,2), dtype = float)
		else:
			self.points = points
		if phases is None:
			self.phases = np.empty(0, dtype = float)
		else:
			self.phases = phases
		if excluded is None:
			self.excluded = np.empty(0, dtype = int)
		else:
			self.excluded = excluded
		self.fit_function = fit_function
		self.clear_canvas()
		self.plot_points()
		self.plot_fit()
	
	def remove_plot_element (self, plot_element):
		if plot_element is not None:
			if isinstance(plot_element,list):
				for sub_element in plot_element:
					self.remove_plot_element(sub_element)
			else:
				try:
					plot_element.remove()
				except:
					pass
	
	def clear_canvas (self):
		# plot objects
		self.remove_plot_element(self.points_plot)
		self.points_plot = None
		self.remove_plot_element(self.fit_plot)
		self.fit_plot = None
		self.ax.cla()
		self.draw()
	
	def plot_points (self):
		if self.points is None:
			return False
		elif len(self.points) == 0:
			return False
		if self.phases is None:
			return False
		elif len(self.phases) == 0:
			return False
		color_norm = self.phases - np.amin(self.phases)
		color_norm /= np.amax(color_norm)
		self.points_plot = self.ax.scatter(self.points[:,0], self.phases,
										marker = '.',
										linestyle = '',
										color = plt.cm.viridis(color_norm),
										zorder = 5)
		if self.excluded is not None and \
		   len(self.excluded) > 0:
			self.excluded_plot = self.ax.plot(self.points[self.excluded,0],
											  self.phases[self.excluded],
											marker = 'x',
											linestyle = '',
											color = 'tab:red',
											zorder = 6)
		self.ax.set_xlim([np.amin(self.points[:,0]),
						  np.amax(self.points[:,0])])
		self.ax.set_ylim([np.amin(self.phases),
						  np.amax(self.phases)])
		self.draw()
	
	def plot_fit (self):
		if self.points is None:
			return False
		elif len(self.points) == 0:
			return False
		if self.fit_function is None:
			return False
		x_points = np.linspace(np.amin(self.points[:,0]),
								np.amax(self.points[:,0]), 1000)
		self.fit_plot = self.ax.plot(x_points, self.fit_function(x_points),
									marker = '',
									linestyle = '-',
									color = 'tab:orange',
									zorder = 6)
		self.draw()

################################################################################
# matplotlib 3d canvas widget #
###############################

class MPL3DCanvas(FigureCanvas):
	def __init__ (self, parent=None, width=8, height=8, dpi=100):
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.ax = self.fig.add_subplot(111, projection='3d')
		FigureCanvas.__init__(self, self.fig)
		self.setParent(parent)
		FigureCanvas.setSizePolicy(self,
				QSizePolicy.Expanding,
				QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		self.fig.tight_layout()
		# stuff to plot
		self.points = np.empty((0,2), dtype = float)
		self.phases = np.empty(0, dtype = float)
		# plot objects
		self.points_plot = None
	
	def update_points (self, points = None, phases = None):
		if points is None:
			self.points = np.empty((0,2), dtype = float)
		else:
			self.points = points
		if phases is None:
			self.phases = np.empty(0, dtype = float)
		else:
			self.phases = phases
		self.clear_canvas()
		self.plot_points()
	
	def remove_plot_element (self, plot_element):
		if plot_element is not None:
			if isinstance(plot_element,list):
				for sub_element in plot_element:
					self.remove_plot_element(sub_element)
			else:
				try:
					plot_element.remove()
				except:
					pass
	
	def clear_canvas (self):
		# plot objects
		self.remove_plot_element(self.points_plot)
		self.points_plot = None
		self.ax.cla()
		#
		self.draw()
	
	def plot_points (self):
		if self.points is None:
			return False
		elif len(self.points) == 0:
			return False
		if self.phases is None:
			return False
		elif len(self.phases) == 0:
			return False
		color_norm = self.phases - np.amin(self.phases)
		color_norm /= np.amax(color_norm)
		self.points_plot = self.ax.scatter(self.points[:,0],
										self.points[:,1],
										self.phases,
										marker = '.',
										linestyle = '',
										color = plt.cm.viridis(color_norm),
										zorder = 5)
		self.draw()


################################################################################
# main window object #
######################

class Window(QWidget):
	def __init__ (self, points = None, phases = None, excluded = None):
		super().__init__()
		self.title = "Phase Map Analysis Tool"
		self.canvas_phasemap = MPLCanvas_PhaseMap()
		self.toolbar_phasemap = NavigationToolbar(self.canvas_phasemap, self)
		self.canvas_phasefit = MPLCanvas_PhaseFit()
		self.toolbar_phasefit = NavigationToolbar(self.canvas_phasefit, self)
		self.canvas_3d = MPL3DCanvas()
		self.toolbar_3d = NavigationToolbar(self.canvas_3d, self)
		#
		self.file_path = None
		self.click_id_map = None
		self.click_id_fit = None
		if points is None:
			self.points = np.empty((0,2), dtype = float)
		else:
			self.points = points
		if phases is None:
			self.phases = np.empty(0, dtype = float)
		else:
			self.phases = phases
		if excluded is None:
			self.excluded = np.empty(0, dtype = int)
		else:
			self.excluded = excluded
		self.mode_select = 'None' # 'Include', 'Exclude'
		self.shift = np.array([0.,0.])
		if points is not None:
			self.shift_origin()
		self.fit_function = None
		self.points_adjusted = None
		self.phases_adjusted = None
		self.stiffness_map = None
		self.polynomial_degree = 6
		self.gaussian_radius = 0.0
		self.resolution = 60.
		self.frequency = 5.
		self.local_degree = 2
		#
		self.setupGUI()
	
	def shift_origin (self):
		if self.points is None or len(self.points) == 0:
			return False
		self.shift = np.mean(self.points, axis=0)
		self.points -= self.shift
	
	def setupGUI (self):
		self.setWindowTitle(self.title)
		# layout for full window
		outer_layout = QVBoxLayout()
		# layout to hold plot layouts
		plots_layout = QHBoxLayout()
		#
		phasemap_layout = QVBoxLayout()
		phasemap_layout.addWidget(self.canvas_phasemap)
		phasemap_layout.addWidget(self.toolbar_phasemap)
		plots_layout.addLayout(phasemap_layout)
		#
		phasefit_layout = QVBoxLayout()
		phasefit_layout.addWidget(self.canvas_phasefit)
		phasefit_layout.addWidget(self.toolbar_phasefit)
		plots_layout.addLayout(phasefit_layout)
		#
		plot3d_layout = QVBoxLayout()
		plot3d_layout.addWidget(self.canvas_3d)
		plot3d_layout.addWidget(self.toolbar_3d)
		plots_layout.addLayout(plot3d_layout)
		#
		outer_layout.addLayout(plots_layout)
		# layout for upper buttons
		upper_buttons_layout = QHBoxLayout()
		self.button_exclude_points = setup_button(self.exclude_points,
											upper_buttons_layout,
											'Exclude Points',
											toggle = True)
		self.button_include_points = setup_button(self.include_points,
											upper_buttons_layout,
											'Include Points',
											toggle = True)
		self.textbox_degree = setup_textbox(
									self.textbox_select,
									upper_buttons_layout, 'Polynomial Degree:',
									initial_value = self.polynomial_degree,
									is_int = True)
		self.textbox_gaussian = setup_textbox(
									self.textbox_select,
									upper_buttons_layout, 'Gaussian Radius:',
									initial_value = self.gaussian_radius,
									is_int = False)
		self.button_fit_phases = setup_button(self.fit_phases,
											upper_buttons_layout,
											'Fit Phases')
		self.button_find_resolution = setup_button(self.find_resolution,
											upper_buttons_layout,
											'Find Resolution')
		outer_layout.addLayout(upper_buttons_layout)
		# layout for lower buttons
		lower_buttons_layout = QHBoxLayout()
		self.button_open_file = setup_button(self.open_file,
											lower_buttons_layout,
											'Open Phases')
		self.button_save_file = setup_button(self.save_file,
											lower_buttons_layout,
											'Save Phases')
		self.textbox_resolution = setup_textbox(
										self.textbox_select,
										lower_buttons_layout, 'Resolution:',
										initial_value = self.resolution,
										is_int = False)
		self.textbox_frequency = setup_textbox(
										self.textbox_select,
										lower_buttons_layout, 'Frequency (Hz):',
										initial_value = self.frequency,
										is_int = False)
		self.button_map_stiffness = setup_button(self.map_stiffness,
											lower_buttons_layout,
											'Map Stiffness')
		self.button_save_stiffness = setup_button(self.save_stiffness,
											lower_buttons_layout,
											'Save Stiffness')
		outer_layout.addLayout(lower_buttons_layout)
		# Set the window's main layout
		self.setLayout(outer_layout)
		if self.phases is not None and len(self.phases)>0:
			self.update_plots()
	
	def textbox_select (self):
		self.polynomial_degree = get_textbox(self.textbox_degree,
											minimum_value = 1,
											maximum_value = 24,
											is_int = True)
		self.gaussian_radius = get_textbox(self.textbox_gaussian,
											minimum_value = 0.,
											maximum_value = 120.,
											is_int = False)
		self.resolution = get_textbox(self.textbox_resolution,
											minimum_value = 40.,
											maximum_value = 240.,
											is_int = False)
		self.frequency = get_textbox(self.textbox_frequency,
											minimum_value = 0.,
											maximum_value = 20.,
											is_int = False)
	
	def open_file (self):
		if self.file_dialog():
			try:
				data = np.genfromtxt(self.file_path, delimiter = ',',
										comments = '#')
				self.points = data[:,0:2]
				self.phases = data[:,2]
				self.shift_origin()
			except:
				print('error opening file')
				self.points = np.empty((0,2), dtype = float)
				self.phases = np.empty(0, dtype = float)
			self.excluded = np.empty(0, dtype = int)
			self.update_plots()
	
	def file_dialog (self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file_name, _ = QFileDialog.getOpenFileName(self,
								'Open Phase Map CSV File', '',
								'CSV Files (*.csv);;' + \
								'All Files (*)',
								options=options)
		if file_name == '':
			return False
		else:
			file_path = Path(file_name)
			if file_path.suffix.lower() == '.csv':
				self.file_path = file_path
				return True
			else:
				self.file_path = None
				return False
	
	def clear_buttons (self):
		self.button_exclude_points.setChecked(False)
		self.button_include_points.setChecked(False)
	
	def exclude_points (self):
		self.clear_buttons()
		if self.mode_select == 'Exclude':
			self.mode_select = 'None'
			if self.click_id_map is not None:
				self.canvas_phasemap.mpl_disconnect(self.click_id_map)
				self.click_id_map = None
			if self.click_id_fit is not None:
				self.canvas_phasemap.mpl_disconnect(self.click_id_fit)
				self.click_id_fit = None
		else:
			self.mode_select = 'Exclude'
			self.button_exclude_points.setChecked(True)
			self.click_id_map = self.canvas_phasemap.mpl_connect(
							'button_press_event', self.on_click_map)
			self.click_id_fit = self.canvas_phasefit.mpl_connect(
							'button_press_event', self.on_click_fit)
	
	def include_points (self):
		self.clear_buttons()
		if self.mode_select == 'Include':
			self.mode_select = 'None'
			if self.click_id_map is not None:
				self.canvas_phasemap.mpl_disconnect(self.click_id_map)
				self.click_id_map = None
			if self.click_id_fit is not None:
				self.canvas_phasemap.mpl_disconnect(self.click_id_fit)
				self.click_id_fit = None
		else:
			self.mode_select = 'Include'
			self.button_include_points.setChecked(True)
			self.click_id_map = self.canvas_phasemap.mpl_connect(
							'button_press_event', self.on_click_map)
			self.click_id_fit = self.canvas_phasefit.mpl_connect(
							'button_press_event', self.on_click_fit)
	
	def on_click_map (self, event):
		try:
			self.position = np.array([event.xdata, event.ydata])
		except:
			return
		if self.points is None or len(self.points) == 0:
			return
		selected_point = np.argmin(np.linalg.norm(
						self.points - self.position, axis=1))
		if self.mode_select == 'Exclude':
			if self.excluded is None:
				self.excluded = np.array([selected_point])
			else:
				self.excluded = np.unique(np.append(self.excluded,
														selected_point))
		elif self.mode_select == 'Include':
			if self.excluded is None or len(self.excluded) == 0:
				pass
			else:
				self.excluded = np.delete(self.excluded, np.where(
											self.excluded == selected_point))
		self.update_plots()
	
	def on_click_fit (self, event):
		try:
			self.position = np.array([event.xdata, event.ydata])
		except:
			return
		if self.points is None or len(self.points) == 0:
			return
		points = np.stack([self.points[:,0], self.phases],axis=-1)
		check_array = points - self.position
		check_array[:,1] *= np.ptp(check_array[:,0])/np.ptp(check_array[:,1])
		selected_point = np.argmin(np.linalg.norm(check_array, axis=1))
		if self.mode_select == 'Exclude':
			if self.excluded is None:
				self.excluded = np.array([selected_point])
			else:
				self.excluded = np.unique(np.append(self.excluded,
														selected_point))
		elif self.mode_select == 'Include':
			if self.excluded is None or len(self.excluded) == 0:
				pass
			else:
				self.excluded = np.delete(self.excluded, np.where(
											self.excluded == selected_point))
		self.update_plots()
	
	def update_plots (self):
		self.canvas_phasemap.update_points(self.points, self.phases,
											self.excluded)
		self.canvas_phasefit.update_points(self.points, self.phases,
											self.excluded, self.fit_function)
		if self.points_adjusted is not None:
			self.canvas_3d.update_points(self.points_adjusted,
										 self.phases_adjusted)
		elif self.excluded is None or len(self.excluded) == 0:
			self.canvas_3d.update_points(self.points, self.phases)
		else:
			self.canvas_3d.update_points(np.delete(self.points,
													self.excluded,
														axis=0),
										 np.delete(self.phases,
													self.excluded,
														axis=0))
	
	def exclude_outer (self):
		pass
	
	def smooth_phases (self, points, phases):
		return smooth_values(points, phases,
					gaussian_radius = self.gaussian_radius)
	
#	def fit_phases (self):
#		if self.excluded is not None and len(self.excluded) > 0:
#			points = np.delete(self.points, self.excluded, axis=0)
#			phases = np.delete(self.phases, self.excluded, axis=0)
#		else:
#			points = self.points.copy()
#			phases = self.phases.copy()
#		coeff = np.polyfit(points[:,0], phases, deg=self.polynomial_degree)
#		self.fit_function = np.poly1d(coeff)
#		phases -= self.fit_function(points[:,0])
#		phases = self.smooth_phases(points, phases)
#		self.points_adjusted = points
#		self.phases_adjusted = phases
#		self.update_plots()
	
	def fit_phases (self):
		if self.excluded is not None and len(self.excluded) > 0:
			points = np.delete(self.points, self.excluded, axis=0)
			phases = np.delete(self.phases, self.excluded, axis=0)
		else:
			points = self.points.copy()
			phases = self.phases.copy()
		#
		def fit_function (points, m, b, *a):
			if len(points.shape) == 2:
				x,y = points.T
			else:
				x = points; y = 0
			return np.polyval(a, x)*(m*y+b)
		#
		y_fit = np.polyfit(points[:,1], phases, 1)
		x_fit = np.polyfit(points[:,0], phases, self.polynomial_degree)
		initial_guess = np.zeros(self.polynomial_degree + 3)
		initial_guess[0:2] = y_fit
		initial_guess[2:] = x_fit
		best_params,_ = curve_fit(fit_function, points, phases, initial_guess)
		# self.fit_function = partial(fit_function, *best_params)
		self.fit_function = lambda points: fit_function(points, *best_params)
		phases -= self.fit_function(points[:,0])
		phases = self.smooth_phases(points, phases)
		self.points_adjusted = points
		self.phases_adjusted = phases
		self.update_plots()
	
	def save_file (self):
		if self.points is None or len(self.points) == 0:
			return False
		if self.points_adjusted is not None:
			array = np.concatenate((self.points_adjusted + self.shift,
									self.phases_adjusted[:,np.newaxis]),
																		axis=1)
		elif self.excluded is None or len(self.excluded) == 0:
			array = np.concatenate((self.points, self.phases[:,np.newaxis]),
																		axis=1)
		else:
			array = np.concatenate((np.delete(self.points + self.shift,
											self.excluded,
												axis=0),
									np.delete(self.phases,
											self.excluded,
												axis=0)[:,np.newaxis]),
																		axis=1)
		header = '# X, Y, Phase'
		output_file = self.file_path.with_suffix(
			'.{0:s}.phase.csv'.format(time.strftime("%Y.%m.%d-%H.%M.%S")))
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file_name, _ = QFileDialog.getSaveFileName(self,
								'Save CSV File',
								str(output_file),
								'All Files (*)',
								options=options)
		if file_name == '':
			return False
		else:
			file_path = Path(file_name)
			if file_path.suffix.lower() == '.csv':
				output_file = file_path
			else:
				output_file = file_path.with_suffix('.csv')
		np.savetxt(output_file, array,
					delimiter = ',', comments = '#', header = header)
		print(f'saved to: {str(output_file):s}')
	
	def get_points (self):
		if self.points is None or len(self.points) == 0:
			return False
		if self.points_adjusted is None:
			if self.excluded is not None and len(self.excluded) > 0:
				points = np.delete(self.points, self.excluded, axis=0)
				phases = np.delete(self.phases, self.excluded, axis=0)
			else:
				points = self.points
				phases = self.phases
		else:
			points = self.points_adjusted
			phases = self.phases_adjusted
		return points, phases
	
	def signal_to_noise (self, points = None, phase = None):
		if points is None or phases is None:
			return False
	
	def find_resolution (self): #TODO: needs fixing.
		if self.points is None or len(self.points) == 0:
			return False
		points, phases = self.get_points()
		res_min = int(np.floor(np.amax([40., self.gaussian_radius])))
		res_max = 200
		resolutions = np.arange(res_min, res_max + 1, 10)
		signal = np.zeros(len(resolutions), dtype = float)
		noise = signal.copy()
		for res_index, resolution in enumerate(resolutions):
			neighbourhoods = get_neighbourhoods(points,
										resolution = resolution)
			enough_points = np.ones(len(phases), dtype = bool)
			hood_signal = np.zeros(len(neighbourhoods), dtype=float)
			hood_noise = hood_signal.copy()
			for hood_index, neighbourhood in enumerate(neighbourhoods):
				if len(neighbourhood) < 6:
					enough_points[hood_index] = False
					continue
				coeff = np.polyfit(points[neighbourhood,1],
									  phases[neighbourhood],
										deg = 1)
				hood_signal[hood_index] = np.abs(coeff[0] * self.resolution)
				hood_noise[hood_index] = np.std(phases[neighbourhood] - \
									points[neighbourhood,1]*coeff[0])
			signal[res_index] = np.mean(hood_signal)
			noise[res_index] = np.mean(hood_noise)
		plt.plot(resolutions, signal/noise)
		plt.show()
	
	# E = 3*rho*(2*pi*f/(d_phi/d_y))^2
	# rho ~ 10^3 kg/m^3
	def map_stiffness (self):
		if self.points is None or len(self.points) == 0:
			return False
		constant = 3*10**3*(2*np.pi*self.frequency)**2/10**12
		points, phases = self.get_points()
		stiffness = np.zeros_like(phases)
		neighbourhoods = get_neighbourhoods(points,
										resolution = self.resolution)
		enough_points = np.ones(len(stiffness), dtype = bool)
		for index, neighbourhood in enumerate(neighbourhoods):
			if len(neighbourhood) < 4:
				enough_points[index] = False
				continue
			coeff = np.polyfit(points[neighbourhood,1],
								phases[neighbourhood],
									deg = 1)
			stiffness[index] = constant / coeff[0]**2
		#stiffness[enough_points] = smooth_values(points[enough_points,:],
		#										stiffness[enough_points],
		#								gaussian_radius = self.gaussian_radius)
		self.stiffness_map = np.concatenate([points[enough_points,:],
										stiffness[enough_points,np.newaxis]],
											axis=-1)
		triangulation = tri.Triangulation(self.stiffness_map[:,0],
										  self.stiffness_map[:,1])
		interpolator = tri.CubicTriInterpolator(triangulation,
										self.stiffness_map[:,2],
										kind='min_E')
		x, y = np.meshgrid(np.linspace(np.amin(self.stiffness_map[:,0]),
									np.amax(self.stiffness_map[:,0]), 240),
							np.linspace(np.amin(self.stiffness_map[:,1]),
									np.amax(self.stiffness_map[:,1]), 240))
		z = interpolator(x,y)
		#plt.contourf(x,y,z)
		plt.tripcolor(triangulation, self.stiffness_map[:,2],
							shading='gouraud')
		plt.colorbar()
		plt.show()
	
	def save_stiffness (self):
		if self.stiffness_map is None:
			return False
		header = '# X(μm), Y(μm), Stiffness (Pa)'
		output_file = self.file_path.with_suffix(
			'.{0:s}.stiffness.csv'.format(time.strftime("%Y.%m.%d-%H.%M.%S")))
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file_name, _ = QFileDialog.getSaveFileName(self,
								'Save CSV File',
								str(output_file),
								'All Files (*)',
								options=options)
		if file_name == '':
			return False
		else:
			file_path = Path(file_name)
			if file_path.suffix.lower() == '.csv':
				output_file = file_path
			else:
				output_file = file_path.with_suffix('.csv')
		np.savetxt(output_file, self.stiffness_map + \
							np.array([[self.shift[0], self.shift[1], 0]]),
					delimiter = ',', comments = '#', header = header)
		print(f'saved to: {str(output_file):s}')

################################################################################

if __name__ == "__main__":
	QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
	QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
	app = QApplication(sys.argv)
	window = Window()
	window.show()
	sys.exit(app.exec_())

################################################################################
# EOF

