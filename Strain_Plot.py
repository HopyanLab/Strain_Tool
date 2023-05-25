#!/usr/bin/env python3

import numpy as np
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from numpy import ma
from pathlib import Path
from zipfile import ZipFile
from PIL import Image
import argparse

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

class zip_tiff_stack:
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
		print(self.shape)
		print(self.channel_names)
		self.physical_pixel_sizes = [1,1,1]
	def get_image_data(self, string, C = 0, T = 0, Z = 0):
		file_index = np.argmax((self.t_list == T) & (self.z_list == Z))
		image_array = np.array(
						Image.open(self.tif_dir/self.name_list[file_index]))
		if string == 'YX':
			return image_array[:,:,C]
		else:
			return image_array

################################################################################

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

def plot_data (data_file):
	data = np.genfromtxt(data_file, delimiter = ',')
	# X, Y, Time, Track ID
	points = data[:,0:2]
#	points[:,1] = -points[:,1]
	time = data[:,2].astype(int)
	cell_id = data[:,3].astype(int)
	initial = points[time == 1]
	centroid = np.mean(initial, axis=0)
#	initial = initial - centroid
	U, s, V = np.linalg.svd(initial)
#	long_dir = V[np.argmax(s)]
#	long_dir /= np.linalg.norm(long_dir)
	final = points[time == np.amax(time)]
#	final = final - centroid
	triangles = Delaunay(initial).simplices
	initial[triangles]
	num_triangles = len(triangles)
	edges_1 = triangles[:,0:2]
	edges_2 = triangles[:,1:]
	edges_3 = triangles[:,0::2]
	edges = np.vstack([edges_1, edges_2, edges_3])
	lengths = np.linalg.norm(initial[edges][:,0,:] -
							 initial[edges][:,1,:], axis=-1)
	mean_length = np.mean(lengths)
	edge_mask = (lengths < mean_length*1.3)
	triangle_mask = np.logical_and(np.logical_and(edge_mask[:num_triangles],
								edge_mask[num_triangles:2*num_triangles]),
								edge_mask[2*num_triangles:3*num_triangles])
	edges = np.sort(edges[edge_mask], axis=1)
	edges = np.unique(edges, axis=0)
	triangles = triangles[triangle_mask]
	num_triangles = len(triangles)
	centroids = np.mean(initial[triangles],axis=1)
	vectors = initial[triangles] - initial[np.roll(triangles, 1, axis=-1)]
	lengths = np.linalg.norm(vectors, axis=-1)
	semi_per = np.sum(lengths, axis=-1)/2
	areas = np.sqrt(semi_per*np.product(
							semi_per[:,np.newaxis] - lengths, axis=-1))
	mean_area = np.mean(areas)
#	vectors_final = initial[triangles] - initial[np.roll(triangles, 1, axis=-1)]
#	lengths_final = np.linalg.norm(vectors_final, axis=-1)
#	semi_per_final = np.sum(lengths_final, axis=-1)/2
#	areas_final = np.sqrt(semi_per_final*np.product(
#						semi_per_final[:,np.newaxis] - lengths_final, axis=-1))
#	mean_area_final = np.mean(areas_final)
	#
	tri_strains = np.zeros((num_triangles,3), dtype=float)
	for index, triangle in enumerate(triangles):
		tri_strains[index] = compute_strain(initial[triangle], final[triangle])
	
	weighted_tri_strains = (tri_strains * areas[:,np.newaxis]) / mean_area
	mean_strains = np.mean(weighted_tri_strains, axis=0)
	strains = np.zeros((len(initial),3), dtype = float)
	for index, point in enumerate(initial):
		mask = (np.sum(triangles == index, axis=-1) != 0)
		strains[index] = np.mean(weighted_tri_strains[mask], axis=0)
	for avg_index in range(1):
		tri_strains = np.mean(strains[triangles], axis=1)
		weighted_tri_strains = (tri_strains * areas[:,np.newaxis]) / mean_area
		mean_strains = np.mean(weighted_tri_strains, axis=0)
		strains = np.zeros((len(initial),3), dtype = float)
		for index, point in enumerate(initial):
			mask = (np.sum(triangles == index, axis=-1) != 0)
			strains[index] = np.mean(weighted_tri_strains[mask], axis=0)
	
	strains -= np.mean(strains, axis=0)[np.newaxis,:]
	max_strain_x = np.amax(strains[:,0])
	min_strain_x = np.amin(strains[:,0])
	cor_x = np.amax((max_strain_x, -min_strain_x))
	max_strain_y = np.amax(strains[:,1])
	min_strain_y = np.amin(strains[:,1])
	cor_y = np.amax((max_strain_y, -min_strain_y))
	#
	cor_x = 0.15
	cor_y = 0.15
	# plots
	zip_file = data_file.with_suffix('')
	while zip_file.suffix:
		zip_file = zip_file.with_suffix('')
	zip_file = zip_file.with_suffix('.zip')
	print(zip_file.name)
	image_file_stack = zip_tiff_stack(zip_file)
	fig = plt.figure()
	colormap = plt.get_cmap('jet')
	ax_x = fig.add_subplot(121, aspect='equal')
	color_vals_x = strains[:,0]
	norm_x = plt.Normalize(-cor_x, cor_x)
	colors = colormap(norm_x(color_vals_x))
	ax_x.imshow(image_file_stack.get_image_data('YX',0,time[0],0), cmap = 'gray')
	for tri_index, triangle in enumerate(triangles):
		grid_x, grid_y = np.mgrid[
				initial[triangle][:,0].min():initial[triangle][:,0].max():20j,
				initial[triangle][:,1].min():initial[triangle][:,1].max():20j]
		grid_z = griddata(initial[triangle], color_vals_x[triangle],
								(grid_x, grid_y), method='cubic')
		ax_x.pcolormesh(grid_x, grid_y, grid_z,
						cmap = colormap,
						vmin = -cor_x,
						vmax = cor_x,
						shading = 'auto',
						zorder = 7)
	line_segments = LineCollection(initial[edges],
									colors = 'gray',
									linestyle = 'solid',
									linewidth = 0.2,
									zorder = 8)
	ax_x.add_collection(line_segments)
	ax_x.scatter(initial[:,0], initial[:,1],
			marker = '.',
			color = colors,
			edgecolors = 'gray',
			linewidths = 0.2,
			zorder = 9)
	ax_x.set_xlim((np.amin(initial[:,0]) - 50,
				   np.amax(initial[:,0]) + 50))
	ax_x.set_ylim((np.amax(initial[:,1]) + 50,
				   np.amin(initial[:,1]) - 50))
	ax_x.set_title('$\epsilon_{xx}$')
	ax_x.tick_params(labelleft=False, left=False,
					 labelbottom=False, bottom=False)
	cbar = plt.colorbar(ScalarMappable(norm = norm_x, cmap = colormap),
						ax=ax_x, shrink=0.7)
#	cbar.ax.axhline(y=mean_strains[0], c='black')
#	original_ticks = list(np.round(np.array(cbar.get_ticks()), decimals=2))
#	cbar.set_ticks(original_ticks + [mean_strains[0]])
#	cbar.set_ticklabels(original_ticks + [''])
	# y strain
	ax_y = fig.add_subplot(122, aspect='equal')
	color_vals_y = strains[:,1]
	norm_y = plt.Normalize(-cor_y, cor_y)
	colors = colormap(norm_y(color_vals_y))
	ax_y.imshow(image_file_stack.get_image_data('YX',0,0,0), cmap = 'gray')
	for tri_index, triangle in enumerate(triangles):
		grid_x, grid_y = np.mgrid[
				initial[triangle][:,0].min():initial[triangle][:,0].max():20j,
				initial[triangle][:,1].min():initial[triangle][:,1].max():20j]
		grid_z = griddata(initial[triangle], color_vals_y[triangle],
								(grid_x, grid_y), method='cubic')
		ax_y.pcolormesh(grid_x, grid_y, grid_z,
						cmap = colormap,
						vmin = -cor_y,
						vmax = cor_y,
						shading = 'auto',
						zorder = 7)
	line_segments = LineCollection(initial[edges],
									colors = 'gray',
									linestyle = 'solid',
									linewidth = 0.2,
									zorder = 8)
	ax_y.add_collection(line_segments)
	ax_y.scatter(initial[:,0], initial[:,1],
			marker = '.',
			color = colors,
			edgecolors = 'gray',
			linewidths = 0.2,
			zorder = 9)
	ax_y.set_xlim((np.amin(initial[:,0]) - 50,
				   np.amax(initial[:,0]) + 50))
	ax_y.set_ylim((np.amax(initial[:,1]) + 50,
				   np.amin(initial[:,1]) - 50))
	ax_y.set_title('$\epsilon_{yy}$')
	ax_y.tick_params(labelleft=False, left=False,
					 labelbottom=False, bottom=False)
	cbar = plt.colorbar(ScalarMappable(norm = norm_y, cmap = colormap),
						ax=ax_y, shrink=0.7)
#	cbar.ax.axhline(y=mean_strains[1], c='black')
#	original_ticks = list(np.round(np.array(cbar.get_ticks()), decimals=2))
#	cbar.set_ticks(original_ticks + [mean_strains[1]])
#	cbar.set_ticklabels(original_ticks + [''])
	fig.suptitle(data_file.with_suffix('').name)
	fig.tight_layout()#(rect=[0,0,1,1])
	fig.savefig(data_file.with_suffix('.png'))
	plt.show()


################################################################################

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
						description = 'Plot strain data.')
	parser.add_argument('data_file',
						type = str,
						help = 'Data file to plot.')
	args = parser.parse_args()
	data_file = Path(args.data_file)
	plot_data(data_file)

################################################################################
# EOF
