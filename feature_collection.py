import numpy as np
import pandas as pd
import os

class FeatureCollection():
	def __init__(self,features_dir,n_segments=15, alpha=0.5):
		self.n_segments = n_segments
		self.alpha = alpha
		self.ld_no_ext = []
		self.ld_csv = []

		ld_with_ext = os.listdir(features_dir)
		ld_no_ext = [x[:-4] for x in ld_with_ext if x.endswith('.csv')]

		for x in ld_no_ext:
			self.ld_csv	.append(os.path.join(features_dir, x + '.csv'))

		self.ld_csv = np.array(self.ld_csv)
		self.ld_no_ext = np.array(self.ld_no_ext)

	def get_gaze_features(self, raw_input):
		"""
		Get gaze features from raw input
		:param raw_input:
		:return:
		"""

		gaze_direction = raw_input[:, 5:11]
		gaze_angle = raw_input[:, 11: 13]
		eye_landmark2D = raw_input[:, 13: 125]
		eye_landmark3D = raw_input[:, 125: 293]
		pose_direction = raw_input[:, 293: 299]

		gaze_direction_std = np.std(gaze_direction, axis=0)
		gaze_direction_mean = np.mean(gaze_direction, axis=0)

		gaze_angle_std = np.std(gaze_angle, axis=0)
		gaze_angle_mean = np.mean(gaze_angle, axis=0)

		eye_landmark2D_shape_0 = np.abs(eye_landmark2D[:, 56 + 9: 56 + 14] - eye_landmark2D[:, 56 + 19: 56 + 14: -1])
		eye_landmark2D_shape_1 = np.abs(eye_landmark2D[:, 56 + 37: 56 + 42] - eye_landmark2D[:, 56 + 47: 56 + 42: -1])
		eye_landmark2D_shape = np.hstack((eye_landmark2D_shape_0, eye_landmark2D_shape_1))
		eye_landmark2D_shape_cov = np.divide(np.std(eye_landmark2D_shape, axis=0),
											 np.mean(eye_landmark2D_shape, axis=0))

		eye_distance = 0.5 * (eye_landmark3D[:, 56 * 2 + 8] + eye_landmark3D[:, 56 * 2 + 42])
		eye_distance_cov = np.std(eye_distance) / np.mean(eye_distance)
		eye_distance_ratio = np.min(eye_distance) / np.max(eye_distance)
		eye_distance_fea = np.array([eye_distance_cov, eye_distance_ratio])

		eye_location2D = []
		for idx in range(4):
			cur_mean = np.mean(eye_landmark2D[:, 28 * idx: 28 * (idx + 1)], axis=1)
			eye_location2D.append(cur_mean)

		eye_location2D = np.vstack(eye_location2D).T
		eye_location2D_mean = np.mean(eye_location2D, axis=0)
		eye_location2D_std = np.std(eye_location2D, axis=0)

		eye_location3D = []
		for idx in range(6):
			cur_mean = np.mean(eye_landmark3D[:, 28 * idx: 28 * (idx + 1)], axis=1)
			eye_location3D.append(cur_mean)
		eye_location3D = np.vstack(eye_location3D).T
		eye_location3D_mean = np.mean(eye_location3D, axis=0)
		eye_location3D_std = np.std(eye_location3D, axis=0)

		pose_direction_mean = np.mean(pose_direction, axis=0)
		pose_direction_std = np.std(pose_direction, axis=0)
		ret_features = np.hstack((gaze_direction_std, gaze_direction_mean, gaze_angle_mean, gaze_angle_std,
								  eye_landmark2D_shape_cov, eye_location2D_mean, eye_location2D_std,
								  eye_location3D_mean,
								  eye_location3D_std, eye_distance_fea, pose_direction_mean, pose_direction_std))

		return ret_features

	def parse_gaze_features(self, txt_path):
		try:
			df = pd.read_csv(txt_path, header=0, sep=',').values
			seq_length = df.shape[0]
			indexing = int((self.n_segments - 1) * (1 - self.alpha))
			k_value = seq_length // (1 + indexing)  # In some case, we will ignore some last frames

			ret = []
			index_st = 0
			for idx in range(self.n_segments):
				index_ed = k_value + int(k_value * (1 - self.alpha) * idx)
				try:
					index_features = self.get_gaze_features(df[index_st: index_ed, :])
				except ValueError:
					index_features = np.zeros(shape=(60,))
				ret.append(index_features)
				index_st = index_ed - int((1 - self.alpha) * k_value)

			ret = np.vstack(ret)
		except:
			print('IO error')
			ret = None
		return ret

	def get_item(self, idx):
		txt_name = self.ld_csv[idx]
		X = self.parse_gaze_features(txt_name)
		return X

	def get_all_data(self):

		ld_features = []
		for ix in range(len(self.ld_csv)):
			z = self.get_item(ix)
			ld_features.append(z)

		return ld_features
