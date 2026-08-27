# DRO: Doppler-Aware Direct Radar Odometry

This repository contains the code associated with our RSS 2025 paper __"DRO: Doppler-Aware Direct Radar Odometry"__ accessible [here](https://arxiv.org/abs/2504.20339).

Authors: Cedric Le Gentil, Leonardo Brizi, Daniil Lisus, Xinyuan Qiao, Giorgio Grisetti, Timothy D. Barfoot.

**EDIT:** Now this code also contains the implementation of the 3D odometry approach described in the paper __"3DRO: Lidar-level SE(3) Odometry Using a 2D Imaging Radar and a Gyroscope"__ accessible [here](https://arxiv.org/abs/2604.12027).


#### Paper abstract
_Compared to cameras or lidars, millimetre-wave radars have the ability to ‘see’ through thin walls, vegetation, and adversarial weather conditions such as heavy rain, fog, snow, and dust.
In this paper, we propose a novel SE(2) odometry approach for spinning frequency-modulated continuous-wave radars.
Our method performs scan-to-local-map registration of the incoming radar data in a direct manner using all the radar intensity information without the need for feature or point cloud extraction.
The method performs locally continuous trajectory estimation and accounts for both motion and Doppler distortion of the radar scans.
If the radar possesses a specific frequency modulation pattern that makes radial Doppler velocities observable, an additional Doppler-based constraint is
formulated to improve the velocity estimate and enable odometry in geometrically feature-deprived scenarios (e.g., featureless tunnels).
Our method has been validated on over 250 km of on-road data sourced from public datasets (Boreas and MulRan) and collected using our automotive platform.
With the aid of a gyroscope, it outperforms state-of-the-art methods and achieves an average relative translation error of 0.26% on the Boreas leaderboard.
When using data with the appropriate Doppler-enabling frequency modulation pattern, the translation error is reduced to 0.18% in similar environments.
We also benchmarked our algorithm using 1.5 hours of data collected with a mobile robot in off-road environments with various levels of structure to demonstrate its versatility._

#### Results Example

[![Supplementary material attached to the paper](https://img.youtube.com/vi/QYVYUbNziwY/0.jpg)](https://www.youtube.com/watch?v=QYVYUbNziwY )

## Dependencies

All the dependencies should be present in the `requirement.txt`:
```
pip install -r requirements.txt
```
Please let us know if any additional dependency is missing.

## Prepare the configuration file

This repo provides different sample configuration file `config_dro_gd.yaml`, `config_dro_g.yaml`, `config_dro_d.yaml`, and `config_dro.yaml`.
The parameters in these files are compatible with the Boreas dataset.
The older sequences do not have the Doppler-enabled data.
The appropriate configuration is `config_dro_g.yaml`.
New sequences (from 2024) have been collected with a different radar firmware.
In that case, the configuration is `config_dro_gd.yaml` is the one to use.
In both cases, you should copy the sample file and rename it to `config.yaml`.
```
cp config_dro_gd.yaml config.yaml
```
The configuration file should be filled with the correct path to the data and adapt the parameters to your needs.
The parameters should be clear from their names or have a description in the comments.
If you want to use the newer data without the novel Doppler-based velocity constraint, you can base your configuration on `config_dro_g.yaml` but need to change the radar and IMU parameters with the ones from `config_dro_gd.yaml`.


## Run DRO

First, you need to download data from the Boreas dataset [here](https://www.boreas.utias.utoronto.ca/#/download).
_At the time of writing, only one of the new Doppler-enabled sequences is available (2024-12-03-12-54).
More sequences will be released in the future._

This implementation is a simple python script that can be run with the following command:
```
python radar_gp_state_estimation.py
```

It will fetch the configuration file locally and output the 2D results, including
`azimuth_odometry.npz`, in the `output_2d` folder (created if it does not exist).

__Note that the visualisation is enabled by default.
However, this slows down the processing significantly.
You can disable it by setting `display` to `False` in the configuration file.__

## Run 3DRO

The script `odom_3d.py` allows to run the 3D odometry approach described in our ICRA 2026 workshop paper.
You will need to first run the 2D odometry to compute the estimated in-plane velocities.
Then, you can run the 3D odometry with the following command (make sure to change the path to 2D results in the script):
```python
python odom_3d.py
```

## Run evaluation

The script `boreas_eval.py` allows to run the evaluation with the KITTI metric.
You will need to change the path to the Boreas dataset within the script.
You will also need to specify if you want to evaluate the 2D or 3D odometry (`dim` variable in the script).


__WARNING:__ This code might yield slightly different results than in the RSS paper, as it has been refactored for release (some parameters were hard-coded and are now available in the configuration files).

## Citation
If you find this code useful, please consider citing our paper:

**DRO**
```bibtex
@inproceedings{legentil2025dro,
  title={DRO: Doppler-Aware Direct Radar Odometry},
  author={Le Gentil, Cedric and Brizi, Leonardo and Lisus, Daniil and Qiao, Xinyuan and Grisetti, Giorgio and Barfoot, Timothy D.},
  booktitle={Robotics: Science and Systems},
  year={2025},
}
```

**3DRO**
```bibtex
@inproceedings{legentil20263dro,
  title={3DRO: Lidar-level SE(3) Odometry Using a 2D Imaging Radar and a Gyroscope},
  author={Le Gentil, Cedric and Lisus, Daniil and Barfoot, Timothy D.},
  booktitle={ICRA Workshop Radar in Robotics},
  year={2026},
}
```
