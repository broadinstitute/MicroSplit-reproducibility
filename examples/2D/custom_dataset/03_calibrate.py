# **Step 3:** Calibration and Error Estimation


# This code:
# 1. uses saved MMSE predictions, pixel-wise posterior sample standard deviations, and target images we did not use during training to calibrate an error estimator, and then
# 2. generates calibration plots and learn how to interpret them, and
# 3. computes uncertainty maps that can point users at predicted images or parts of predicted images that likely contain unreliable predictions, to finally
# 4. checks how the true error (wrt. to potentially noisy target data) compares to the pixel-wise variability between posterior samples.

import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import pooch

from careamics.lvae_training.calibration import (
    Calibration,
    plot_calibration,
)


# **Step 3.1:** Load and Prepare the Data
dataset_prefix = "user_annotated_blurry_images_"  # TODO: fill in with your own dataset prefix

calib_data_dir = f"calibration_data"
assert os.path.exists(calib_data_dir)

# Load data triplet for Validation data portion...
pred_val_fname = os.path.join(calib_data_dir, f"prediction_{dataset_prefix}Val.tif")
std_val_fname = os.path.join(calib_data_dir, f"std_{dataset_prefix}Val.tif")
target_val_fname = os.path.join(calib_data_dir, f"target_{dataset_prefix}Val.tif")
assert os.path.exists(
    pred_val_fname
), f"File containing MMSE Predictions ({pred_val_fname}) not found. Run the `02_predict.ipynb` on the Validation data and save this file (as instructed in the notebook)."
assert os.path.exists(
    std_val_fname
), f"File containing the pixel-wise std values of posterior samples ({std_val_fname}) not found. Run the `02_predict.ipynb` on the Validation data and save this file (as instructed in the notebook)."
assert os.path.exists(
    target_val_fname
), f"File containing training-data like target data ({target_val_fname}) not found. Run the `02_predict.ipynb` and save this file (as instructed in the notebook)."
pred_val = tifffile.imread(pred_val_fname)
std_val = tifffile.imread(std_val_fname)
target_val = tifffile.imread(target_val_fname)

# Load data triplet for Test data portion...
pred_test_fname = os.path.join(calib_data_dir, f"prediction_{dataset_prefix}Test.tif")
std_test_fname = os.path.join(calib_data_dir, f"std_{dataset_prefix}Test.tif")
target_test_fname = os.path.join(calib_data_dir, f"target_{dataset_prefix}Test.tif")
assert os.path.exists(
    pred_test_fname
), f"File containing MMSE Predictions ({pred_test_fname}) not found. Run the `02_predict.ipynb` on the Test data and save this file (as instructed in the notebook)."
assert os.path.exists(
    std_test_fname
), f"File containing the pixel-wise std values of posterior samples ({std_test_fname}) not found. Run the `02_predict.ipynb` on the Test data and save this file (as instructed in the notebook)."
assert os.path.exists(
    target_test_fname
), f"File containing training-data like target data ({target_test_fname}) not found. Run the `02_predict.ipynb` and save this file (as instructed in the notebook)."
pred_test = tifffile.imread(pred_test_fname)
std_test = tifffile.imread(std_test_fname)
target_test = tifffile.imread(target_test_fname)

if target_test.shape != pred_test.shape:
    assert target_test.shape[1:] == pred_test.shape[1:]
    assert pred_test.shape[0] == 1
    print('You probably enabled reduce_data=True in the `02_predict.ipynb` notebook. This is fine, but you will get calibration on a smaller data.')
    target_test = target_test[:1]


print("✅ if you see no errors, all required data is now loaded!")

# **Step 3.2:** Compute calibration using Validation data
# In order to evaluate how calibrated our network is we must compare the variability between posterior samples at any pixel with the true error of these pixel predictions with respect to high-SNR 'ground truth'. For this we will load the posterior samples we predicted on the ***validation data*** we did not use during model training.


calib = Calibration(
    num_bins=50, # experiment with number of bins?
)
native_stats = calib.compute_stats(pred=pred_val, pred_std=std_val, target=target_val)
count = np.array(native_stats[0]["bin_count"])
count = count / count.sum()

# Compute calibration factors for the channels
calib_factors, factors_array = calib.get_calibrated_factor_for_stdev()
print(f"Calibration factors: {calib_factors}")

# **Step 3.3:** Use calibration setting computed on Validation data and check how calibrated the Test data is
# Once we know how the inter-sample variability (*ie.* RMV) relates to the true error (*ie.* RMSE), we would like to predict the error <nobr>Micro$\mathbb{S}$plit</nobr> makes on our ***test data***, wich we did not use for computing the calibration above (we used the validation data for that).

# Use calibration factor we previously computed on the Validation data...

# ...on the validation data
print("Compute calibration for validation data...", end="")
calib_val = Calibration(num_bins=50)
stats_val = calib_val.compute_stats(
    pred_val, std_val * factors_array["scalar"] + factors_array["offset"], target_val
)
print("✅")

# ...on the test data
print("Compute calibration for test data...", end="")
calib_test = Calibration(num_bins=50)
stats_test = calib_test.compute_stats(
    pred_test, std_test * factors_array["scalar"] + factors_array["offset"], target_test
)
print("✅")

# Finally, plotting the results!
print("Plotting results...")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))

# _,ax = plt.subplots(figsize=(5,5))
# plt.title("Test-data Calibration")
ax[0].set_title("Validation-data Calibration")
plot_calibration(ax[0], stats_val)

# _,ax = plt.subplots(figsize=(5,5))
# plt.title("Test-data Calibration")
ax[1].set_title("Test-data Calibration")
plot_calibration(ax[1], stats_test)

# On the **x-axis** we plot the [root-mean-variance (RMV)](https://pubs.aip.org/aip/aml/article/1/4/046121/2930395/Calibration-in-machine-learning-uncertainty) between the posterior samples we computed in the notebook `02_predict.ipynb` and stored in the `calibration_data` folder.
# On the **y-axis** we plot the [root-mean-squared-error (RMSE)](https://en.wikipedia.org/wiki/Root_mean_square_deviation) between the MMSE predictions we also computed in the predictiion notebook and the (potentially even noisy) target data, also both stored in the `calibration_data` folder.
# If the RMV between posterior samples would perfectly scale with the true error (measured via RMSE), the plots (one for each of the two predicted channels) would perfectly lie on the diagonal $y=x$. Now, this is not the case, but it is not all to far off, which is good news!

# Step 3.4: Plotting estimated true error (RMSE maps)

# Compute the predicted true error
pred_error_test = (
    std_test * factors_array["scalar"].squeeze()[0]
    + factors_array["offset"].squeeze()[0]
)

_, ax = plt.subplots(figsize=(12, 8), ncols=3, nrows=2)
hs = 1200
ws = 250
sz = 400
ax[0, 0].imshow(pred_test[0, hs : hs + sz, ws : ws + sz, 0])
ax[0, 1].imshow(pred_error_test[0, hs : hs + sz, ws : ws + sz, 0], cmap="coolwarm")
ax[0, 2].imshow(target_test[0, hs : hs + sz, ws : ws + sz, 0])

ax[1, 0].imshow(pred_test[0, hs : hs + sz, ws : ws + sz, 1])
ax[1, 1].imshow(pred_error_test[0, hs : hs + sz, ws : ws + sz, 1], cmap="coolwarm")
ax[1, 2].imshow(target_test[0, hs : hs + sz, ws : ws + sz, 1])

ax[0, 0].set_title("Prediction")
ax[0, 1].set_title("Estimated RMSE")
ax[0, 2].set_title("Target")
ax[0, 0].set_ylabel("Channel 1")
ax[1, 0].set_ylabel("Channel 2")

print("Intensity stats for Channel 1:")
print(
    f" > Prediciton (min, max): ({pred_test[...,0].min():6.3f},{pred_test[...,0].max():6.3f})"
)
print(
    f" > Target (min, max):     ({target_test[...,0].min():6.3f},{target_test[...,0].max():6.3f})"
)
print(
    f" > RMSE (min, max):       ({pred_error_test[...,0].min():6.3f},{pred_error_test[...,0].max():6.3f})"
)
print("Intensity stats for Channel 2:")
print(
    f" > Prediciton (min, max): ({pred_test[...,1].min():6.3f},{pred_test[...,1].max():6.3f})"
)
print(
    f" > Target (min, max):     ({target_test[...,1].min():6.3f},{target_test[...,1].max():6.3f})"
)
print(
    f" > RMSE (min, max):       ({pred_error_test[...,1].min():6.3f},{pred_error_test[...,1].max():6.3f})"
)

diff_pred_target = np.abs(pred_test - target_test) ** 2
true_L2_error = np.sqrt(np.mean(diff_pred_target))
estimated_L2_error = np.mean(pred_error_test)

print(f"Esimated error: {estimated_L2_error:8.2f}")
print(f"True error:     {true_L2_error:8.2f}")
print(f"Difference:     {np.abs(estimated_L2_error-true_L2_error)}")

