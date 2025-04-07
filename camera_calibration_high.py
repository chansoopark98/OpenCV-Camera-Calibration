import cv2
import numpy as np
import glob
import os
import json

class CameraCalibrator:
    def __init__(self, chessboard_size):
        """
        Initialize the camera calibrator with the given chessboard pattern size.
        :param chessboard_size: tuple (columns, rows) of inner chessboard corners.
        """
        self.chessboard_size = chessboard_size  # Use user-provided chessboard size
        # Prepare a single object points template based on the chessboard size.
        # This is a grid of (x,  y, z) coordinates for each chessboard corner in the pattern (z=0 here).
        objp_points = chessboard_size[0] * chessboard_size[1]  # Total number of corners
        self.objp = np.zeros((objp_points, 3), np.float32)
        # Fill the object points with grid coordinates (e.g., (0,0), (1,0), (2,0), ...).
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        # Arrays to store points from all images
        self.objpoints = []  # 3D points in real-world space
        self.imgpoints = []  # 2D points in image plane
        # Store the last image size (used for outlier filtering)
        self.last_image_size = None
        # Collection of images for corner visualization
        self.corner_images = []
        # Store original image paths (for post-processing)
        self.image_paths = []

    def add_calibration_image(self, image, img_path=None, visualize=False):
        """
        Process a single calibration image to find chessboard corners and add to points list.
        :param image: Input image (as a NumPy array) containing the chessboard pattern.
        :param img_path: Image file path (optional).
        :param visualize: Whether to visualize the corner detection results.
        :return: True if corners were found and added, False otherwise.
        """
        self.last_image_size = image.shape[1::-1]  # (width, height)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Preprocess the image - enhance contrast using histogram equalization
        gray = cv2.equalizeHist(gray)
        
        # Apply enhanced chessboard detection parameters
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        found, corners = cv2.findChessboardCorners(gray, self.chessboard_size, flags)
        
        if found and corners is not None:
            # Define criteria for corner refinement (sub-pixel accuracy)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # Refine corner positions to sub-pixel accuracy (only if corners were found)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Visualize corner detection results
            if visualize:
                vis_img = image.copy()
                cv2.drawChessboardCorners(vis_img, self.chessboard_size, corners_refined, found)
                self.corner_images.append(vis_img)
            
            # Add object points and image points to their respective lists
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners_refined)
            
            # Save image path (if provided)
            if img_path:
                self.image_paths.append(img_path)
            
            return True
        else:
            # No corners found in this image
            return False

    def calibrate_camera(self, image_size, use_advanced_model=False):
        """
        Perform camera calibration using all collected object and image points.
        :param image_size: Tuple (width, height) of the images used for calibration.
        :param use_advanced_model: Whether to use an advanced distortion model.
        :return: Dictionary containing calibration error, camera matrix, distortion coefficients, 
                 rotation vectors, and translation vectors.
        """
        if not self.objpoints or not self.imgpoints:
            raise ValueError("No calibration data. Add images with chessboard corners before calibration.")
        
        # Set flags for advanced distortion model
        flags = 0
        if use_advanced_model:
            flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL
        
        # Calibrate the camera using the accumulated object and image points
        calibration_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None, flags=flags
        )
        
        # Print the results in a readable format
        print("Camera calibration completed.")
        print(f"RMS reprojection error: {calibration_error:.4f}")
        print(f"Camera matrix:\n{camera_matrix}")
        print(f"Distortion coefficients:\n{dist_coeffs.ravel()}")  # Flatten for display
        
        # Return the calibration results in a dictionary for further use
        return {
            "reprojection_error": calibration_error,
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "rotation_vectors": rvecs,
            "translation_vectors": tvecs
        }
    
    def filter_outliers(self, threshold=1.0):
        """
        Filters out images with high reprojection errors to improve calibration accuracy.
        :param threshold: Maximum allowable reprojection error.
        :return: Number of filtered images, list of reprojection errors for each image.
        """
        if not self.objpoints or not self.imgpoints:
            return 0, []
        
        # Perform initial calibration with current data
        _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.last_image_size, None, None
        )
        
        # Calculate reprojection errors for each image
        errors = []
        for i in range(len(self.objpoints)):
            proj_imgpoints, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i], mtx, dist
            )
            error = cv2.norm(self.imgpoints[i], proj_imgpoints, cv2.NORM_L2) / len(proj_imgpoints)
            errors.append(error)
        
        # Filter out images with errors above the threshold
        good_indices = [i for i, error in enumerate(errors) if error < threshold]
        
        # Update data with filtered results
        self.objpoints = [self.objpoints[i] for i in good_indices]
        self.imgpoints = [self.imgpoints[i] for i in good_indices]
        
        # Filter stored image paths if available
        if self.image_paths:
            self.image_paths = [self.image_paths[i] for i in good_indices]
        
        # Filter visualization images if available
        if self.corner_images:
            self.corner_images = [self.corner_images[i] for i in good_indices]
        
        return len(errors) - len(good_indices), errors
    
    def calculate_per_view_errors(self, calib_results):
        """
        Calculate the reprojection error for each view (image).
        :param calib_results: Results from the calibrate_camera function.
        :return: List of reprojection errors for each image.
        """
        errors = []
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], 
                calib_results["rotation_vectors"][i], 
                calib_results["translation_vectors"][i], 
                calib_results["camera_matrix"], 
                calib_results["dist_coeffs"]
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            errors.append(error)
        return errors

    def visualize_undistortion(self, image, camera_matrix, dist_coeffs):
        """
        Visualize the image before and after distortion correction.
        :param image: Original image
        :param camera_matrix: Camera matrix
        :param dist_coeffs: Distortion coefficients
        :return: An image showing the original and undistorted images side by side
        """
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 0, (w, h)
        )
        
        # Perform distortion correction
        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # Crop the image based on the ROI
        x, y, w, h = roi
        if w > 0 and h > 0:  # Only crop if ROI is valid
            undistorted_cropped = undistorted[y:y+h, x:x+w]
            # Resize to match the original image size
            undistorted_resized = cv2.resize(undistorted_cropped, (image.shape[1], image.shape[0]))
        else:
            undistorted_resized = undistorted
        
        # Display the original and corrected images side by side
        comparison = np.hstack((image, undistorted_resized))
        
        # Draw grid lines (helps visualize distortion)
        step = 50
        for i in range(0, comparison.shape[0], step):
            cv2.line(comparison, (0, i), (comparison.shape[1], i), (0, 255, 0), 1)
        for i in range(0, comparison.shape[1], step):
            cv2.line(comparison, (i, 0), (i, comparison.shape[0]), (0, 255, 0), 1)
        
        # Draw a boundary line
        cv2.line(comparison, (image.shape[1], 0), (image.shape[1], comparison.shape[0]), (0, 0, 255), 2)
        
        # Add descriptive text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (0, 0, 255), 2)
        cv2.putText(comparison, "Undistorted", (image.shape[1] + 10, 30), font, 1, (0, 0, 255), 2)
        
        return comparison

    def create_calibration_report(self, output_dir, calib_results, errors):
        """
        Generate a calibration results report.
        :param output_dir: Output directory
        :param calib_results: Calibration results
        :param errors: Per-image reprojection errors
        """
        import matplotlib.pyplot as plt
        
        # Generate error graph
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(errors)), errors, color='skyblue')
        plt.axhline(y=np.mean(errors), color='red', linestyle='-', label=f'Mean: {np.mean(errors):.4f}')
        plt.xlabel('Image Number')
        plt.ylabel('Reprojection Error')
        plt.title('Reprojection Errors for Each Calibration Image')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/reprojection_errors.png')
        plt.close()
        
        # Generate histogram
        plt.figure(figsize=(8, 4))
        plt.hist(errors, bins=20, color='skyblue', edgecolor='black')
        plt.axvline(x=np.mean(errors), color='red', linestyle='-', label=f'Mean: {np.mean(errors):.4f}')
        plt.xlabel('Reprojection Error')
        plt.ylabel('Number of Images')
        plt.title('Distribution of Reprojection Errors')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/error_histogram.png')
        plt.close()
        
        # Create text report
        with open(f'{output_dir}/calibration_report.txt', 'w') as f:
            f.write("Camera Calibration Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Chessboard pattern size used: {self.chessboard_size}\n")
            f.write(f"Total images used for calibration: {len(self.objpoints)}\n")
            f.write(f"Overall reprojection error (RMS): {calib_results['reprojection_error']:.6f}\n\n")
            f.write("Camera intrinsic parameters (camera matrix):\n")
            f.write(f"{calib_results['camera_matrix']}\n\n")
            f.write("Distortion coefficients:\n")
            f.write(f"{calib_results['dist_coeffs'].ravel()}\n\n")
            f.write("Per-image reprojection errors:\n")
            for i, error in enumerate(errors):
                img_path = self.image_paths[i] if i < len(self.image_paths) else f"Image {i+1}"
                f.write(f"{img_path}: {error:.6f}\n")

if __name__ == '__main__':
    import tqdm
    import matplotlib.pyplot as plt
    
    # Create output directory for calibration results
    output_dir = './assets/calibration_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load calibration images
    calibration_images = glob.glob('./assets/calibration/*.png')
    if not calibration_images:
        print("No calibration images found. Ensure PNG images are located in './assets/calibration/'.")
        exit(1)
    
    img = cv2.imread(calibration_images[0])
    if img is None:
        print(f"Unable to load image: {calibration_images[0]}")
        exit(1)
    
    img_size = img.shape[1::-1]  # (width, height)
    calibrator = CameraCalibrator(chessboard_size=(9, 6))
    
    # Process images and display progress
    valid_points = 0
    with tqdm.tqdm(calibration_images, desc="Processing images") as pbar:
        for img_path in pbar:
            img = cv2.imread(img_path)
            if img is None:
                continue
            if calibrator.add_calibration_image(img, img_path, visualize=True):
                valid_points += 1
            pbar.set_postfix(Detected=valid_points)
    
    print(f"Chessboard pattern detected in {valid_points} out of {len(calibration_images)} images.")
    
    # Remove outliers based on reprojection error
    threshold = 0.8  # Set threshold (adjust as needed)
    filtered, initial_errors = calibrator.filter_outliers(threshold=threshold)
    print(f"Excluded {filtered} images with reprojection error greater than {threshold}.")
    
    # Perform calibration using advanced distortion model
    calib_results = calibrator.calibrate_camera(img_size, use_advanced_model=True)
    
    # Calculate per-view reprojection errors
    per_view_errors = calibrator.calculate_per_view_errors(calib_results)
    
    # Generate calibration report
    calibrator.create_calibration_report(output_dir, calib_results, per_view_errors)
    
    # Save results in JSON format (structured data)
    with open(f'{output_dir}/calibration_results.json', 'w') as f:
        # Convert NumPy arrays to lists
        json_results = {
            "reprojection_error": float(calib_results["reprojection_error"]),
            "camera_matrix": calib_results["camera_matrix"].tolist(),
            "dist_coeffs": calib_results["dist_coeffs"].ravel().tolist(),
        }
        json.dump(json_results, f, indent=4)
    
    # Save matrices in NumPy format (preserves precision)
    np.save(f'{output_dir}/camera_matrix.npy', calib_results["camera_matrix"])
    np.save(f'{output_dir}/dist_coeffs.npy', calib_results["dist_coeffs"])
    
    # Save results in CSV format (for compatibility)
    import csv
    with open(f'{output_dir}/calibration_results.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Reprojection Error", "Camera Matrix", "Distortion Coefficients"])
        csvwriter.writerow([calib_results["reprojection_error"], 
                           str(calib_results["camera_matrix"]), 
                           str(calib_results["dist_coeffs"].ravel())])
    
    # Apply distortion correction to test images and save visualizations
    for i, img_path in enumerate(calibration_images[:3]):  # Visualize first 3 images only
        test_img = cv2.imread(img_path)
        if test_img is None:
            continue
        undist_comparison = calibrator.visualize_undistortion(
            test_img, calib_results["camera_matrix"], calib_results["dist_coeffs"]
        )
        cv2.imwrite(f'{output_dir}/undistortion_comparison_{i+1}.jpg', undist_comparison)
    
    # Save corner detection visualizations (up to 5 images)
    for i, img in enumerate(calibrator.corner_images[:5]):
        cv2.imwrite(f'{output_dir}/corners_{i+1}.jpg', img)
    
    print(f"Calibration complete! Results saved in the {output_dir} directory.")