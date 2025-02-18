import cv2
import numpy as np
import glob

class CameraCalibrator:
    def __init__(self, chessboard_size):
        """
        Initialize the camera calibrator with the given chessboard pattern size.
        :param chessboard_size: tuple (columns, rows) of inner chessboard corners.
        """
        self.chessboard_size = chessboard_size  # use user-provided chessboard size
        # Prepare a single object points template based on the chessboard size.
        # This is a grid of (x,y,z) coordinates for each chessboard corner in the pattern (z=0 here).
        objp_points = chessboard_size[0] * chessboard_size[1]  # total number of corners
        self.objp = np.zeros((objp_points, 3), np.float32)
        # Fill the object points with a grid coordinates (e.g., (0,0), (1,0), (2,0), ...).
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        # Arrays to store points from all images
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane

    def add_calibration_image(self, image):
        """
        Process a single calibration image to find chessboard corners and add to points list.
        :param image: Input image (as a NumPy array) containing the chessboard pattern.
        :return: True if corners were found and added, False otherwise.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners in the image
        found, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        if found and corners is not None:
            # Define criteria for corner refinement (sub-pixel accuracy)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # Refine corner positions to sub-pixel accuracy (only if corners were found)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # Add object points and image points to their respective lists
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners_refined)
            return True
        else:
            # No corners found in this image
            return False

    def calibrate_camera(self, image_size):
        """
        Run camera calibration using all collected object and image points.
        :param image_size: Tuple (width, height) of the images used for calibration.
        :return: Dictionary containing calibration error, camera matrix, distortion coefficients, 
                 rotation vectors, and translation vectors.
        """
        if not self.objpoints or not self.imgpoints:
            raise ValueError("No calibration data. Add images with chessboard corners before calibration.")
        # Calibrate the camera using the accumulated object and image points
        calibration_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None
        )
        # Print the results in a readable format
        print("Camera calibration completed.")
        print(f"RMS re-projection error: {calibration_error:.4f}")
        print(f"Camera matrix:\n{camera_matrix}")
        print(f"Distortion coefficients:\n{dist_coeffs.ravel()}")  # flatten for display
        # Return the calibration results in a dictionary for further use
        return {
            "reprojection_error": calibration_error,
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "rotation_vectors": rvecs,
            "translation_vectors": tvecs
        }

if __name__ == '__main__':
    import tqdm
    import csv
    calibration_images = glob.glob('./assets/calibration/*.png')
    img_size = cv2.imread(calibration_images[0]).shape[:2][::-1]
    calibrator = CameraCalibrator(chessboard_size=(9, 6))
    valid_points = 0
    with tqdm.tqdm(calibration_images, desc="Processing calibration images") as pbar:
        for img_path in pbar:
            img = cv2.imread(img_path)
            if calibrator.add_calibration_image(img):
                valid_points += 1
            pbar.set_postfix(valid_points=valid_points)
        
    calib_results = calibrator.calibrate_camera(img_size)
    print(calib_results)

    # Save the calibration results to a CSV file
    with open('./assets/calibration_results.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Reprojection Error", "Camera Matrix", "Distortion Coefficients"])
        csvwriter.writerow([calib_results["reprojection_error"], calib_results["camera_matrix"], calib_results["dist_coeffs"]])